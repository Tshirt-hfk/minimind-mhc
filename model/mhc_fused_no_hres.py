"""mHC（Manifold-Constrained Hyper-Connections）—— H^res 冻结为 I_n 的极速版。

H^res ≡ I_n（不可学），跳过 Sinkhorn-Knopp 与路间 einsum，专注把
RMSNorm + phi GEMM + sigmoid + sublayer_in reduce + write_back 融合到 2 个
Triton kernel（forward / backward 各一）。

对比标准 mHC：
  - 失去 H^res dynamic redistribution（论文 Table 1：~81% mHC 收益来自 H^res）
  - 换来 mHC overhead 减半 → 单 sublayer 加速 5-15% / 整 step 加速 10-20%
  - phi 参数节省 67%（fused_dim: 2n+n² → 2n）

约束（同 mhc_fused.py）：input_norm='global' / h_pre_activation='sigmoid' /
disable_h_pre=disable_h_post=False / disable_h_res=True / n ∈ {1,2,4,8,16}。
"""

import math
import os
import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

from model.mhc_common import (
    max_sram_bytes,
    AUTOTUNE_CFGS_HEAVY, AUTOTUNE_CFGS_LIGHT,
    _USE_BLOCK_PTR_W,   # H1.2: TMA / block_ptr switch
)

# 一键禁用 fused 路径（A-B 对比 / 排障）
_FUSED_MHC_NO_HRES_ENABLED = os.environ.get("MINIMIND_FUSED_MHC_NO_HRES", "1") != "0"


def _is_pow2(n: int) -> bool:
    return n >= 1 and (n & (n - 1)) == 0


# ──────────────────────────────────────────────────────────────────────────────
# Triton kernels
# ──────────────────────────────────────────────────────────────────────────────

if TRITON_AVAILABLE:

    @triton.autotune(configs=AUTOTUNE_CFGS_HEAVY, key=['NC', 'C', 'N', 'FD'])
    @triton.jit
    def _pre_phase_fwd_kernel(
        # ── 输入张量（指针）──
        X_ptr,              # (BL, NC)        x_flat (bf16/fp32)
        Wphi_ptr,           # (NC, FD)        phi.weight 转置后 (NC, FD) row-major
        AlphaPre_ptr,       # (1,)            scalar (fp32)
        BiasPre_ptr,        # (N,)
        AlphaPost_ptr,      # (1,)
        BiasPost_ptr,       # (N,)
        # ── 输出张量 ──
        SublayerIn_ptr,     # (BL, C)
        HPre_ptr,           # (BL, N)
        HPost_ptr,          # (BL, N)
        # ── 反向需要保存的中间张量 ──
        InvRms_ptr,         # (BL,) fp32
        MixesRaw_ptr,       # (BL, FD) fp32 (raw 即 phi 输出，未经 alpha/bias/sigmoid)
        # ── 标量 ──
        post_mult,          # H_post 激活倍数（典型 2.0）
        pre_eps,            # H_pre = sigmoid + pre_eps
        rms_eps,            # RMSNorm 数值稳定项
        # ── 编译期常量 ──
        NC: tl.constexpr,           # n·C
        C: tl.constexpr,            # 单路 hidden_size
        N: tl.constexpr,            # hyper-connection 路数（2 的幂）
        FD: tl.constexpr,           # fused_dim = 2N
        BLOCK_NC: tl.constexpr,     # ≥ NC，是 2 的幂（一次 load 整行 x）
        USE_BLOCK_PTR: tl.constexpr,    # H1.2: block_ptr / TMA path
    ):
        """1 program / 1 token: PRE phase = RMSNorm + phi GEMM + split(pre/post)
        + sigmoid + sublayer_in reduce。W 全 SRAM；x 一次 load 复用。"""
        pid = tl.program_id(0)

        # ── 偏移与 mask ──
        offs_nc = tl.arange(0, BLOCK_NC)        # (BLOCK_NC,)
        mask_nc = offs_nc < NC
        offs_n  = tl.arange(0, N)               # (N,)
        offs_c  = tl.arange(0, C)               # (C,)
        offs_fd = tl.arange(0, FD)              # (FD,)

        # ── Stage 1: load x + 算 inv_rms ──
        x = tl.load(X_ptr + pid * NC + offs_nc,
                    mask=mask_nc, other=0.0).to(tl.float32)  # (BLOCK_NC,)
        sum_sq = tl.sum(x * x)
        inv_rms = tl.rsqrt(sum_sq / NC + rms_eps)             # scalar fp32
        tl.store(InvRms_ptr + pid, inv_rms)
        x_norm = x * inv_rms                                  # (BLOCK_NC,) fp32

        # ── Stage 2: phi GEMM (mixes_raw = x_norm @ W) ──
        # W (NC, FD) 全 load 进 SRAM（FD 很小，nC·FD·2B ≈ 64KB bf16，L2 cache 友好）
        # H1.2: block_ptr 在 Hopper 上 Triton 自动 lower 到 cp.async.bulk.tensor (TMA)
        if USE_BLOCK_PTR:
            w_bptr = tl.make_block_ptr(
                base=Wphi_ptr,
                shape=(NC, FD), strides=(FD, 1),
                offsets=(0, 0), block_shape=(BLOCK_NC, FD),
                order=(1, 0),
            )
            w = tl.load(w_bptr).to(tl.float32)                # (BLOCK_NC, FD)
        else:
            w = tl.load(
                Wphi_ptr + offs_nc[:, None] * FD + offs_fd[None, :],
                mask=mask_nc[:, None], other=0.0,
            ).to(tl.float32)                                  # (BLOCK_NC, FD)
        mixes_raw = tl.sum(x_norm[:, None] * w, axis=0)       # (FD,) fp32
        tl.store(MixesRaw_ptr + pid * FD + offs_fd, mixes_raw)

        # ── Stage 3: split + alpha·raw + bias + sigmoid ──
        # 从 mixes_raw 提取 pre_seg / post_seg
        # offs_fd 中 [0:N] 是 pre，[N:2N] 是 post
        pre_seg  = tl.sum(tl.where(offs_fd[None, :] == offs_n[:, None],
                                   mixes_raw[None, :], 0.0), axis=1)        # (N,)
        post_seg = tl.sum(tl.where(offs_fd[None, :] == (offs_n[:, None] + N),
                                   mixes_raw[None, :], 0.0), axis=1)        # (N,)

        alpha_pre  = tl.load(AlphaPre_ptr ).to(tl.float32)
        alpha_post = tl.load(AlphaPost_ptr).to(tl.float32)
        bias_pre   = tl.load(BiasPre_ptr  + offs_n).to(tl.float32)          # (N,)
        bias_post  = tl.load(BiasPost_ptr + offs_n).to(tl.float32)

        h_pre_raw  = pre_seg  * alpha_pre  + bias_pre                       # (N,) fp32
        h_post_raw = post_seg * alpha_post + bias_post

        h_pre  = tl.sigmoid(h_pre_raw)  + pre_eps                           # (ε, 1+ε)
        h_post = tl.sigmoid(h_post_raw) * post_mult                         # (0, post_mult)

        tl.store(HPre_ptr  + pid * N + offs_n, h_pre.to(HPre_ptr.dtype.element_ty))
        tl.store(HPost_ptr + pid * N + offs_n, h_post.to(HPost_ptr.dtype.element_ty))

        # ── Stage 4: sublayer_in = Σᵢ H_pre[i] · x[i, :] ──
        # P2: 直接 reshape Stage 1 已 load 的 x（要求 BLOCK_NC == NC，下方 launcher 已 assert），
        # 省去第二次 HBM/L2 load
        x_2d = tl.reshape(x, (N, C))                                        # (N, C) view
        sublayer_in = tl.sum(h_pre[:, None] * x_2d, axis=0)                 # (C,) fp32

        offs_2d = offs_n[:, None] * C + offs_c[None, :]                     # (N, C) 仅 store 用
        tl.store(SublayerIn_ptr + pid * C + offs_c,
                 sublayer_in.to(SublayerIn_ptr.dtype.element_ty))


    @triton.autotune(configs=AUTOTUNE_CFGS_HEAVY, key=['NC', 'C', 'N', 'FD'])
    @triton.jit
    def _pre_phase_bwd_kernel(
        # ── forward 保存的张量 ──
        X_ptr,              # (BL, NC)
        Wphi_ptr,           # (NC, FD)
        AlphaPre_ptr,       # (1,)
        BiasPre_ptr,        # (N,)
        AlphaPost_ptr,      # (1,)
        BiasPost_ptr,       # (N,)
        InvRms_ptr,         # (BL,) fp32
        MixesRaw_ptr,       # (BL, FD) fp32  raw phi 输出（未经 alpha/bias/sigmoid）
        # ── 上游梯度 ──
        GradSublayerIn_ptr, # (BL, C)
        GradHPre_ptr,       # (BL, N)        通常全 0（H_pre 不直接被下游用）
        GradHPost_ptr,      # (BL, N)        来自 post_phase backward
        # ── 输出梯度 ──
        GradX_ptr,          # (BL, NC)       grad_x（sublayer_in 反向 + RMSNorm 反向）
        GradMixesRaw_ptr,   # (BL, FD)       供后续 cuBLAS 算 grad_W
        # ── per-token α/β grad（避免 atomic_add 序列化竞争；外部 PyTorch reduce）──
        GradAlphaPrePerTok_ptr,   # (BL,)   fp32
        GradBiasPrePerTok_ptr,    # (BL, N) fp32
        GradAlphaPostPerTok_ptr,  # (BL,)   fp32
        GradBiasPostPerTok_ptr,   # (BL, N) fp32
        # ── 标量 ──
        post_mult,
        pre_eps,
        rms_eps,
        # ── 编译期常量 ──
        NC: tl.constexpr,
        C: tl.constexpr,
        N: tl.constexpr,
        FD: tl.constexpr,
        BLOCK_NC: tl.constexpr,
        USE_BLOCK_PTR: tl.constexpr,    # H1.2: block_ptr / TMA path
    ):
        """1 program / 1 token: PRE phase 全反向（sublayer_in reduce + sigmoid + α/β +
        phi mini-GEMM + RMSNorm 反向）。grad_W 外部 cuBLAS 单次 GEMM 处理。"""
        pid = tl.program_id(0)

        offs_nc = tl.arange(0, BLOCK_NC)
        mask_nc = offs_nc < NC
        offs_n  = tl.arange(0, N)
        offs_c  = tl.arange(0, C)
        offs_fd = tl.arange(0, FD)

        # ── 重新 load forward 中保存的张量 ──
        x        = tl.load(X_ptr + pid * NC + offs_nc, mask=mask_nc, other=0.0).to(tl.float32)
        inv_rms  = tl.load(InvRms_ptr + pid).to(tl.float32)
        mixes_raw= tl.load(MixesRaw_ptr + pid * FD + offs_fd).to(tl.float32)

        alpha_pre  = tl.load(AlphaPre_ptr ).to(tl.float32)
        alpha_post = tl.load(AlphaPost_ptr).to(tl.float32)
        bias_pre   = tl.load(BiasPre_ptr  + offs_n).to(tl.float32)
        bias_post  = tl.load(BiasPost_ptr + offs_n).to(tl.float32)

        # ── 重算 forward 中间量 ──
        x_norm = x * inv_rms                                                # (BLOCK_NC,) fp32

        pre_seg  = tl.sum(tl.where(offs_fd[None, :] == offs_n[:, None],
                                   mixes_raw[None, :], 0.0), axis=1)        # (N,)
        post_seg = tl.sum(tl.where(offs_fd[None, :] == (offs_n[:, None] + N),
                                   mixes_raw[None, :], 0.0), axis=1)

        h_pre_raw  = pre_seg  * alpha_pre  + bias_pre
        h_post_raw = post_seg * alpha_post + bias_post
        sigmoid_pre  = tl.sigmoid(h_pre_raw)                                # 不含 +ε
        sigmoid_post = tl.sigmoid(h_post_raw)
        h_pre  = sigmoid_pre  + pre_eps
        # h_post = sigmoid_post * post_mult  (forward 时计算，bwd 不需要重新)

        # ── 上游梯度 ──
        grad_sub_in   = tl.load(GradSublayerIn_ptr + pid * C + offs_c).to(tl.float32)
        grad_h_pre_up = tl.load(GradHPre_ptr  + pid * N + offs_n).to(tl.float32)
        grad_h_post   = tl.load(GradHPost_ptr + pid * N + offs_n).to(tl.float32)

        # ── 1) sublayer_in reduce 反向 ──
        # P2: 直接 reshape 已 load 的 x，省第二次 HBM/L2 load
        x_2d = tl.reshape(x, (N, C))                                        # (N, C) view
        grad_h_pre_from_reduce = tl.sum(grad_sub_in[None, :] * x_2d, axis=1)# (N,)
        grad_h_pre = grad_h_pre_up + grad_h_pre_from_reduce                 # (N,)

        # ── 2) Sigmoid 反向（H_post 还含 ·post_mult 链）──
        grad_h_pre_raw  = grad_h_pre  * sigmoid_pre  * (1.0 - sigmoid_pre)
        grad_h_post_raw = grad_h_post * sigmoid_post * (1.0 - sigmoid_post) * post_mult

        # ── 3) alpha·raw + bias 反向（P1: per-token store，零原子竞争；外部 sum reduce）──
        grad_pre_seg  = grad_h_pre_raw  * alpha_pre                         # (N,)
        grad_post_seg = grad_h_post_raw * alpha_post                        # (N,)
        grad_alpha_pre_local  = tl.sum(grad_h_pre_raw  * pre_seg)           # scalar
        grad_alpha_post_local = tl.sum(grad_h_post_raw * post_seg)
        tl.store(GradAlphaPrePerTok_ptr  + pid, grad_alpha_pre_local)
        tl.store(GradAlphaPostPerTok_ptr + pid, grad_alpha_post_local)
        tl.store(GradBiasPrePerTok_ptr   + pid * N + offs_n, grad_h_pre_raw)
        tl.store(GradBiasPostPerTok_ptr  + pid * N + offs_n, grad_h_post_raw)

        # ── 4) 组装 grad_mixes_raw (FD,) = concat(grad_pre_seg, grad_post_seg) ──
        grad_mixes_raw = (
            tl.sum(tl.where(offs_n[:, None] == offs_fd[None, :],
                            grad_pre_seg[:, None], 0.0), axis=0)
            + tl.sum(tl.where((offs_n[:, None] + N) == offs_fd[None, :],
                              grad_post_seg[:, None], 0.0), axis=0)
        )                                                                    # (FD,)
        tl.store(GradMixesRaw_ptr + pid * FD + offs_fd,
                 grad_mixes_raw.to(GradMixesRaw_ptr.dtype.element_ty))

        # ── 5) phi 反向 grad_x_norm = grad_mixes_raw @ W.T (per-token mini-GEMM) ──
        # H1.2: 同 fwd，W 通过 block_ptr 走 TMA / async 路径
        if USE_BLOCK_PTR:
            w_bptr = tl.make_block_ptr(
                base=Wphi_ptr,
                shape=(NC, FD), strides=(FD, 1),
                offsets=(0, 0), block_shape=(BLOCK_NC, FD),
                order=(1, 0),
            )
            w = tl.load(w_bptr).to(tl.float32)                              # (BLOCK_NC, FD)
        else:
            w = tl.load(
                Wphi_ptr + offs_nc[:, None] * FD + offs_fd[None, :],
                mask=mask_nc[:, None], other=0.0,
            ).to(tl.float32)                                                # (BLOCK_NC, FD)
        grad_x_norm = tl.sum(grad_mixes_raw[None, :] * w, axis=1)           # (BLOCK_NC,)

        # ── 6) RMSNorm 反向（无 weight 形式）──
        mean_y_gy = tl.sum(x_norm * grad_x_norm) / NC                       # scalar fp32
        grad_x_b = inv_rms * (grad_x_norm - x_norm * mean_y_gy)             # (BLOCK_NC,)

        # ── 7) 合并 grad_x_a（sublayer_in 反向）+ grad_x_b（RMSNorm 反向），单次 store ──
        grad_x_a_2d = h_pre[:, None] * grad_sub_in[None, :]                 # (N, C)
        grad_x_b_2d = tl.reshape(grad_x_b, (N, C))                          # (N, C) view
        grad_x_total_2d = grad_x_b_2d + grad_x_a_2d                         # (N, C)
        offs_2d = offs_n[:, None] * C + offs_c[None, :]                     # 仅 store 用
        tl.store(GradX_ptr + pid * NC + offs_2d,
                 grad_x_total_2d.to(GradX_ptr.dtype.element_ty))


    @triton.autotune(configs=AUTOTUNE_CFGS_LIGHT, key=['N', 'C'])
    @triton.jit
    def _post_phase_fwd_kernel(
        X_ptr,              # (BL, N, C)         residual stream（视为 (BL, N*C) flat）
        HPost_ptr,          # (BL, N)
        FOut_ptr,           # (BL, C)            sublayer 输出
        Out_ptr,            # (BL, N, C)
        N: tl.constexpr,
        C: tl.constexpr,
        BLOCK_C: tl.constexpr,  # ≥ C，2 的幂
    ):
        """1 program / 1 token: out[i,c] = x[i,c] + H_post[i] · F_out[c]
        （H_res = I_n，每路独立累加 write_back）。"""
        pid = tl.program_id(0)

        offs_n = tl.arange(0, N)
        offs_c = tl.arange(0, BLOCK_C)
        mask_c = offs_c < C

        # load F_out (C,)
        f_out = tl.load(FOut_ptr + pid * C + offs_c, mask=mask_c, other=0.0).to(tl.float32)
        # load H_post (N,)
        h_post = tl.load(HPost_ptr + pid * N + offs_n).to(tl.float32)
        # load x (N, C)
        offs_2d = offs_n[:, None] * C + offs_c[None, :]
        mask_2d = mask_c[None, :]
        x = tl.load(X_ptr + pid * N * C + offs_2d, mask=mask_2d, other=0.0).to(tl.float32)

        # out[i,c] = x[i,c] + H_post[i] * F_out[c]
        out = x + h_post[:, None] * f_out[None, :]

        tl.store(Out_ptr + pid * N * C + offs_2d,
                 out.to(Out_ptr.dtype.element_ty), mask=mask_2d)


    @triton.autotune(configs=AUTOTUNE_CFGS_LIGHT, key=['N', 'C'])
    @triton.jit
    def _post_phase_bwd_kernel(
        GradOut_ptr,        # (BL, N, C)
        HPost_ptr,          # (BL, N)
        FOut_ptr,           # (BL, C)
        # 输出
        GradX_ptr,          # (BL, N, C)        = grad_out  (恒等，因 out = x + ...)
        GradHPost_ptr,      # (BL, N)
        GradFOut_ptr,       # (BL, C)
        N: tl.constexpr,
        C: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        """1 program / 1 token: grad_x = grad_out (直通); grad_H_post / grad_F_out
        分别由 g_out 在 c / i 维 reduce 得到。"""
        pid = tl.program_id(0)

        offs_n = tl.arange(0, N)
        offs_c = tl.arange(0, BLOCK_C)
        mask_c = offs_c < C

        # load grad_out (N, C)
        offs_2d = offs_n[:, None] * C + offs_c[None, :]
        mask_2d = mask_c[None, :]
        g_out = tl.load(GradOut_ptr + pid * N * C + offs_2d, mask=mask_2d, other=0.0).to(tl.float32)
        # load H_post (N,), F_out (C,)
        h_post = tl.load(HPost_ptr + pid * N + offs_n).to(tl.float32)
        f_out  = tl.load(FOut_ptr  + pid * C + offs_c, mask=mask_c, other=0.0).to(tl.float32)

        # grad_x = grad_out（直通）
        tl.store(GradX_ptr + pid * N * C + offs_2d,
                 g_out.to(GradX_ptr.dtype.element_ty), mask=mask_2d)

        # grad_H_post[i] = Σ_c grad_out[i,c] · F_out[c]
        grad_h_post = tl.sum(g_out * f_out[None, :], axis=1)            # (N,)
        tl.store(GradHPost_ptr + pid * N + offs_n,
                 grad_h_post.to(GradHPost_ptr.dtype.element_ty))

        # grad_F_out[c] = Σ_i grad_out[i,c] · H_post[i]
        grad_f_out = tl.sum(g_out * h_post[:, None], axis=0)            # (C,)
        tl.store(GradFOut_ptr + pid * C + offs_c,
                 grad_f_out.to(GradFOut_ptr.dtype.element_ty), mask=mask_c)


def _pre_phase_ref(x_flat, w_phi, alpha_pre, bias_pre, alpha_post, bias_post,
                   post_mult, pre_eps, rms_eps, N, C):
    """PyTorch ref（fallback / 比对）：PRE phase。"""
    B, L, NC = x_flat.shape
    assert NC == N * C
    # RMSNorm（fp32 累加平方和）
    x_f = x_flat.float()
    inv_rms = x_f.pow(2).mean(-1, keepdim=True).add(rms_eps).rsqrt()    # (B,L,1)
    x_norm = (x_f * inv_rms).to(x_flat.dtype)                            # (B,L,NC)
    # phi
    mixes_raw = x_norm @ w_phi                                           # (B,L,FD)
    FD = mixes_raw.shape[-1]
    assert FD == 2 * N
    pre_seg  = mixes_raw[..., :N]
    post_seg = mixes_raw[..., N:2*N]
    h_pre_raw  = pre_seg  * alpha_pre  + bias_pre
    h_post_raw = post_seg * alpha_post + bias_post
    h_pre  = torch.sigmoid(h_pre_raw)  + pre_eps
    h_post = torch.sigmoid(h_post_raw) * post_mult
    # sublayer_in
    x = x_flat.view(B, L, N, C)
    sublayer_in = (h_pre.unsqueeze(-1) * x).sum(dim=-2)                  # (B,L,C)
    return sublayer_in, h_pre, h_post, inv_rms.squeeze(-1).float(), mixes_raw.float()


def _post_phase_ref(x_flat, h_post, f_out, N, C):
    """PyTorch ref（fallback / 比对）：POST phase。"""
    B, L, NC = x_flat.shape
    x = x_flat.view(B, L, N, C)
    out = x + h_post.unsqueeze(-1) * f_out.unsqueeze(-2)                 # (B,L,N,C)
    return out.reshape(B, L, NC)


class _PrePhaseFn(torch.autograd.Function):
    """PRE phase fused fwd+bwd → (sublayer_in, h_pre, h_post)。"""

    @staticmethod
    def forward(ctx, x_flat, w_phi, alpha_pre, bias_pre, alpha_post, bias_post,
                post_mult, pre_eps, rms_eps, n, c):
        # w_phi: (n·c, 2n) row-major
        assert x_flat.is_cuda and TRITON_AVAILABLE, "fused 路径需要 CUDA + Triton"
        assert x_flat.dim() == 3
        assert w_phi.shape == (n * c, 2 * n)

        B, L, NC = x_flat.shape
        BL = B * L
        FD = 2 * n

        x_flat_c = x_flat.contiguous().view(BL, NC)
        w_phi_c = w_phi.contiguous()

        # 输出
        sublayer_in = torch.empty(BL, c, dtype=x_flat.dtype, device=x_flat.device)
        h_pre       = torch.empty(BL, n, dtype=x_flat.dtype, device=x_flat.device)
        h_post      = torch.empty(BL, n, dtype=x_flat.dtype, device=x_flat.device)
        # 反向需要保存
        inv_rms     = torch.empty(BL,     dtype=torch.float32, device=x_flat.device)
        mixes_raw   = torch.empty(BL, FD, dtype=torch.float32, device=x_flat.device)

        BLOCK_NC = triton.next_power_of_2(NC)
        # n·C 必须是 2 的幂（bwd 中 tl.reshape(grad_x_b, (N,C)) 要求 BLOCK_NC == N*C）
        assert BLOCK_NC == NC, (
            f"n·dim={NC} 必须是 2 的幂（当前 next_pow2={BLOCK_NC}）；"
            f"实际项目里 n ∈ {{1,2,4,8,16}}、dim ∈ {{256,512,1024,2048}} 都满足"
        )
        # device-aware SRAM 上限（H800 228KB / A100 192KB / consumer ~100KB）
        sram_bytes = BLOCK_NC * 4 + BLOCK_NC * FD * 4
        sram_limit = max_sram_bytes(x_flat.device)
        assert sram_bytes <= sram_limit, (
            f"BLOCK_NC={BLOCK_NC} FD={FD} 需要 {sram_bytes/1024:.0f}KB SRAM，"
            f"超出当前 GPU 上限 {sram_limit/1024:.0f}KB"
        )

        alpha_pre_f  = alpha_pre.float().reshape(1)
        alpha_post_f = alpha_post.float().reshape(1)

        _pre_phase_fwd_kernel[(BL,)](
            x_flat_c, w_phi_c,
            alpha_pre_f, bias_pre.float(),
            alpha_post_f, bias_post.float(),
            sublayer_in, h_pre, h_post,
            inv_rms, mixes_raw,
            float(post_mult), float(pre_eps), float(rms_eps),
            NC=NC, C=c, N=n, FD=FD, BLOCK_NC=BLOCK_NC,
            USE_BLOCK_PTR=_USE_BLOCK_PTR_W,
        )

        # save for backward
        ctx.save_for_backward(x_flat_c, w_phi_c, alpha_pre_f, bias_pre,
                              alpha_post_f, bias_post, inv_rms, mixes_raw)
        ctx.post_mult = float(post_mult)
        ctx.pre_eps   = float(pre_eps)
        ctx.rms_eps   = float(rms_eps)
        ctx.n = n
        ctx.c = c
        ctx.shape = (B, L, NC)

        return (sublayer_in.view(B, L, c),
                h_pre.view(B, L, n),
                h_post.view(B, L, n))

    @staticmethod
    def backward(ctx, grad_sublayer_in, grad_h_pre, grad_h_post):
        (x_flat_c, w_phi_c, alpha_pre_f, bias_pre,
         alpha_post_f, bias_post, inv_rms, mixes_raw) = ctx.saved_tensors

        B, L, NC = ctx.shape
        BL = B * L
        n  = ctx.n
        c  = ctx.c
        FD = 2 * n

        # 上游 grad 可能为 None（下游 loss 未使用对应输出）
        if grad_sublayer_in is None:
            grad_sub_in_c = torch.zeros(BL, c, dtype=x_flat_c.dtype, device=x_flat_c.device)
        else:
            grad_sub_in_c = grad_sublayer_in.contiguous().view(BL, c)
        if grad_h_pre is None:
            grad_h_pre_c = torch.zeros(BL, n, dtype=x_flat_c.dtype, device=x_flat_c.device)
        else:
            grad_h_pre_c = grad_h_pre.contiguous().view(BL, n)
        if grad_h_post is None:
            grad_h_post_c = torch.zeros(BL, n, dtype=x_flat_c.dtype, device=x_flat_c.device)
        else:
            grad_h_post_c = grad_h_post.contiguous().view(BL, n)

        grad_x         = torch.empty(BL, NC, dtype=x_flat_c.dtype, device=x_flat_c.device)
        grad_mixes_raw = torch.empty(BL, FD, dtype=torch.float32, device=x_flat_c.device)
        # per-token α/β grad（P1：避免 atomic_add 序列化竞争；下方 PyTorch reduce）
        grad_alpha_pre_per_tok  = torch.empty(BL,    dtype=torch.float32, device=x_flat_c.device)
        grad_bias_pre_per_tok   = torch.empty(BL, n, dtype=torch.float32, device=x_flat_c.device)
        grad_alpha_post_per_tok = torch.empty(BL,    dtype=torch.float32, device=x_flat_c.device)
        grad_bias_post_per_tok  = torch.empty(BL, n, dtype=torch.float32, device=x_flat_c.device)

        BLOCK_NC = triton.next_power_of_2(NC)

        _pre_phase_bwd_kernel[(BL,)](
            x_flat_c, w_phi_c,
            alpha_pre_f, bias_pre.float(),
            alpha_post_f, bias_post.float(),
            inv_rms, mixes_raw,
            grad_sub_in_c, grad_h_pre_c, grad_h_post_c,
            grad_x, grad_mixes_raw,
            grad_alpha_pre_per_tok, grad_bias_pre_per_tok,
            grad_alpha_post_per_tok, grad_bias_post_per_tok,
            ctx.post_mult, ctx.pre_eps, ctx.rms_eps,
            NC=NC, C=c, N=n, FD=FD, BLOCK_NC=BLOCK_NC,
            USE_BLOCK_PTR=_USE_BLOCK_PTR_W,
        )

        # grad_W = x_norm.T @ grad_mixes_raw （单次 cuBLAS GEMM；x_norm recompute）
        x_norm = x_flat_c.float() * inv_rms.unsqueeze(-1)                   # (BL, NC) fp32
        grad_w_phi = (x_norm.T @ grad_mixes_raw).to(w_phi_c.dtype)          # (NC, FD)

        # α/β grad：单次 PyTorch reduce 替代 BL 个 atomic_add（零原子竞争）
        # 注意：alpha_pre/post 在 MHCConnection_FusedNoHres 中是 torch.tensor(scalar) 0-dim，
        # 所以 grad 也必须返回 0-dim（用 .squeeze() 把 (1,) 压成 ()）
        grad_alpha_pre  = grad_alpha_pre_per_tok.sum()                       # 0-dim
        grad_alpha_post = grad_alpha_post_per_tok.sum()                      # 0-dim
        grad_bias_pre_t  = grad_bias_pre_per_tok.sum(dim=0)                  # (n,)
        grad_bias_post_t = grad_bias_post_per_tok.sum(dim=0)                 # (n,)

        return (grad_x.view(B, L, NC), grad_w_phi,
                grad_alpha_pre.to(alpha_pre_f.dtype),  grad_bias_pre_t.to(bias_pre.dtype),
                grad_alpha_post.to(alpha_post_f.dtype), grad_bias_post_t.to(bias_post.dtype),
                None, None, None, None, None)


class _PostPhaseFn(torch.autograd.Function):
    """POST phase fused fwd+bwd: out = x + H_post · F_out。"""

    @staticmethod
    def forward(ctx, x_flat, h_post, f_out, n, c):
        assert x_flat.is_cuda and TRITON_AVAILABLE
        B, L, NC = x_flat.shape
        BL = B * L

        x_c = x_flat.contiguous().view(BL, n * c)
        h_c = h_post.contiguous().view(BL, n)
        f_c = f_out.contiguous().view(BL, c)
        out = torch.empty_like(x_c)

        BLOCK_C = triton.next_power_of_2(c)
        _post_phase_fwd_kernel[(BL,)](
            x_c, h_c, f_c, out,
            N=n, C=c, BLOCK_C=BLOCK_C,
        )

        ctx.save_for_backward(h_c, f_c)
        ctx.n = n
        ctx.c = c
        ctx.shape = (B, L, NC)
        return out.view(B, L, NC)

    @staticmethod
    def backward(ctx, grad_out):
        h_c, f_c = ctx.saved_tensors
        B, L, NC = ctx.shape
        BL = B * L
        n  = ctx.n
        c  = ctx.c

        grad_out_c = grad_out.contiguous().view(BL, n * c)
        grad_x      = torch.empty_like(grad_out_c)
        grad_h_post = torch.empty_like(h_c)
        grad_f_out  = torch.empty_like(f_c)

        BLOCK_C = triton.next_power_of_2(c)
        _post_phase_bwd_kernel[(BL,)](
            grad_out_c, h_c, f_c,
            grad_x, grad_h_post, grad_f_out,
            N=n, C=c, BLOCK_C=BLOCK_C,
        )

        return (grad_x.view(B, L, NC),
                grad_h_post.view(B, L, n),
                grad_f_out.view(B, L, c),
                None, None)


class MHCConnection_FusedNoHres(nn.Module):
    """mHC 单 sublayer 包装：H^res = I_n（不可学），其余全 Triton fused。
    固化配置：input_norm='global' / h_pre_activation='sigmoid' / disable_h_res=True。
    """

    def __init__(self, n: int, dim: int, alpha_init: float = 0.01,
                 post_mult_value: float = 2.0, eps: float = 1e-6,
                 pre_eps: float = 1e-6):
        super().__init__()
        assert n >= 1 and isinstance(n, int)
        assert _is_pow2(n) and n <= 16, f"n 必须 ∈ {{1,2,4,8,16}}（Triton kernel 约束），得到 {n}"
        assert post_mult_value > 1.0
        self.n = n
        self.dim = dim
        self.nC = n * dim
        self.fused_dim = 2 * n            # disable_h_res 时只有 pre+post 两段
        self.post_mult_value = post_mult_value
        self.pre_eps = pre_eps
        self.eps = eps

        # phi: (n·C, 2n) row-major 直接存储（便于 Triton 行主序索引；保留 autograd
        # 梯度连接，避免 .t().contiguous() 切断）。
        self.phi_weight = nn.Parameter(torch.empty(self.nC, self.fused_dim))
        nn.init.kaiming_uniform_(self.phi_weight, a=math.sqrt(5))

        # alpha (scalar) + bias 分段独立
        self.alpha_pre  = nn.Parameter(torch.tensor(alpha_init))
        self.bias_pre   = nn.Parameter(torch.zeros(n))
        self.alpha_post = nn.Parameter(torch.tensor(alpha_init))
        self.bias_post  = nn.Parameter(torch.zeros(n))

    def extra_repr(self) -> str:
        return (f'n={self.n}, dim={self.dim}, fused_dim={self.fused_dim}, '
                f'post_mult_value={self.post_mult_value}, '
                f'input_norm=\'global\'(fused), h_pre_activation=\'sigmoid\'(fused), '
                f'disable_h_res=True(file-level)')

    def forward(self, x_flat: torch.Tensor, sublayer_fn) -> torch.Tensor:
        """单 sublayer mHC 残差更新（H^res = I_n 快路径）：PRE(Triton) → sublayer F → POST(Triton)。"""
        if _FUSED_MHC_NO_HRES_ENABLED and TRITON_AVAILABLE and x_flat.is_cuda:
            sublayer_in, _h_pre, h_post = _PrePhaseFn.apply(
                x_flat, self.phi_weight, self.alpha_pre, self.bias_pre,
                self.alpha_post, self.bias_post,
                self.post_mult_value, self.pre_eps, self.eps,
                self.n, self.dim,
            )
            f_out = sublayer_fn(sublayer_in)
            return _PostPhaseFn.apply(x_flat, h_post, f_out, self.n, self.dim)

        return self._forward_ref(x_flat, sublayer_fn)

    def _forward_ref(self, x_flat: torch.Tensor, sublayer_fn) -> torch.Tensor:
        """PyTorch ref（fallback / 比对）：与 fused 路径数学等价。"""
        B, L, _ = x_flat.shape
        n, C = self.n, self.dim

        x = x_flat.view(B, L, n, C)

        # RMSNorm(nC) + phi
        x_f = x_flat.float()
        inv_rms = x_f.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        x_norm = (x_f * inv_rms).to(x_flat.dtype)
        mixes_raw = x_norm @ self.phi_weight                              # (B, L, 2n)

        pre_seg  = mixes_raw[..., :n]
        post_seg = mixes_raw[..., n:2*n]
        h_pre_raw  = pre_seg  * self.alpha_pre  + self.bias_pre
        h_post_raw = post_seg * self.alpha_post + self.bias_post
        h_pre  = torch.sigmoid(h_pre_raw)  + self.pre_eps
        h_post = torch.sigmoid(h_post_raw) * self.post_mult_value

        sublayer_in = (h_pre.unsqueeze(-1) * x).sum(dim=-2)              # (B, L, C)
        f_out = sublayer_fn(sublayer_in)                                  # (B, L, C)
        write_back = h_post.unsqueeze(-1) * f_out.unsqueeze(-2)           # (B, L, n, C)
        out = x + write_back                                              # disable_h_res 快路径
        return out.reshape(B, L, self.nC)


# ──────────────────────────────────────────────────────────────────────────────
# 正确性测试 + benchmark（python -m model.mhc_fused_no_hres）
# ──────────────────────────────────────────────────────────────────────────────

def _check(name, ref, got, atol, rtol):
    diff = (ref.float() - got.float()).abs().max().item()
    rel = diff / max(ref.float().abs().max().item(), 1e-8)
    flag = "✓" if (diff < atol or rel < rtol) else "✗"
    print(f"  {flag} {name:>22s}: max|diff|={diff:.3e}  rel={rel:.3e}")
    assert diff < atol or rel < rtol, f"{name} 不对齐: abs={diff}, rel={rel}"


def _test_pre_post_phase(B, L, n, C, dtype, atol, rtol):
    torch.manual_seed(0)
    device = 'cuda'
    NC = n * C
    FD = 2 * n

    # 输入
    x = torch.randn(B, L, NC, device=device, dtype=dtype, requires_grad=True)
    f_out = torch.randn(B, L, C, device=device, dtype=dtype, requires_grad=True)

    # phi.weight (FD, nC) Kaiming
    phi_w_t = torch.empty(FD, NC, device=device, dtype=dtype)
    nn.init.kaiming_uniform_(phi_w_t, a=math.sqrt(5))
    w_phi = phi_w_t.t().contiguous().detach().requires_grad_(True)        # (NC, FD)

    alpha_pre  = torch.tensor(0.01, device=device, requires_grad=True)
    bias_pre   = torch.zeros(n, device=device, requires_grad=True)
    alpha_post = torch.tensor(0.01, device=device, requires_grad=True)
    bias_post  = torch.zeros(n, device=device, requires_grad=True)

    post_mult, pre_eps, rms_eps = 2.0, 1e-6, 1e-6

    # ── ref forward ──
    sub_in_ref, h_pre_ref, h_post_ref, inv_rms_ref, _mixes_ref = _pre_phase_ref(
        x, w_phi, alpha_pre, bias_pre, alpha_post, bias_post,
        post_mult, pre_eps, rms_eps, n, C
    )

    # ── fused forward ──
    x2 = x.detach().clone().requires_grad_(True)
    w2 = w_phi.detach().clone().requires_grad_(True)
    ap = alpha_pre.detach().clone().requires_grad_(True)
    bp = bias_pre.detach().clone().requires_grad_(True)
    ao = alpha_post.detach().clone().requires_grad_(True)
    bo = bias_post.detach().clone().requires_grad_(True)
    sub_in_fused, h_pre_fused, h_post_fused = _PrePhaseFn.apply(
        x2, w2, ap, bp, ao, bo, post_mult, pre_eps, rms_eps, n, C
    )

    print(f"[PRE phase B={B} L={L} n={n} C={C} dtype={str(dtype).replace('torch.','')}]")
    _check("sublayer_in", sub_in_ref, sub_in_fused, atol, rtol)
    _check("H_pre",       h_pre_ref,  h_pre_fused,  atol, rtol)
    _check("H_post",      h_post_ref, h_post_fused, atol, rtol)

    # ── ref backward ──
    g_sub_in = torch.randn_like(sub_in_ref)
    g_h_post = torch.randn_like(h_post_ref)
    (sub_in_ref * g_sub_in).sum().backward(retain_graph=True)
    (h_post_ref * g_h_post).sum().backward()

    # ── fused backward ──
    (sub_in_fused * g_sub_in).sum().backward(retain_graph=True)
    (h_post_fused * g_h_post).sum().backward()

    _check("grad_x",       x.grad,          x2.grad, atol, rtol)
    _check("grad_W_phi",   w_phi.grad,      w2.grad, atol*5, rtol*5)
    _check("grad_alpha_pre",  alpha_pre.grad,  ap.grad,  atol, rtol)
    _check("grad_bias_pre",   bias_pre.grad,   bp.grad,  atol, rtol)
    _check("grad_alpha_post", alpha_post.grad, ao.grad,  atol, rtol)
    _check("grad_bias_post",  bias_post.grad,  bo.grad,  atol, rtol)
    print()


def _test_post_phase(B, L, n, C, dtype, atol, rtol):
    torch.manual_seed(0)
    device = 'cuda'
    NC = n * C

    x = torch.randn(B, L, NC, device=device, dtype=dtype, requires_grad=True)
    h_post = torch.rand(B, L, n, device=device, dtype=dtype, requires_grad=True) * 2.0
    f_out = torch.randn(B, L, C, device=device, dtype=dtype, requires_grad=True)

    # ref
    out_ref = _post_phase_ref(x, h_post, f_out, n, C)

    # fused
    x2 = x.detach().clone().requires_grad_(True)
    h2 = h_post.detach().clone().requires_grad_(True)
    f2 = f_out.detach().clone().requires_grad_(True)
    out_fused = _PostPhaseFn.apply(x2, h2, f2, n, C)

    print(f"[POST phase B={B} L={L} n={n} C={C} dtype={str(dtype).replace('torch.','')}]")
    _check("out", out_ref, out_fused, atol, rtol)

    g_out = torch.randn_like(out_ref)
    out_ref.backward(g_out)
    out_fused.backward(g_out)
    _check("grad_x",      x.grad,      x2.grad, atol, rtol)
    _check("grad_h_post", h_post.grad, h2.grad, atol, rtol)
    _check("grad_f_out",  f_out.grad,  f2.grad, atol, rtol)
    print()


def _bench(B, L, n, C, dtype, warmup=10, runs=50):
    """对比 ref / fused 路径的 forward+backward 端到端耗时。"""
    device = 'cuda'
    NC = n * C

    # 装配一个最简 sublayer_fn（线性层模拟 attn/mlp 计算量，但与 mHC overhead 解耦）
    sublayer_fn = nn.Sequential(
        nn.Linear(C, 4 * C, bias=False),
        nn.SiLU(),
        nn.Linear(4 * C, C, bias=False),
    ).to(device=device, dtype=dtype)

    mhc = MHCConnection_FusedNoHres(n, C).to(device=device, dtype=dtype)

    x_template = torch.randn(B, L, NC, device=device, dtype=dtype)

    def _one_step(use_fused: bool):
        # 通过环境变量切换 dispatch
        global _FUSED_MHC_NO_HRES_ENABLED
        prev = _FUSED_MHC_NO_HRES_ENABLED
        _FUSED_MHC_NO_HRES_ENABLED = use_fused
        try:
            x = x_template.clone().requires_grad_(True)
            out = mhc(x, sublayer_fn)
            loss = out.float().pow(2).mean()
            loss.backward()
        finally:
            _FUSED_MHC_NO_HRES_ENABLED = prev

    # warmup
    for _ in range(warmup):
        _one_step(True)
        _one_step(False)
        for p in mhc.parameters():
            if p.grad is not None: p.grad = None
        for p in sublayer_fn.parameters():
            if p.grad is not None: p.grad = None

    # bench fused
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(runs):
        _one_step(True)
        for p in mhc.parameters():
            if p.grad is not None: p.grad = None
        for p in sublayer_fn.parameters():
            if p.grad is not None: p.grad = None
    end.record(); torch.cuda.synchronize()
    t_fused = start.elapsed_time(end) / runs

    # bench ref
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(runs):
        _one_step(False)
        for p in mhc.parameters():
            if p.grad is not None: p.grad = None
        for p in sublayer_fn.parameters():
            if p.grad is not None: p.grad = None
    end.record(); torch.cuda.synchronize()
    t_ref = start.elapsed_time(end) / runs

    speedup = t_ref / t_fused
    print(f"[bench B={B} L={L} n={n} C={C} dtype={str(dtype).replace('torch.','')}]"
          f"  ref={t_ref:.3f}ms  fused={t_fused:.3f}ms  speedup={speedup:.2f}x")


if __name__ == "__main__":
    if not (torch.cuda.is_available() and TRITON_AVAILABLE):
        print("[skip] 需要 CUDA + Triton")
        raise SystemExit(0)

    print("=" * 78)
    print("mhc_fused_no_hres.py: 数值正确性测试")
    print("=" * 78)

    # POST phase（简单算子优先验证）
    _test_post_phase(B=2, L=64,  n=4, C=128, dtype=torch.float32, atol=1e-5, rtol=1e-5)
    _test_post_phase(B=2, L=64,  n=4, C=128, dtype=torch.bfloat16, atol=5e-2, rtol=5e-3)
    _test_post_phase(B=1, L=512, n=8, C=256, dtype=torch.bfloat16, atol=5e-2, rtol=5e-3)

    # PRE phase（含 RMSNorm + phi GEMM + sigmoid + reduce 全链路）
    _test_pre_post_phase(B=2, L=64,  n=4, C=128, dtype=torch.float32, atol=2e-4, rtol=2e-4)
    _test_pre_post_phase(B=2, L=64,  n=4, C=128, dtype=torch.bfloat16, atol=1e-1, rtol=1e-2)
    _test_pre_post_phase(B=1, L=256, n=2, C=512, dtype=torch.bfloat16, atol=1e-1, rtol=1e-2)

    print("=" * 78)
    print("mhc_fused_no_hres.py: 性能 benchmark（ref vs fused）")
    print("=" * 78)
    for cfg in [
        (4, 512,  4, 256),
        (4, 1024, 4, 512),
        (2, 2048, 8, 512),
    ]:
        _bench(*cfg, dtype=torch.bfloat16)

    print("\n所有测试通过 ✓")