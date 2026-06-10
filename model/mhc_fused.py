"""mHC（Manifold-Constrained Hyper-Connections）—— 标准版，含 H^res 完整 fused 实现。

mhc_fused_no_hres.py 的超集，额外提供：
  - H^res 动态路径（Sinkhorn-Knopp doubly stochastic 投影）
  - POST phase 升级为 out = H^res·x + H^post·F_out（含 (n,n)×(n,C) matmul）

约束（精简 ablation 范围以最大化融合深度）：
  input_norm='global' / h_pre_activation='sigmoid' / disable_h_*=False
  n ∈ {1, 2, 4, 8, 16}；n·dim 是 2 的幂。

forward kernels：A=pre_phase（RMSNorm+phi+split+σ/raw） · B=SK · sublayer · C=post_phase
工程决策：大 grad_W 走 cuBLAS（避 N²·FD atomic）；小 grad（α/β）走 fp32 atomic_add；
grad_x 单次 store 写回；phi_weight 以 (n·C, 2n+n²) row-major 存储。
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

# env 一键禁用 fused 路径
_FUSED_MHC_ENABLED = os.environ.get("MINIMIND_FUSED_MHC", "1") != "0"
_FUSED_SINKHORN_ENABLED = os.environ.get("MINIMIND_FUSED_SINKHORN", "1") != "0"


def _is_pow2(n: int) -> bool:
    return n >= 1 and (n & (n - 1)) == 0


# ──────────────────────────────────────────────────────────────────────────────
# Sinkhorn-Knopp（H^res 专属）：(..., n, n) raw → Birkhoff polytope (doubly stochastic)
# ──────────────────────────────────────────────────────────────────────────────

def _sinkhorn_ref(M: torch.Tensor, iters: int, eps: float) -> torch.Tensor:
    """PyTorch ref（对齐官方）：softmax(-1) 起步 + 1 col + (iters-1) 次 row/col 交替；
    eps 用于 0/0 退化兜底，避免行/列 mass→0 时除法爆炸。
    """
    x = M.softmax(dim=-1) + eps
    x = x / (x.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        x = x / (x.sum(dim=-1, keepdim=True) + eps)
        x = x / (x.sum(dim=-2, keepdim=True) + eps)
    return x


if TRITON_AVAILABLE:

    @triton.autotune(configs=AUTOTUNE_CFGS_LIGHT, key=['N', 'ITERS'])
    @triton.jit
    def _sinkhorn_fwd_kernel(
        M_ptr,                  # (BL, N, N) contiguous
        Y_ptr,                  # (BL, N, N) contiguous
        eps,                    # float scalar
        N: tl.constexpr,        # 矩阵阶数（必须是 2 的幂）
        ITERS: tl.constexpr,    # 迭代次数（编译期常量，循环全展开）
    ):
        """1 program / 1 矩阵：N×N 全寄存器驻留；softmax 起步 + ITERS-1 次 row/col 交替全展开。"""
        pid = tl.program_id(0)

        rows = tl.arange(0, N)[:, None]                # (N, 1)
        cols = tl.arange(0, N)[None, :]                # (1, N)
        offs = pid * N * N + rows * N + cols           # (N, N)

        x = tl.load(M_ptr + offs).to(tl.float32)

        # 第 1 次 row normalize：softmax(dim=-1) + eps（含数值稳定的减 max）
        row_max = tl.max(x, axis=1)[:, None]
        x = tl.exp(x - row_max)
        x = x / tl.sum(x, axis=1)[:, None]
        x = x + eps

        # 第 1 次 col normalize
        x = x / (tl.sum(x, axis=0)[None, :] + eps)

        # 剩余 (ITERS-1) 次 row/col 交替（编译期全展开）
        for _ in tl.static_range(0, ITERS - 1):
            x = x / (tl.sum(x, axis=1)[:, None] + eps)
            x = x / (tl.sum(x, axis=0)[None, :] + eps)

        tl.store(Y_ptr + offs, x.to(Y_ptr.dtype.element_ty))


class _SinkhornFn(torch.autograd.Function):
    """Sinkhorn-Knopp：fused forward + recompute backward。

    forward：Triton fused kernel —— softmax 起步 + iters 次 row/col 交替合并为
    单 program，N×N 全程驻留寄存器，规避 PyTorch 路径下 iters 次 op launch +
    中间张量 HBM 往返。

    backward：**recompute 策略**——只保存输入 M（小张量），反向时调 _sinkhorn_ref
    重跑 forward 并通过 autograd 求导：
      1) 反向只需保存输入 M（O(N²·BL) 显存），无需保存 iters 次中间 trajectory
      2) ref 反向的 op 数和 forward 同量级，开销与未 fused 时持平
      3) SK 在 mHC 训练中通常不是瓶颈（每层 attn/mlp 内部 GEMM 占绝对大头）
    """

    @staticmethod
    def forward(ctx, M: torch.Tensor, iters: int, eps: float) -> torch.Tensor:
        orig_shape = M.shape
        N = M.shape[-1]
        assert M.shape[-2] == N, f"最后两维必须相等，得到 {tuple(M.shape)}"

        M_flat = M.contiguous().view(-1, N, N)
        BL = M_flat.shape[0]
        Y_flat = torch.empty_like(M_flat)

        _sinkhorn_fwd_kernel[(BL,)](
            M_flat, Y_flat, eps,
            N=N, ITERS=iters,
        )

        ctx.save_for_backward(M)
        ctx.iters = iters
        ctx.eps = eps
        return Y_flat.view(orig_shape)

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        (M,) = ctx.saved_tensors
        with torch.enable_grad(), torch.amp.autocast("cuda", enabled=False):
            M_ = M.detach().requires_grad_(True)
            y_ = _sinkhorn_ref(M_, ctx.iters, ctx.eps)
            (dx,) = torch.autograd.grad(y_, M_, dy)
        return dx, None, None


@torch.amp.autocast("cuda", enabled=False)  # 强制 fp32 路径，规避 bf16 exp 溢出
def sinkhorn_normalize(M: torch.Tensor, iters: int = 10, eps: float = 1e-6) -> torch.Tensor:
    """(..., n, n) raw → Birkhoff polytope。CUDA+Triton+n 为 2 的幂时走 fused，否则 ref。
    env: `MINIMIND_FUSED_SINKHORN=0` 可禁用 fused。"""
    N = M.shape[-1]
    use_fused = (
        _FUSED_SINKHORN_ENABLED
        and TRITON_AVAILABLE
        and M.is_cuda
        and M.shape[-2] == N
        and _is_pow2(N)
    )
    if use_fused:
        return _SinkhornFn.apply(M, iters, eps)
    return _sinkhorn_ref(M, iters, eps)


# ──────────────────────────────────────────────────────────────────────────────
# Triton kernels
# ──────────────────────────────────────────────────────────────────────────────

if TRITON_AVAILABLE:

    @triton.autotune(configs=AUTOTUNE_CFGS_HEAVY, key=['NC', 'C', 'N', 'N2', 'FD'])
    @triton.jit
    def _pre_phase_full_fwd_kernel(
        # ── 输入张量 ──
        X_ptr,              # (BL, NC)        x_flat
        Wphi_ptr,           # (NC, FD)        phi_weight，row-major (NC, FD)
        AlphaPre_ptr,       # (1,) fp32
        BiasPre_ptr,        # (N,) fp32
        AlphaPost_ptr,      # (1,) fp32
        BiasPost_ptr,       # (N,) fp32
        AlphaRes_ptr,       # (1,) fp32
        BiasRes_ptr,        # (N, N) fp32     β_res 是 (n,n) 矩阵（init=eye）
        # ── 输出 ──
        SublayerIn_ptr,     # (BL, C)
        HPre_ptr,           # (BL, N)
        HPost_ptr,          # (BL, N)
        HResRaw_ptr,        # (BL, N, N)      α_res·raw + β_res，未经 SK 投影
        # ── 反向需要保存 ──
        InvRms_ptr,         # (BL,) fp32
        MixesRaw_ptr,       # (BL, FD) fp32   完整 raw phi 输出
        # ── 标量 ──
        post_mult,
        pre_eps,
        rms_eps,
        # ── 编译期常量 ──
        NC: tl.constexpr,
        C: tl.constexpr,
        N: tl.constexpr,
        N2: tl.constexpr,           # N*N = res 段维度
        FD: tl.constexpr,           # 2N + N²
        BLOCK_NC: tl.constexpr,
        USE_BLOCK_PTR: tl.constexpr,    # H1.2: block_ptr / TMA path
    ):
        """1 program / 1 token: PRE phase = RMSNorm + phi GEMM + split (pre/post/res)
        + 仿射激活（pre/post sigmoid, res 等 SK） + sublayer_in reduce。"""
        pid = tl.program_id(0)

        offs_nc = tl.arange(0, BLOCK_NC)
        mask_nc = offs_nc < NC
        offs_n  = tl.arange(0, N)
        offs_c  = tl.arange(0, C)
        offs_fd = tl.arange(0, FD)
        offs_n2 = tl.arange(0, N2)              # (N²,) flat indices for res 段

        # ── Stage 1: load x + RMSNorm ──
        x = tl.load(X_ptr + pid * NC + offs_nc, mask=mask_nc, other=0.0).to(tl.float32)
        sum_sq = tl.sum(x * x)
        inv_rms = tl.rsqrt(sum_sq / NC + rms_eps)
        tl.store(InvRms_ptr + pid, inv_rms)
        x_norm = x * inv_rms                                                # (BLOCK_NC,) fp32

        # ── Stage 2: phi GEMM (W 全 SRAM；nC·FD bf16 ≈ 192KB 临界，L2 cache 必中) ──
        # H1.2: block_ptr 在 Hopper 上 Triton 自动 lower 到 cp.async.bulk.tensor (TMA);
        # FD = 2N+N²，W 最大（含 H^res 段），TMA 收益最显著。
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
        mixes_raw = tl.sum(x_norm[:, None] * w, axis=0)                     # (FD,) fp32
        tl.store(MixesRaw_ptr + pid * FD + offs_fd, mixes_raw)

        # ── Stage 3: split pre/post/res 并仿射激活 ──
        # 用 where-broadcast 从 mixes_raw 提取 pre/post 段（与 no_hres 一致）
        pre_seg = tl.sum(tl.where(offs_fd[None, :] == offs_n[:, None],
                                  mixes_raw[None, :], 0.0), axis=1)         # (N,)
        post_seg = tl.sum(tl.where(offs_fd[None, :] == (offs_n[:, None] + N),
                                   mixes_raw[None, :], 0.0), axis=1)        # (N,)
        # res 段：offs_n2 ∈ [0, N²)，对应 mixes_raw[2N + offs_n2]
        res_seg = tl.sum(tl.where(offs_fd[None, :] == (offs_n2[:, None] + 2 * N),
                                  mixes_raw[None, :], 0.0), axis=1)         # (N²,) flat

        alpha_pre  = tl.load(AlphaPre_ptr ).to(tl.float32)
        alpha_post = tl.load(AlphaPost_ptr).to(tl.float32)
        alpha_res  = tl.load(AlphaRes_ptr ).to(tl.float32)
        bias_pre   = tl.load(BiasPre_ptr  + offs_n ).to(tl.float32)         # (N,)
        bias_post  = tl.load(BiasPost_ptr + offs_n ).to(tl.float32)
        bias_res   = tl.load(BiasRes_ptr  + offs_n2).to(tl.float32)         # (N²,)

        h_pre_raw  = pre_seg  * alpha_pre  + bias_pre
        h_post_raw = post_seg * alpha_post + bias_post
        h_pre  = tl.sigmoid(h_pre_raw)  + pre_eps
        h_post = tl.sigmoid(h_post_raw) * post_mult

        # H_res_raw 是 (n,n) 矩阵的 flat 形式，等 SK 投影
        h_res_raw = res_seg * alpha_res + bias_res                          # (N²,) fp32

        tl.store(HPre_ptr   + pid * N  + offs_n,  h_pre.to(HPre_ptr.dtype.element_ty))
        tl.store(HPost_ptr  + pid * N  + offs_n,  h_post.to(HPost_ptr.dtype.element_ty))
        tl.store(HResRaw_ptr + pid * N2 + offs_n2, h_res_raw.to(HResRaw_ptr.dtype.element_ty))

        # ── Stage 4: sublayer_in = Σᵢ H_pre[i] · x[i, :] ──
        # P2: 直接 reshape Stage 1 已 load 的 x（BLOCK_NC == NC，launcher 已 assert），
        # 省第二次 HBM/L2 load
        x_2d = tl.reshape(x, (N, C))                                        # (N, C) view
        sublayer_in = tl.sum(h_pre[:, None] * x_2d, axis=0)                 # (C,)
        tl.store(SublayerIn_ptr + pid * C + offs_c,
                 sublayer_in.to(SublayerIn_ptr.dtype.element_ty))


    @triton.autotune(configs=AUTOTUNE_CFGS_HEAVY, key=['NC', 'C', 'N', 'N2', 'FD'])
    @triton.jit
    def _pre_phase_full_bwd_kernel(
        # ── forward 保存 ──
        X_ptr,              # (BL, NC)
        Wphi_ptr,           # (NC, FD)
        AlphaPre_ptr, BiasPre_ptr,
        AlphaPost_ptr, BiasPost_ptr,
        AlphaRes_ptr, BiasRes_ptr,
        InvRms_ptr,         # (BL,) fp32
        MixesRaw_ptr,       # (BL, FD) fp32
        # ── 上游 grad ──
        GradSublayerIn_ptr, # (BL, C)
        GradHPre_ptr,       # (BL, N)        通常 0（H_pre 不直接被下游用）
        GradHPost_ptr,      # (BL, N)        来自 post_phase backward
        GradHResRaw_ptr,    # (BL, N, N)     来自 SK backward（recompute）
        # ── 输出 grad ──
        GradX_ptr,          # (BL, NC)
        GradMixesRaw_ptr,   # (BL, FD)
        # ── per-token α/β grad（P1: 替代 atomic_add；外部 PyTorch reduce 完成最终汇总）──
        # 仅需 pre/post 的 grad_h_pre_raw / grad_h_post_raw（sigmoid 反向后），
        # res 段无激活，外部直接对 grad_h_res_raw_c 做 sum / dot 即可，省两次 store
        GradHPreRawPerTok_ptr,   # (BL, N) fp32
        GradHPostRawPerTok_ptr,  # (BL, N) fp32
        # ── 标量 ──
        post_mult,
        pre_eps,
        rms_eps,
        # ── 编译期常量 ──
        NC: tl.constexpr,
        C: tl.constexpr,
        N: tl.constexpr,
        N2: tl.constexpr,
        FD: tl.constexpr,
        BLOCK_NC: tl.constexpr,
        USE_BLOCK_PTR: tl.constexpr,    # H1.2: block_ptr / TMA path
    ):
        """1 program / 1 token: PRE phase 全反向（pre/post/res 三段并行 α·+β 反向 +
        sublayer_in reduce 反向 + phi mini-GEMM + RMSNorm 反向 + 合并 grad_x）。"""
        pid = tl.program_id(0)

        offs_nc = tl.arange(0, BLOCK_NC)
        mask_nc = offs_nc < NC
        offs_n  = tl.arange(0, N)
        offs_c  = tl.arange(0, C)
        offs_fd = tl.arange(0, FD)
        offs_n2 = tl.arange(0, N2)

        # ── 重新 load forward 中保存的张量 ──
        x         = tl.load(X_ptr + pid * NC + offs_nc, mask=mask_nc, other=0.0).to(tl.float32)
        inv_rms   = tl.load(InvRms_ptr + pid).to(tl.float32)
        mixes_raw = tl.load(MixesRaw_ptr + pid * FD + offs_fd).to(tl.float32)

        alpha_pre  = tl.load(AlphaPre_ptr ).to(tl.float32)
        alpha_post = tl.load(AlphaPost_ptr).to(tl.float32)
        alpha_res  = tl.load(AlphaRes_ptr ).to(tl.float32)
        bias_pre   = tl.load(BiasPre_ptr  + offs_n ).to(tl.float32)
        bias_post  = tl.load(BiasPost_ptr + offs_n ).to(tl.float32)
        # bias_res 不需要在 bwd 加载（α·raw+β 反向不需要 β 本身）

        # ── 重算 forward 中间量 ──
        x_norm = x * inv_rms

        pre_seg = tl.sum(tl.where(offs_fd[None, :] == offs_n[:, None],
                                  mixes_raw[None, :], 0.0), axis=1)         # (N,)
        post_seg = tl.sum(tl.where(offs_fd[None, :] == (offs_n[:, None] + N),
                                   mixes_raw[None, :], 0.0), axis=1)
        res_seg = tl.sum(tl.where(offs_fd[None, :] == (offs_n2[:, None] + 2 * N),
                                  mixes_raw[None, :], 0.0), axis=1)         # (N²,)

        h_pre_raw  = pre_seg  * alpha_pre  + bias_pre
        h_post_raw = post_seg * alpha_post + bias_post
        sigmoid_pre  = tl.sigmoid(h_pre_raw)
        sigmoid_post = tl.sigmoid(h_post_raw)
        h_pre = sigmoid_pre + pre_eps

        # ── 上游 grad ──
        grad_sub_in   = tl.load(GradSublayerIn_ptr + pid * C  + offs_c).to(tl.float32)
        grad_h_pre_up = tl.load(GradHPre_ptr   + pid * N  + offs_n ).to(tl.float32)
        grad_h_post   = tl.load(GradHPost_ptr  + pid * N  + offs_n ).to(tl.float32)
        grad_h_res_raw= tl.load(GradHResRaw_ptr+ pid * N2 + offs_n2).to(tl.float32)  # (N²,)

        # ── 1) sublayer_in reduce 反向 ──
        # P2: 直接 reshape 已 load 的 x，省第二次 HBM/L2 load
        x_2d = tl.reshape(x, (N, C))                                        # (N, C) view
        grad_h_pre_from_reduce = tl.sum(grad_sub_in[None, :] * x_2d, axis=1)# (N,)
        grad_h_pre = grad_h_pre_up + grad_h_pre_from_reduce

        # ── 2) Sigmoid 反向 ──
        grad_h_pre_raw  = grad_h_pre  * sigmoid_pre  * (1.0 - sigmoid_pre)
        grad_h_post_raw = grad_h_post * sigmoid_post * (1.0 - sigmoid_post) * post_mult

        # ── 3) α·raw + β 反向：grad_pre/post/res_seg 仍需 in-kernel（供下游 phi 组装），
        #     但 α/β 累加全部移到 PyTorch 端：
        #       pre/post 段：per-token store grad_h_pre_raw / grad_h_post_raw
        #       res 段：grad_h_res_raw 就是上游 grad_h_res_raw_c，外部直接 sum / dot
        grad_pre_seg  = grad_h_pre_raw  * alpha_pre                         # (N,)
        grad_post_seg = grad_h_post_raw * alpha_post                        # (N,)
        grad_res_seg  = grad_h_res_raw  * alpha_res                         # (N²,) flat

        tl.store(GradHPreRawPerTok_ptr  + pid * N + offs_n, grad_h_pre_raw )
        tl.store(GradHPostRawPerTok_ptr + pid * N + offs_n, grad_h_post_raw)

        # ── 4) 组装 grad_mixes_raw (FD,) = concat(grad_pre_seg, grad_post_seg, grad_res_seg) ──
        grad_mixes_raw = (
            tl.sum(tl.where(offs_n[:, None]        == offs_fd[None, :],
                            grad_pre_seg[:, None],  0.0), axis=0)
            + tl.sum(tl.where((offs_n[:, None] + N) == offs_fd[None, :],
                              grad_post_seg[:, None], 0.0), axis=0)
            + tl.sum(tl.where((offs_n2[:, None] + 2 * N) == offs_fd[None, :],
                              grad_res_seg[:, None],  0.0), axis=0)
        )                                                                   # (FD,)
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

        # ── 6) RMSNorm 反向 ──
        mean_y_gy = tl.sum(x_norm * grad_x_norm) / NC
        grad_x_b = inv_rms * (grad_x_norm - x_norm * mean_y_gy)             # (BLOCK_NC,)

        # ── 7) 合并 grad_x_b 与 grad_x_a (sublayer_in 反向) 单次 store ──
        grad_x_a_2d = h_pre[:, None] * grad_sub_in[None, :]                 # (N, C)
        grad_x_b_2d = tl.reshape(grad_x_b, (N, C))                          # (N, C) view
        grad_x_total = grad_x_b_2d + grad_x_a_2d
        offs_2d = offs_n[:, None] * C + offs_c[None, :]                     # 仅 store 用
        tl.store(GradX_ptr + pid * NC + offs_2d,
                 grad_x_total.to(GradX_ptr.dtype.element_ty))


    @triton.autotune(configs=AUTOTUNE_CFGS_LIGHT, key=['N', 'C'])
    @triton.jit
    def _post_phase_full_fwd_kernel(
        X_ptr,              # (BL, N, C)  flat 残差流（视为 (BL, N*C)）
        HRes_ptr,           # (BL, N, N)  SK 后的 doubly stochastic 矩阵
        HPost_ptr,          # (BL, N)
        FOut_ptr,           # (BL, C)
        Out_ptr,            # (BL, N, C)
        N: tl.constexpr,
        C: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        """1 program / 1 token: out[i,c] = Σⱼ H_res[i,j]·x[j,c] + H_post[i]·F_out[c]。"""
        pid = tl.program_id(0)

        offs_n = tl.arange(0, N)
        offs_n2 = tl.arange(0, N)               # 列索引 j（与 offs_n 同范围）
        offs_c = tl.arange(0, BLOCK_C)
        mask_c = offs_c < C

        # load H_res (N, N) 进 SRAM
        h_res_idx = offs_n[:, None] * N + offs_n2[None, :]                  # (N, N)
        h_res = tl.load(HRes_ptr + pid * N * N + h_res_idx).to(tl.float32)  # (N, N)
        # load H_post (N,) 与 F_out (C,)
        h_post = tl.load(HPost_ptr + pid * N + offs_n).to(tl.float32)
        f_out  = tl.load(FOut_ptr  + pid * C + offs_c, mask=mask_c, other=0.0).to(tl.float32)
        # load x (N, C)
        offs_2d = offs_n2[:, None] * C + offs_c[None, :]                    # j 在 row, c 在 col
        mask_2d = mask_c[None, :]
        x = tl.load(X_ptr + pid * N * C + offs_2d, mask=mask_2d, other=0.0).to(tl.float32)
        # x shape: (N=j, C=c)

        # mixed = H_res @ x: 用 broadcast-reduce 而非 tl.dot（n ∈ {1,2,4,8} 不满足
        # tl.dot 最小 16 形状约束；小 n 下 broadcast 显著更优）
        mixed = tl.sum(h_res[:, :, None] * x[None, :, :], axis=1)           # (N, C) fp32

        # write_back[i, c] = H_post[i] · F_out[c]
        write_back = h_post[:, None] * f_out[None, :]                       # (N, C)
        out = mixed + write_back

        # store (i, c)
        out_idx = offs_n[:, None] * C + offs_c[None, :]                     # (N, C)
        tl.store(Out_ptr + pid * N * C + out_idx,
                 out.to(Out_ptr.dtype.element_ty), mask=mask_2d)


    @triton.autotune(configs=AUTOTUNE_CFGS_LIGHT, key=['N', 'C'])
    @triton.jit
    def _post_phase_full_bwd_kernel(
        # forward 保存
        X_ptr,              # (BL, N, C)
        HRes_ptr,           # (BL, N, N)
        HPost_ptr,          # (BL, N)
        FOut_ptr,           # (BL, C)
        # 上游 grad
        GradOut_ptr,        # (BL, N, C)
        # 输出 grad
        GradX_ptr,          # (BL, N, C)
        GradHRes_ptr,       # (BL, N, N)
        GradHPost_ptr,      # (BL, N)
        GradFOut_ptr,       # (BL, C)
        N: tl.constexpr,
        C: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        """1 program / 1 token: POST phase 全反向（H_res.T @ g_out → grad_x;
        g_out @ x.T → grad_H_res; reduce on c/i → grad_H_post / grad_F_out）。"""
        pid = tl.program_id(0)

        offs_n  = tl.arange(0, N)
        offs_n2 = tl.arange(0, N)
        offs_c  = tl.arange(0, BLOCK_C)
        mask_c  = offs_c < C
        mask_2d = mask_c[None, :]

        # load grad_out (N=i, C=c)
        out_idx = offs_n[:, None] * C + offs_c[None, :]
        g_out = tl.load(GradOut_ptr + pid * N * C + out_idx,
                        mask=mask_2d, other=0.0).to(tl.float32)             # (N, C)

        # load H_res (N=i, N=j), H_post (N,), F_out (C,)
        h_res_idx = offs_n[:, None] * N + offs_n2[None, :]
        h_res = tl.load(HRes_ptr + pid * N * N + h_res_idx).to(tl.float32)  # (N, N)
        h_post = tl.load(HPost_ptr + pid * N + offs_n).to(tl.float32)
        f_out  = tl.load(FOut_ptr  + pid * C + offs_c, mask=mask_c, other=0.0).to(tl.float32)

        # load x (N=j, C=c)
        x_idx = offs_n2[:, None] * C + offs_c[None, :]
        x = tl.load(X_ptr + pid * N * C + x_idx,
                    mask=mask_2d, other=0.0).to(tl.float32)                 # (N=j, C)

        # grad_x = H_res.T @ g_out  (broadcast-reduce over i)
        grad_x = tl.sum(h_res[:, :, None] * g_out[:, None, :], axis=0)      # (N=j, C)
        # grad_H_res = g_out @ x.T  (broadcast-reduce over c)
        grad_h_res = tl.sum(g_out[:, None, :] * x[None, :, :], axis=2)      # (N=i, N=j)

        # ── grad_H_post[i] = Σ_c g_out[i, c] · F_out[c] ──
        grad_h_post = tl.sum(g_out * f_out[None, :], axis=1)                # (N,)

        # ── grad_F_out[c] = Σᵢ g_out[i, c] · H_post[i] ──
        grad_f_out = tl.sum(g_out * h_post[:, None], axis=0)                # (C,)

        # store
        tl.store(GradX_ptr + pid * N * C + x_idx,
                 grad_x.to(GradX_ptr.dtype.element_ty), mask=mask_2d)
        tl.store(GradHRes_ptr + pid * N * N + h_res_idx,
                 grad_h_res.to(GradHRes_ptr.dtype.element_ty))
        tl.store(GradHPost_ptr + pid * N + offs_n,
                 grad_h_post.to(GradHPost_ptr.dtype.element_ty))
        tl.store(GradFOut_ptr + pid * C + offs_c,
                 grad_f_out.to(GradFOut_ptr.dtype.element_ty), mask=mask_c)


# ──────────────────────────────────────────────────────────────────────────────
# PyTorch reference 实现（fallback / 正确性比对）
# ──────────────────────────────────────────────────────────────────────────────

def _pre_phase_full_ref(x_flat, w_phi, alpha_pre, bias_pre, alpha_post, bias_post,
                        alpha_res, bias_res, post_mult, pre_eps, rms_eps, N, C):
    """PyTorch ref：PRE phase（含 res 段；返回 h_res_raw，尚未做 SK 投影）。"""
    B, L, NC = x_flat.shape
    assert NC == N * C
    x_f = x_flat.float()
    inv_rms = x_f.pow(2).mean(-1, keepdim=True).add(rms_eps).rsqrt()
    x_norm = (x_f * inv_rms).to(x_flat.dtype)
    mixes_raw = x_norm @ w_phi                                              # (B,L, 2N+N²)
    FD = mixes_raw.shape[-1]
    assert FD == 2 * N + N * N

    pre_seg  = mixes_raw[..., :N]
    post_seg = mixes_raw[..., N:2*N]
    res_seg  = mixes_raw[..., 2*N:2*N+N*N].view(B, L, N, N)

    h_pre_raw  = pre_seg  * alpha_pre  + bias_pre
    h_post_raw = post_seg * alpha_post + bias_post
    h_pre  = torch.sigmoid(h_pre_raw)  + pre_eps
    h_post = torch.sigmoid(h_post_raw) * post_mult

    h_res_raw = res_seg * alpha_res + bias_res                              # (B,L,N,N)

    x = x_flat.view(B, L, N, C)
    sublayer_in = (h_pre.unsqueeze(-1) * x).sum(dim=-2)                     # (B,L,C)

    return sublayer_in, h_pre, h_post, h_res_raw, inv_rms.squeeze(-1).float(), mixes_raw.float()


def _post_phase_full_ref(x_flat, h_res, h_post, f_out, N, C):
    """完整 POST phase: out = H_res · x + H_post · F_out。"""
    B, L, NC = x_flat.shape
    x = x_flat.view(B, L, N, C)
    mixed = torch.einsum('blij,bljc->blic', h_res, x)                       # (B,L,N,C)
    write_back = h_post.unsqueeze(-1) * f_out.unsqueeze(-2)                 # (B,L,N,C)
    return (mixed + write_back).reshape(B, L, NC)


class _PrePhaseFullFn(torch.autograd.Function):
    """mHC PRE phase fused fwd+bwd → (sublayer_in, h_pre, h_post, h_res_raw)。"""

    @staticmethod
    def forward(ctx, x_flat, w_phi, alpha_pre, bias_pre, alpha_post, bias_post,
                alpha_res, bias_res, post_mult, pre_eps, rms_eps, n, c):
        # w_phi: (n·c, 2n+n²) row-major; bias_res: (n,n) init=eye
        assert x_flat.is_cuda and TRITON_AVAILABLE
        B, L, NC = x_flat.shape
        BL = B * L
        N2 = n * n
        FD = 2 * n + N2
        assert w_phi.shape == (NC, FD)

        x_flat_c = x_flat.contiguous().view(BL, NC)
        w_phi_c  = w_phi.contiguous()

        # 输出
        sublayer_in = torch.empty(BL, c,     dtype=x_flat.dtype, device=x_flat.device)
        h_pre       = torch.empty(BL, n,     dtype=x_flat.dtype, device=x_flat.device)
        h_post      = torch.empty(BL, n,     dtype=x_flat.dtype, device=x_flat.device)
        h_res_raw   = torch.empty(BL, n, n,  dtype=x_flat.dtype, device=x_flat.device)
        # 反向需要保存
        inv_rms     = torch.empty(BL,        dtype=torch.float32, device=x_flat.device)
        mixes_raw   = torch.empty(BL, FD,    dtype=torch.float32, device=x_flat.device)

        BLOCK_NC = triton.next_power_of_2(NC)
        assert BLOCK_NC == NC, (
            f"n·dim={NC} 必须是 2 的幂（当前 next_pow2={BLOCK_NC}）；"
            f"实际项目里 n ∈ {{1,2,4,8,16}}、dim ∈ {{256,512,1024,2048}} 都满足"
        )
        # device-aware SRAM 上限（H800 228KB / A100 192KB / consumer ~100KB）
        sram_bytes = NC * 4 + NC * FD * 4
        sram_limit = max_sram_bytes(x_flat.device)
        assert sram_bytes <= sram_limit, (
            f"NC={NC} FD={FD} 需要 {sram_bytes/1024:.0f}KB SRAM；"
            f"超出当前 GPU 上限 {sram_limit/1024:.0f}KB"
        )

        alpha_pre_f  = alpha_pre.float().reshape(1)
        alpha_post_f = alpha_post.float().reshape(1)
        alpha_res_f  = alpha_res.float().reshape(1)
        bias_res_flat = bias_res.float().reshape(N2)                        # (n,n) → (n²,)

        _pre_phase_full_fwd_kernel[(BL,)](
            x_flat_c, w_phi_c,
            alpha_pre_f,  bias_pre.float(),
            alpha_post_f, bias_post.float(),
            alpha_res_f,  bias_res_flat,
            sublayer_in, h_pre, h_post, h_res_raw,
            inv_rms, mixes_raw,
            float(post_mult), float(pre_eps), float(rms_eps),
            NC=NC, C=c, N=n, N2=N2, FD=FD, BLOCK_NC=BLOCK_NC,
            USE_BLOCK_PTR=_USE_BLOCK_PTR_W,
        )

        ctx.save_for_backward(x_flat_c, w_phi_c,
                              alpha_pre_f, bias_pre,
                              alpha_post_f, bias_post,
                              alpha_res_f, bias_res,
                              inv_rms, mixes_raw)
        ctx.post_mult = float(post_mult)
        ctx.pre_eps   = float(pre_eps)
        ctx.rms_eps   = float(rms_eps)
        ctx.n = n; ctx.c = c
        ctx.shape = (B, L, NC)

        return (sublayer_in.view(B, L, c),
                h_pre.view(B, L, n),
                h_post.view(B, L, n),
                h_res_raw.view(B, L, n, n))

    @staticmethod
    def backward(ctx, grad_sublayer_in, grad_h_pre, grad_h_post, grad_h_res_raw):
        (x_flat_c, w_phi_c,
         alpha_pre_f, bias_pre,
         alpha_post_f, bias_post,
         alpha_res_f, bias_res,
         inv_rms, mixes_raw) = ctx.saved_tensors

        B, L, NC = ctx.shape
        BL = B * L
        n  = ctx.n
        c  = ctx.c
        N2 = n * n
        FD = 2 * n + N2

        # 上游 grad None handling
        def _zeros_like(shape, dtype=x_flat_c.dtype):
            return torch.zeros(*shape, dtype=dtype, device=x_flat_c.device)
        grad_sub_in_c   = (grad_sublayer_in.contiguous().view(BL, c)    if grad_sublayer_in is not None
                           else _zeros_like((BL, c)))
        grad_h_pre_c    = (grad_h_pre.contiguous().view(BL, n)          if grad_h_pre is not None
                           else _zeros_like((BL, n)))
        grad_h_post_c   = (grad_h_post.contiguous().view(BL, n)         if grad_h_post is not None
                           else _zeros_like((BL, n)))
        grad_h_res_raw_c= (grad_h_res_raw.contiguous().view(BL, N2)     if grad_h_res_raw is not None
                           else _zeros_like((BL, N2)))

        # 输出 grad
        grad_x         = torch.empty(BL, NC, dtype=x_flat_c.dtype, device=x_flat_c.device)
        grad_mixes_raw = torch.empty(BL, FD, dtype=torch.float32, device=x_flat_c.device)
        # per-token α/β grad（P1: 替代 atomic_add；只为 pre/post 段开辟，res 段直接复用上游 grad）
        grad_h_pre_raw_per_tok  = torch.empty(BL, n, dtype=torch.float32, device=x_flat_c.device)
        grad_h_post_raw_per_tok = torch.empty(BL, n, dtype=torch.float32, device=x_flat_c.device)

        BLOCK_NC = NC                                                       # 已 assert pow-of-2
        # bias_res 在 bwd kernel 只是占位（kernel 内部不再使用 bias_res；保留接口对称）
        bias_res_flat_dummy = bias_res.float().reshape(N2)

        _pre_phase_full_bwd_kernel[(BL,)](
            x_flat_c, w_phi_c,
            alpha_pre_f,  bias_pre.float(),
            alpha_post_f, bias_post.float(),
            alpha_res_f,  bias_res_flat_dummy,
            inv_rms, mixes_raw,
            grad_sub_in_c, grad_h_pre_c, grad_h_post_c, grad_h_res_raw_c,
            grad_x, grad_mixes_raw,
            grad_h_pre_raw_per_tok, grad_h_post_raw_per_tok,
            ctx.post_mult, ctx.pre_eps, ctx.rms_eps,
            NC=NC, C=c, N=n, N2=N2, FD=FD, BLOCK_NC=BLOCK_NC,
            USE_BLOCK_PTR=_USE_BLOCK_PTR_W,
        )

        # grad_W = x_norm.T @ grad_mixes_raw （单次 cuBLAS GEMM）
        x_norm = x_flat_c.float() * inv_rms.unsqueeze(-1)                   # (BL, NC) fp32
        grad_w_phi = (x_norm.T @ grad_mixes_raw).to(w_phi_c.dtype)          # (NC, FD)

        # α/β grad（单次 PyTorch reduce 替代 BL 个 atomic_add；零原子竞争）
        # pre/post 段：per-token grad_h_pre/post_raw 已 store；α grad 需乘 pre/post_seg
        pre_seg_per_tok  = mixes_raw[:, :n]                                  # (BL, n)
        post_seg_per_tok = mixes_raw[:, n:2*n]                               # (BL, n)
        grad_alpha_pre  = (grad_h_pre_raw_per_tok  * pre_seg_per_tok ).sum()
        grad_alpha_post = (grad_h_post_raw_per_tok * post_seg_per_tok).sum()
        grad_bias_pre_t  = grad_h_pre_raw_per_tok.sum(dim=0)                 # (n,)
        grad_bias_post_t = grad_h_post_raw_per_tok.sum(dim=0)
        # res 段无激活：grad_h_res_raw 直接是上游 grad_h_res_raw_c，省 per-token store
        res_seg_per_tok = mixes_raw[:, 2*n:2*n + N2]                         # (BL, N²) fp32
        grad_h_res_raw_f = grad_h_res_raw_c.float()                          # (BL, N²)
        grad_alpha_res = (grad_h_res_raw_f * res_seg_per_tok).sum()
        grad_bias_res  = grad_h_res_raw_f.sum(dim=0).view(n, n).to(bias_res.dtype)

        return (grad_x.view(B, L, NC), grad_w_phi,
                grad_alpha_pre.to(alpha_pre_f.dtype),   grad_bias_pre_t.to(bias_pre.dtype),
                grad_alpha_post.to(alpha_post_f.dtype), grad_bias_post_t.to(bias_post.dtype),
                grad_alpha_res.to(alpha_res_f.dtype),   grad_bias_res,
                None, None, None, None, None)


class _PostPhaseFullFn(torch.autograd.Function):
    """mHC POST phase fused fwd+bwd: out = H_res · x + H_post · F_out。"""

    @staticmethod
    def forward(ctx, x_flat, h_res, h_post, f_out, n, c):
        assert x_flat.is_cuda and TRITON_AVAILABLE
        B, L, NC = x_flat.shape
        BL = B * L

        x_c     = x_flat.contiguous().view(BL, n * c)
        h_res_c = h_res.contiguous().view(BL, n, n)
        h_c     = h_post.contiguous().view(BL, n)
        f_c     = f_out.contiguous().view(BL, c)
        out     = torch.empty_like(x_c)

        BLOCK_C = triton.next_power_of_2(c)
        _post_phase_full_fwd_kernel[(BL,)](
            x_c, h_res_c, h_c, f_c, out,
            N=n, C=c, BLOCK_C=BLOCK_C,
        )

        ctx.save_for_backward(x_c, h_res_c, h_c, f_c)
        ctx.n = n; ctx.c = c
        ctx.shape = (B, L, NC)
        return out.view(B, L, NC)

    @staticmethod
    def backward(ctx, grad_out):
        x_c, h_res_c, h_c, f_c = ctx.saved_tensors
        B, L, NC = ctx.shape
        BL = B * L
        n  = ctx.n
        c  = ctx.c

        grad_out_c   = grad_out.contiguous().view(BL, n * c)
        grad_x       = torch.empty_like(x_c)
        grad_h_res   = torch.empty_like(h_res_c)
        grad_h_post  = torch.empty_like(h_c)
        grad_f_out   = torch.empty_like(f_c)

        BLOCK_C = triton.next_power_of_2(c)
        _post_phase_full_bwd_kernel[(BL,)](
            x_c, h_res_c, h_c, f_c,
            grad_out_c,
            grad_x, grad_h_res, grad_h_post, grad_f_out,
            N=n, C=c, BLOCK_C=BLOCK_C,
        )

        return (grad_x.view(B, L, NC),
                grad_h_res.view(B, L, n, n),
                grad_h_post.view(B, L, n),
                grad_f_out.view(B, L, c),
                None, None)


class MHCConnection_Fused(nn.Module):
    """完整 mHC 单 sublayer 包装：H^pre / H^post / H^res 三段全 dynamic + Triton fused。
    固化配置：input_norm='global' / h_pre_activation='sigmoid' / disable_h_*=False。
    """

    def __init__(self, n: int, dim: int, alpha_init: float = 0.01,
                 sinkhorn_iters: int = 10, post_mult_value: float = 2.0,
                 eps: float = 1e-6, pre_eps: float = 1e-6, sinkhorn_eps: float = 1e-6):
        super().__init__()
        assert _is_pow2(n) and n <= 16, f"n 必须 ∈ {{1,2,4,8,16}}，得到 {n}"
        assert post_mult_value > 1.0
        self.n = n
        self.dim = dim
        self.nC = n * dim
        self.N2 = n * n
        self.fused_dim = 2 * n + self.N2                                    # pre + post + res
        self.sinkhorn_iters = sinkhorn_iters
        self.post_mult_value = post_mult_value
        self.pre_eps = pre_eps
        self.sinkhorn_eps = sinkhorn_eps
        self.eps = eps

        # phi: row-major (nC, FD) 直接存储（保留 autograd 梯度连接）
        self.phi_weight = nn.Parameter(torch.empty(self.nC, self.fused_dim))
        nn.init.kaiming_uniform_(self.phi_weight, a=math.sqrt(5))

        # alpha / bias 分段独立
        self.alpha_pre  = nn.Parameter(torch.tensor(alpha_init))
        self.bias_pre   = nn.Parameter(torch.zeros(n))
        self.alpha_post = nn.Parameter(torch.tensor(alpha_init))
        self.bias_post  = nn.Parameter(torch.zeros(n))
        self.alpha_res  = nn.Parameter(torch.tensor(alpha_init))
        # bias_res init = eye(n)：初始 H_res 落在 SK(exp(I)) 对角占优区，规避 SK 在
        # uniform 处雅可比梯度压制。
        self.bias_res   = nn.Parameter(torch.eye(n))

    def extra_repr(self) -> str:
        return (f'n={self.n}, dim={self.dim}, fused_dim={self.fused_dim}, '
                f'sinkhorn_iters={self.sinkhorn_iters}, post_mult_value={self.post_mult_value}, '
                f'input_norm=\'global\'(fused), h_pre_activation=\'sigmoid\'(fused), '
                f'disable_h_*=False(file-level)')

    def forward(self, x_flat: torch.Tensor, sublayer_fn) -> torch.Tensor:
        """单 sublayer mHC 残差更新：PRE(Triton) → SK → sublayer F → POST(Triton)。"""
        if _FUSED_MHC_ENABLED and TRITON_AVAILABLE and x_flat.is_cuda:
            sublayer_in, _h_pre, h_post, h_res_raw = _PrePhaseFullFn.apply(
                x_flat, self.phi_weight,
                self.alpha_pre, self.bias_pre,
                self.alpha_post, self.bias_post,
                self.alpha_res, self.bias_res,
                self.post_mult_value, self.pre_eps, self.eps,
                self.n, self.dim,
            )
            h_res = sinkhorn_normalize(
                h_res_raw, iters=self.sinkhorn_iters, eps=self.sinkhorn_eps,
            ).to(x_flat.dtype)                                              # (B, L, n, n)
            f_out = sublayer_fn(sublayer_in)
            return _PostPhaseFullFn.apply(x_flat, h_res, h_post, f_out, self.n, self.dim)

        return self._forward_ref(x_flat, sublayer_fn)

    def _forward_ref(self, x_flat: torch.Tensor, sublayer_fn) -> torch.Tensor:
        """PyTorch ref：与 fused 路径数学等价（fallback / 比对）。"""
        B, L, _ = x_flat.shape
        n, C = self.n, self.dim
        x = x_flat.view(B, L, n, C)

        # RMSNorm + phi
        x_f = x_flat.float()
        inv_rms = x_f.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        x_norm = (x_f * inv_rms).to(x_flat.dtype)
        mixes_raw = x_norm @ self.phi_weight                                # (B, L, 2n+n²)

        pre_seg  = mixes_raw[..., :n]
        post_seg = mixes_raw[..., n:2*n]
        res_seg  = mixes_raw[..., 2*n:2*n+n*n].view(B, L, n, n)

        h_pre  = torch.sigmoid(pre_seg  * self.alpha_pre  + self.bias_pre)  + self.pre_eps
        h_post = torch.sigmoid(post_seg * self.alpha_post + self.bias_post) * self.post_mult_value
        h_res_raw = res_seg * self.alpha_res + self.bias_res
        h_res = sinkhorn_normalize(
            h_res_raw.float(), iters=self.sinkhorn_iters, eps=self.sinkhorn_eps,
        ).to(x_flat.dtype)

        sublayer_in = (h_pre.unsqueeze(-1) * x).sum(dim=-2)
        f_out = sublayer_fn(sublayer_in)
        mixed = torch.einsum('blij,bljc->blic', h_res, x)
        write_back = h_post.unsqueeze(-1) * f_out.unsqueeze(-2)
        out = mixed + write_back
        return out.reshape(B, L, self.nC)


# ──────────────────────────────────────────────────────────────────────────────
# 正确性测试 + benchmark（python -m model.mhc_fused）
# ──────────────────────────────────────────────────────────────────────────────

def _check(name, ref, got, atol, rtol):
    diff = (ref.float() - got.float()).abs().max().item()
    rel = diff / max(ref.float().abs().max().item(), 1e-8)
    flag = "✓" if (diff < atol or rel < rtol) else "✗"
    print(f"  {flag} {name:>22s}: max|diff|={diff:.3e}  rel={rel:.3e}")
    assert diff < atol or rel < rtol, f"{name} 不对齐: abs={diff}, rel={rel}"


def _test_pre_phase_full(B, L, n, C, dtype, atol, rtol):
    torch.manual_seed(0)
    device = 'cuda'
    NC = n * C
    N2 = n * n
    FD = 2 * n + N2

    x = torch.randn(B, L, NC, device=device, dtype=dtype, requires_grad=True)

    phi_w_t = torch.empty(FD, NC, device=device, dtype=dtype)
    nn.init.kaiming_uniform_(phi_w_t, a=math.sqrt(5))
    w_phi = phi_w_t.t().contiguous().detach().requires_grad_(True)

    alpha_pre  = torch.tensor(0.01, device=device, requires_grad=True)
    bias_pre   = torch.zeros(n, device=device, requires_grad=True)
    alpha_post = torch.tensor(0.01, device=device, requires_grad=True)
    bias_post  = torch.zeros(n, device=device, requires_grad=True)
    alpha_res  = torch.tensor(0.01, device=device, requires_grad=True)
    bias_res   = torch.eye(n, device=device, requires_grad=True)

    post_mult, pre_eps, rms_eps = 2.0, 1e-6, 1e-6

    # ref
    sub_ref, hp_ref, ho_ref, hr_ref, _, _ = _pre_phase_full_ref(
        x, w_phi, alpha_pre, bias_pre, alpha_post, bias_post,
        alpha_res, bias_res, post_mult, pre_eps, rms_eps, n, C,
    )

    # fused
    x2 = x.detach().clone().requires_grad_(True)
    w2 = w_phi.detach().clone().requires_grad_(True)
    ap = alpha_pre.detach().clone().requires_grad_(True)
    bp = bias_pre.detach().clone().requires_grad_(True)
    ao = alpha_post.detach().clone().requires_grad_(True)
    bo = bias_post.detach().clone().requires_grad_(True)
    ar = alpha_res.detach().clone().requires_grad_(True)
    br = bias_res.detach().clone().requires_grad_(True)
    sub_f, hp_f, ho_f, hr_f = _PrePhaseFullFn.apply(
        x2, w2, ap, bp, ao, bo, ar, br,
        post_mult, pre_eps, rms_eps, n, C,
    )

    print(f"[PRE-full B={B} L={L} n={n} C={C} dtype={str(dtype).replace('torch.','')}]")
    _check("sublayer_in", sub_ref, sub_f, atol, rtol)
    _check("H_pre",       hp_ref,  hp_f,  atol, rtol)
    _check("H_post",      ho_ref,  ho_f,  atol, rtol)
    _check("H_res_raw",   hr_ref,  hr_f,  atol, rtol)

    # ── 反向：多个上游 grad 同时反传 ──
    g_sub = torch.randn_like(sub_ref)
    g_ho  = torch.randn_like(ho_ref)
    g_hr  = torch.randn_like(hr_ref)
    loss_ref = (sub_ref * g_sub).sum() + (ho_ref * g_ho).sum() + (hr_ref * g_hr).sum()
    loss_f   = (sub_f   * g_sub).sum() + (ho_f   * g_ho).sum() + (hr_f   * g_hr).sum()
    loss_ref.backward()
    loss_f.backward()

    _check("grad_x",          x.grad,         x2.grad, atol, rtol)
    _check("grad_W_phi",      w_phi.grad,     w2.grad, atol*5, rtol*5)
    _check("grad_alpha_pre",  alpha_pre.grad, ap.grad, atol, rtol)
    _check("grad_bias_pre",   bias_pre.grad,  bp.grad, atol, rtol)
    _check("grad_alpha_post", alpha_post.grad,ao.grad, atol, rtol)
    _check("grad_bias_post",  bias_post.grad, bo.grad, atol, rtol)
    _check("grad_alpha_res",  alpha_res.grad, ar.grad, atol, rtol)
    _check("grad_bias_res",   bias_res.grad,  br.grad, atol, rtol)
    print()


def _test_post_phase_full(B, L, n, C, dtype, atol, rtol):
    torch.manual_seed(0)
    device = 'cuda'
    NC = n * C

    x = torch.randn(B, L, NC, device=device, dtype=dtype, requires_grad=True)
    # H_res 是 doubly stochastic：用 SK 投影随机矩阵
    h_res_raw = torch.randn(B, L, n, n, device=device, dtype=torch.float32)
    h_res = sinkhorn_normalize(h_res_raw, iters=10, eps=1e-6).to(dtype).detach().requires_grad_(True)
    h_post = (torch.rand(B, L, n, device=device, dtype=dtype) * 2.0).requires_grad_(True)
    f_out = torch.randn(B, L, C, device=device, dtype=dtype, requires_grad=True)

    # ref
    out_ref = _post_phase_full_ref(x, h_res, h_post, f_out, n, C)

    # fused
    x2 = x.detach().clone().requires_grad_(True)
    hr2 = h_res.detach().clone().requires_grad_(True)
    hp2 = h_post.detach().clone().requires_grad_(True)
    fo2 = f_out.detach().clone().requires_grad_(True)
    out_f = _PostPhaseFullFn.apply(x2, hr2, hp2, fo2, n, C)

    print(f"[POST-full B={B} L={L} n={n} C={C} dtype={str(dtype).replace('torch.','')}]")
    _check("out", out_ref, out_f, atol, rtol)

    g_out = torch.randn_like(out_ref)
    out_ref.backward(g_out)
    out_f.backward(g_out)
    _check("grad_x",      x.grad,      x2.grad, atol, rtol)
    _check("grad_h_res",  h_res.grad,  hr2.grad, atol*5, rtol*5)            # matmul 反向稍宽容
    _check("grad_h_post", h_post.grad, hp2.grad, atol, rtol)
    _check("grad_f_out",  f_out.grad,  fo2.grad, atol, rtol)
    print()


def _test_end2end(B, L, n, C, dtype, atol, rtol):
    """端到端测试：完整 MHCConnection_Fused vs PyTorch ref 路径。"""
    torch.manual_seed(0)
    device = 'cuda'
    NC = n * C

    mhc = MHCConnection_Fused(n, C).to(device=device, dtype=dtype)
    sublayer_fn = nn.Sequential(
        nn.Linear(C, 4 * C, bias=False),
        nn.SiLU(),
        nn.Linear(4 * C, C, bias=False),
    ).to(device=device, dtype=dtype)

    x = torch.randn(B, L, NC, device=device, dtype=dtype, requires_grad=True)

    # ref (用 _forward_ref)
    global _FUSED_MHC_ENABLED
    prev = _FUSED_MHC_ENABLED
    _FUSED_MHC_ENABLED = False
    out_ref = mhc(x, sublayer_fn)
    loss_ref = out_ref.float().pow(2).mean()
    loss_ref.backward()
    grads_ref = {k: v.grad.detach().clone() for k, v in mhc.named_parameters()}
    grads_ref['x'] = x.grad.detach().clone()

    # 清梯度
    x.grad = None
    for p in mhc.parameters(): p.grad = None

    # fused
    _FUSED_MHC_ENABLED = True
    out_f = mhc(x, sublayer_fn)
    loss_f = out_f.float().pow(2).mean()
    loss_f.backward()

    _FUSED_MHC_ENABLED = prev

    print(f"[end2end B={B} L={L} n={n} C={C} dtype={str(dtype).replace('torch.','')}]")
    _check("out", out_ref, out_f, atol, rtol)
    _check("grad_x", grads_ref['x'], x.grad, atol, rtol)
    for name, p in mhc.named_parameters():
        if p.grad is None: continue
        _check(f"grad_{name}", grads_ref[name], p.grad, atol*5, rtol*5)
    print()


def _bench(B, L, n, C, dtype, warmup=10, runs=50):
    """对比 ref / fused 端到端 forward+backward 耗时。"""
    device = 'cuda'
    NC = n * C

    sublayer_fn = nn.Sequential(
        nn.Linear(C, 4 * C, bias=False),
        nn.SiLU(),
        nn.Linear(4 * C, C, bias=False),
    ).to(device=device, dtype=dtype)

    mhc = MHCConnection_Fused(n, C).to(device=device, dtype=dtype)
    x_template = torch.randn(B, L, NC, device=device, dtype=dtype)

    def _one_step(use_fused: bool):
        global _FUSED_MHC_ENABLED
        prev = _FUSED_MHC_ENABLED
        _FUSED_MHC_ENABLED = use_fused
        try:
            x = x_template.clone().requires_grad_(True)
            out = mhc(x, sublayer_fn)
            loss = out.float().pow(2).mean()
            loss.backward()
        finally:
            _FUSED_MHC_ENABLED = prev

    for _ in range(warmup):
        _one_step(True); _one_step(False)
        for p in mhc.parameters():
            if p.grad is not None: p.grad = None
        for p in sublayer_fn.parameters():
            if p.grad is not None: p.grad = None

    def _measure(use_fused):
        torch.cuda.synchronize()
        st = torch.cuda.Event(enable_timing=True); ed = torch.cuda.Event(enable_timing=True)
        st.record()
        for _ in range(runs):
            _one_step(use_fused)
            for p in mhc.parameters():
                if p.grad is not None: p.grad = None
            for p in sublayer_fn.parameters():
                if p.grad is not None: p.grad = None
        ed.record(); torch.cuda.synchronize()
        return st.elapsed_time(ed) / runs

    t_fused = _measure(True)
    t_ref   = _measure(False)
    print(f"[bench B={B} L={L} n={n} C={C} dtype={str(dtype).replace('torch.','')}]"
          f"  ref={t_ref:.3f}ms  fused={t_fused:.3f}ms  speedup={t_ref/t_fused:.2f}x")


if __name__ == "__main__":
    if not (torch.cuda.is_available() and TRITON_AVAILABLE):
        print("[skip] 需要 CUDA + Triton")
        raise SystemExit(0)

    print("=" * 78)
    print("mhc_fused.py: 数值正确性测试")
    print("=" * 78)

    # POST phase（含 H_res matmul）
    _test_post_phase_full(B=2, L=64, n=4, C=128, dtype=torch.float32, atol=1e-5, rtol=1e-5)
    _test_post_phase_full(B=2, L=64, n=4, C=128, dtype=torch.bfloat16, atol=5e-2, rtol=5e-3)

    # PRE phase（含 res 段）
    _test_pre_phase_full(B=2, L=64, n=4, C=128, dtype=torch.float32, atol=2e-4, rtol=2e-4)
    _test_pre_phase_full(B=2, L=64, n=4, C=128, dtype=torch.bfloat16, atol=1e-1, rtol=1e-2)

    # 端到端（含 SK）
    _test_end2end(B=2, L=64, n=4, C=128, dtype=torch.bfloat16, atol=2e-1, rtol=2e-2)

    print("=" * 78)
    print("mhc_fused.py: 性能 benchmark（ref vs fused）")
    print("=" * 78)
    for cfg in [
        (4, 512,  4, 256),
        (4, 1024, 4, 512),
        (2, 2048, 8, 512),
    ]:
        _bench(*cfg, dtype=torch.bfloat16)

    print("\n所有测试通过 ✓")
