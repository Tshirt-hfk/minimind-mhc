"""mHC entry/exit 通用组件，被 mhc_fused.py 和 mhc_fused_no_hres.py 共用。

  - StreamExpand(n, D) : (B,L,D) → (B,L,n·D) 对称复制（无参）
  - MHCHead(n, D)      : (B,L,n·D) → (B,L,D) Triton fused
                         （RMSNorm + phi GEMM + α·+β + σ + ε + weighted reduce）

MHCHead 参数化简化：原版 `RMSNorm(NC).weight ⊗ phi` 与 `diag(w_rms) @ phi` 重参数化
等价，去掉单独的 w_rms 后表达力完全保留，fused bwd 不再需要 grad_w_rms 的全 batch
reduce + atomic_add。
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

# env 一键禁用 fused MHCHead（仍走 PyTorch ref）
_FUSED_MHC_HEAD_ENABLED = os.environ.get("MINIMIND_FUSED_MHC_HEAD", "1") != "0"
# env 一键禁用 Triton autotune（首次 warmup 慢；调试 / CI 可关）
_TRITON_AUTOTUNE_ENABLED = os.environ.get("MINIMIND_TRITON_AUTOTUNE", "1") != "0"
# H1.2 — env 一键禁用 W block_ptr / TMA 路径（强制旧 mask+stride load；老 Triton 兼容 / debug）
_USE_BLOCK_PTR_W = os.environ.get("MINIMIND_TRITON_BLOCK_PTR", "1") != "0"


def _is_pow2(n: int) -> bool:
    return n >= 1 and (n & (n - 1)) == 0


# ──────────────────────────────────────────────────────────────────────────────
# H0.2 — Device-aware SRAM 上限：
#   A100 (sm_80)        : 192 KB   |  H100/H800 (sm_90): 228 KB
#   A40/L40 (sm_86/89)  : 100 KB   |  其他默认            : 96  KB（保守）
# ──────────────────────────────────────────────────────────────────────────────
def max_sram_bytes(device=None) -> int:
    """返回当前 GPU 的单 SM SRAM 上限（bytes）；不在 CUDA 上时返回 192KB 保守值。"""
    if not torch.cuda.is_available():
        return 192 * 1024
    try:
        major, minor = torch.cuda.get_device_capability(device)
    except Exception:
        return 192 * 1024
    cc = major * 10 + minor
    if cc >= 90:       # Hopper：H100 / H800
        return 228 * 1024
    if cc == 80:       # Ampere：A100
        return 192 * 1024
    if cc in (86, 89): # Ampere consumer：A40 / L40 / RTX 30xx/40xx
        return 100 * 1024
    return 96 * 1024


# ──────────────────────────────────────────────────────────────────────────────
# H0.1 — Triton autotune 配置池（仅 num_warps / num_stages 调参；BLOCK 是 constexpr 固定）
#   HEAVY  : PRE-phase fwd/bwd（W ≥ 16 KB、寄存器压力大），H800 上深 pipeline 收益高
#   MEDIUM : MHCHead fwd/bwd（W 较小但 BL × NC fused 路径）
#   LIGHT  : POST-phase / Sinkhorn（小 SRAM、低延迟）
# 通过 env 关闭：MINIMIND_TRITON_AUTOTUNE=0 → 退回 num_warps=4 num_stages=2 默认
#
# H1.2 — W 通过 tl.make_block_ptr 加载（替代 strided ptr + mask）；
#   Hopper (sm_90+)：Triton 编译器自动 lower 到 cp.async.bulk.tensor（TMA 硬件指令）；
#   Ampere/Volta   ：fallback 到 cp.async（无硬件 TMA）；
#   省 mask 计算 + 与 RMSNorm 计算 async overlap（配合 num_stages=3-4 软件流水）。
#   通过 env 关闭：MINIMIND_TRITON_BLOCK_PTR=0 → 退回旧 mask+stride load（兼容 / debug）。
# ──────────────────────────────────────────────────────────────────────────────
if TRITON_AVAILABLE:
    if _TRITON_AUTOTUNE_ENABLED:
        AUTOTUNE_CFGS_HEAVY = [
            triton.Config({}, num_warps=4, num_stages=2),
            triton.Config({}, num_warps=4, num_stages=3),
            triton.Config({}, num_warps=8, num_stages=2),
            triton.Config({}, num_warps=8, num_stages=3),
            triton.Config({}, num_warps=8, num_stages=4),  # Hopper deep pipeline
        ]
        AUTOTUNE_CFGS_MEDIUM = [
            triton.Config({}, num_warps=4, num_stages=2),
            triton.Config({}, num_warps=4, num_stages=3),
            triton.Config({}, num_warps=8, num_stages=3),
        ]
        AUTOTUNE_CFGS_LIGHT = [
            triton.Config({}, num_warps=2, num_stages=2),
            triton.Config({}, num_warps=4, num_stages=2),
            triton.Config({}, num_warps=4, num_stages=3),
        ]
    else:
        # 关闭 autotune：退回单一默认配置（与历史行为一致）
        AUTOTUNE_CFGS_HEAVY  = [triton.Config({}, num_warps=4, num_stages=2)]
        AUTOTUNE_CFGS_MEDIUM = [triton.Config({}, num_warps=4, num_stages=2)]
        AUTOTUNE_CFGS_LIGHT  = [triton.Config({}, num_warps=4, num_stages=2)]
else:
    # 无 Triton 环境（CPU-only / 单测）下仍允许 import 该 module；
    # 这些常量永远不会被实际使用（kernel 装饰器在 TRITON_AVAILABLE 分支内部）
    AUTOTUNE_CFGS_HEAVY  = []
    AUTOTUNE_CFGS_MEDIUM = []
    AUTOTUNE_CFGS_LIGHT  = []


class StreamExpand(nn.Module):
    """(B, L, D) → (B, L, n·D)：对称复制 n 份（无参）。

    init 时 n 路对称使 H^res·x = x，但 H^post 经 phi Kaiming init 在各路上不同，
    write_back 立即破坏对称，后续层 H^res 即可驱动梯度。
    """
    def __init__(self, n: int, dim: int):
        super().__init__()
        self.n = n
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.n == 1:
            return x
        B, L, D = x.shape
        return x.unsqueeze(-2).expand(B, L, self.n, D).reshape(B, L, self.n * D)


# ──────────────────────────────────────────────────────────────────────────────
# MHCHead fused Triton kernels
# ──────────────────────────────────────────────────────────────────────────────

if TRITON_AVAILABLE:

    @triton.autotune(configs=AUTOTUNE_CFGS_MEDIUM, key=['NC', 'C', 'N'])
    @triton.jit
    def _mhc_head_fwd_kernel(
        X_ptr,              # (BL, NC)        x_flat
        Wphi_ptr,           # (NC, N)         phi_weight, row-major
        Alpha_ptr,          # (1,) fp32
        Bias_ptr,           # (N,) fp32
        # ── 输出 ──
        Out_ptr,            # (BL, C)
        # ── 反向保存 ──
        InvRms_ptr,         # (BL,) fp32
        MixRaw_ptr,         # (BL, N) fp32   x_norm @ phi 的 raw（未仿射）
        # ── 标量 ──
        pre_eps,
        rms_eps,
        # ── 编译期常量 ──
        NC: tl.constexpr,
        C: tl.constexpr,
        N: tl.constexpr,
        BLOCK_NC: tl.constexpr,
        USE_BLOCK_PTR: tl.constexpr,    # H1.2: block_ptr / TMA path
    ):
        """1 program / 1 token: RMSNorm → phi GEMM → α+β+σ+ε → weighted reduce。
        W 全 SRAM 驻留（NC·N·4B ≪ 192KB），消除 mix_act/mix/weighted_temp 中间张量。
        """
        pid = tl.program_id(0)

        offs_nc = tl.arange(0, BLOCK_NC)
        mask_nc = offs_nc < NC
        offs_n  = tl.arange(0, N)
        offs_c  = tl.arange(0, C)

        # ── Stage 1: load x + RMSNorm ──
        x = tl.load(X_ptr + pid * NC + offs_nc, mask=mask_nc, other=0.0).to(tl.float32)
        sum_sq = tl.sum(x * x)
        inv_rms = tl.rsqrt(sum_sq / NC + rms_eps)
        tl.store(InvRms_ptr + pid, inv_rms)
        x_norm = x * inv_rms                                                # (BLOCK_NC,) fp32

        # ── Stage 2: phi GEMM (W 全 SRAM) ──
        # H1.2: block_ptr 路径在 Hopper 上由 Triton 编译器自动 lower 到 cp.async.bulk.tensor
        # (TMA)；省 mask 计算 + async overlap RMSNorm 计算。BLOCK_NC == NC 已 assert，无越界。
        if USE_BLOCK_PTR:
            w_bptr = tl.make_block_ptr(
                base=Wphi_ptr,
                shape=(NC, N), strides=(N, 1),
                offsets=(0, 0), block_shape=(BLOCK_NC, N),
                order=(1, 0),
            )
            w = tl.load(w_bptr).to(tl.float32)                              # (BLOCK_NC, N)
        else:
            w = tl.load(
                Wphi_ptr + offs_nc[:, None] * N + offs_n[None, :],
                mask=mask_nc[:, None], other=0.0,
            ).to(tl.float32)                                                # (BLOCK_NC, N)
        mix_raw = tl.sum(x_norm[:, None] * w, axis=0)                       # (N,) fp32
        tl.store(MixRaw_ptr + pid * N + offs_n, mix_raw)

        # ── Stage 3: α + β + σ + ε ──
        alpha = tl.load(Alpha_ptr).to(tl.float32)
        bias  = tl.load(Bias_ptr + offs_n).to(tl.float32)
        mix_act = mix_raw * alpha + bias
        mix = tl.sigmoid(mix_act) + pre_eps                                 # (N,) fp32

        # ── Stage 4: weighted reduce（x 寄存器复用，省第二次 HBM load）──
        x_2d = tl.reshape(x, (N, C))                                        # (N, C) view of stage-1 load
        out = tl.sum(mix[:, None] * x_2d, axis=0)                           # (C,) fp32
        tl.store(Out_ptr + pid * C + offs_c, out.to(Out_ptr.dtype.element_ty))


    @triton.autotune(configs=AUTOTUNE_CFGS_MEDIUM, key=['NC', 'C', 'N'])
    @triton.jit
    def _mhc_head_bwd_kernel(
        # ── forward 保存 ──
        X_ptr,              # (BL, NC)
        Wphi_ptr,           # (NC, N)
        Alpha_ptr,          # (1,) fp32
        Bias_ptr,           # (N,) fp32       仅用于校验/未来扩展；当前 bwd 不读
        InvRms_ptr,         # (BL,) fp32
        MixRaw_ptr,         # (BL, N) fp32
        # ── 上游 grad ──
        GradOut_ptr,        # (BL, C)
        # ── 输出 grad ──
        GradX_ptr,          # (BL, NC)
        GradMixRaw_ptr,     # (BL, N) fp32   用于 cuBLAS 算 grad_phi_W
        # ── per-token grad（用于 Python 端 reduce，规避 atomic_add 竞争）──
        GradAlphaPerTok_ptr,    # (BL,) fp32     scalar α grad per token
        GradMixActPerTok_ptr,   # (BL, N) fp32   = grad_bias per token（β grad）
        # ── 标量 ──
        rms_eps,
        # ── 编译期常量 ──
        NC: tl.constexpr,
        C: tl.constexpr,
        N: tl.constexpr,
        BLOCK_NC: tl.constexpr,
        USE_BLOCK_PTR: tl.constexpr,    # H1.2: block_ptr / TMA path
    ):
        """1 program / 1 token: reduce + sigmoid + α/β + phi + RMSNorm 反向全融合。
        α/β grad 以 per-token store 累计，Python 端 PyTorch sum reduce（零 atomic 竞争）；
        grad_mix_raw 写回供 cuBLAS 单次算 grad_W。
        """
        pid = tl.program_id(0)

        offs_nc = tl.arange(0, BLOCK_NC)
        mask_nc = offs_nc < NC
        offs_n  = tl.arange(0, N)
        offs_c  = tl.arange(0, C)

        # ── load forward 中保存的张量 ──
        x        = tl.load(X_ptr + pid * NC + offs_nc, mask=mask_nc, other=0.0).to(tl.float32)
        inv_rms  = tl.load(InvRms_ptr + pid).to(tl.float32)
        mix_raw  = tl.load(MixRaw_ptr + pid * N + offs_n).to(tl.float32)
        alpha    = tl.load(Alpha_ptr).to(tl.float32)
        bias     = tl.load(Bias_ptr  + offs_n).to(tl.float32)
        grad_out = tl.load(GradOut_ptr + pid * C + offs_c).to(tl.float32)   # (C,)

        x_norm = x * inv_rms

        # 重算 sigmoid（避免存额外中间张量）
        mix_act = mix_raw * alpha + bias
        sigmoid_mix = tl.sigmoid(mix_act)
        # mix 用于 grad_x_via_reduce；+ε 不影响梯度
        mix = sigmoid_mix

        # ── 1) grad_mix[i] = Σ_c grad_out[c] · x[i, c]（x 寄存器复用）──
        x_2d = tl.reshape(x, (N, C))                                        # (N, C) view
        grad_mix = tl.sum(grad_out[None, :] * x_2d, axis=1)                 # (N,)

        # ── 2) sigmoid 反向 ──
        grad_mix_act = grad_mix * sigmoid_mix * (1.0 - sigmoid_mix)         # (N,)

        # ── 3) α + β 反向（per-token store，Python 端 reduce）──
        grad_mix_raw = grad_mix_act * alpha                                 # (N,)
        grad_alpha_local = tl.sum(grad_mix_act * mix_raw)                   # scalar
        tl.store(GradAlphaPerTok_ptr  + pid, grad_alpha_local)              # (BL,)
        tl.store(GradMixActPerTok_ptr + pid * N + offs_n, grad_mix_act)     # (BL, N)

        # 存 grad_mix_raw 供 cuBLAS 算 grad_phi_W
        tl.store(GradMixRaw_ptr + pid * N + offs_n, grad_mix_raw)

        # ── 4) phi 反向 grad_x_norm = grad_mix_raw @ W.T （per-token mini-GEMM）──
        # H1.2: 同 fwd，W 通过 block_ptr 走 TMA / async 路径
        if USE_BLOCK_PTR:
            w_bptr = tl.make_block_ptr(
                base=Wphi_ptr,
                shape=(NC, N), strides=(N, 1),
                offsets=(0, 0), block_shape=(BLOCK_NC, N),
                order=(1, 0),
            )
            w = tl.load(w_bptr).to(tl.float32)                              # (BLOCK_NC, N)
        else:
            w = tl.load(
                Wphi_ptr + offs_nc[:, None] * N + offs_n[None, :],
                mask=mask_nc[:, None], other=0.0,
            ).to(tl.float32)                                                # (BLOCK_NC, N)
        grad_x_norm = tl.sum(grad_mix_raw[None, :] * w, axis=1)             # (BLOCK_NC,)

        # ── 5) RMSNorm 反向（无 weight 形式）──
        mean_y_gy = tl.sum(x_norm * grad_x_norm) / NC
        grad_x_via_phi = inv_rms * (grad_x_norm - x_norm * mean_y_gy)       # (BLOCK_NC,)

        # ── 6) 合并 grad_x_via_phi 与 grad_x_via_reduce（mix · grad_out, (N,C)）──
        grad_x_via_reduce = mix[:, None] * grad_out[None, :]                # (N, C)
        grad_x_via_phi_2d = tl.reshape(grad_x_via_phi, (N, C))              # (N, C) view
        grad_x_total = grad_x_via_reduce + grad_x_via_phi_2d
        offs_2d = offs_n[:, None] * C + offs_c[None, :]
        tl.store(GradX_ptr + pid * NC + offs_2d,
                 grad_x_total.to(GradX_ptr.dtype.element_ty))


def _mhc_head_ref(x_flat, w_phi, alpha, bias, pre_eps, rms_eps, n, c):
    """PyTorch ref（fallback / 比对）：unweighted RMSNorm + phi + σ + reduce。"""
    B, L, NC = x_flat.shape
    assert NC == n * c
    x_f = x_flat.float()
    inv_rms = x_f.pow(2).mean(-1, keepdim=True).add(rms_eps).rsqrt()
    x_norm = (x_f * inv_rms).to(x_flat.dtype)
    mix_raw = x_norm @ w_phi                                                # (B, L, n)
    mix = torch.sigmoid(mix_raw * alpha + bias) + pre_eps                   # (B, L, n)
    x_2d = x_flat.view(B, L, n, c)
    return (mix.unsqueeze(-1) * x_2d).sum(dim=-2)                           # (B, L, c)


class _MHCHeadFn(torch.autograd.Function):
    """MHCHead fused fwd + bwd（bwd 末尾再 cuBLAS GEMM 算 grad_phi_W）。"""

    @staticmethod
    def forward(ctx, x_flat, w_phi, alpha, bias, pre_eps, rms_eps, n, c):
        # x_flat (B,L,n·c) · w_phi (n·c, n) row-major · alpha (1,) · bias (n,) → out (B,L,c)
        assert x_flat.is_cuda and TRITON_AVAILABLE
        B, L, NC = x_flat.shape
        BL = B * L
        assert w_phi.shape == (NC, n)

        x_flat_c = x_flat.contiguous().view(BL, NC)
        w_phi_c  = w_phi.contiguous()

        out      = torch.empty(BL, c,    dtype=x_flat.dtype, device=x_flat.device)
        inv_rms  = torch.empty(BL,       dtype=torch.float32, device=x_flat.device)
        mix_raw  = torch.empty(BL, n,    dtype=torch.float32, device=x_flat.device)

        BLOCK_NC = triton.next_power_of_2(NC)
        assert BLOCK_NC == NC, (
            f"n·dim={NC} 必须是 2 的幂（当前 next_pow2={BLOCK_NC}）；"
            f"实际项目里 n ∈ {{1,2,4,8,16}}、dim ∈ {{256,512,1024,2048}} 都满足"
        )
        # SRAM 校验：W 主导（NC·n·4B fp32），device-aware 上限（H800 228KB / A100 192KB）
        sram_bytes = NC * 4 + NC * n * 4
        sram_limit = max_sram_bytes(x_flat.device)
        assert sram_bytes <= sram_limit, (
            f"NC={NC} n={n} 需要 {sram_bytes/1024:.0f}KB SRAM；"
            f"超出当前 GPU 上限 {sram_limit/1024:.0f}KB"
        )

        alpha_f = alpha.float().reshape(1)

        _mhc_head_fwd_kernel[(BL,)](
            x_flat_c, w_phi_c,
            alpha_f, bias.float(),
            out, inv_rms, mix_raw,
            float(pre_eps), float(rms_eps),
            NC=NC, C=c, N=n, BLOCK_NC=BLOCK_NC,
            USE_BLOCK_PTR=_USE_BLOCK_PTR_W,
        )

        ctx.save_for_backward(x_flat_c, w_phi_c, alpha_f, bias, inv_rms, mix_raw)
        ctx.rms_eps = float(rms_eps)
        ctx.n = n; ctx.c = c
        ctx.shape = (B, L, NC)
        return out.view(B, L, c)

    @staticmethod
    def backward(ctx, grad_out):
        (x_flat_c, w_phi_c, alpha_f, bias, inv_rms, mix_raw) = ctx.saved_tensors
        B, L, NC = ctx.shape
        BL = B * L
        n  = ctx.n
        c  = ctx.c

        grad_out_c = grad_out.contiguous().view(BL, c)

        grad_x       = torch.empty(BL, NC, dtype=x_flat_c.dtype, device=x_flat_c.device)
        grad_mix_raw = torch.empty(BL, n,  dtype=torch.float32, device=x_flat_c.device)
        # per-token α/β grad（避免 atomic_add 序列化竞争；Python 端 reduce）
        grad_alpha_per_tok   = torch.empty(BL,    dtype=torch.float32, device=x_flat_c.device)
        grad_mix_act_per_tok = torch.empty(BL, n, dtype=torch.float32, device=x_flat_c.device)

        BLOCK_NC = NC                                                       # forward 已 assert pow-of-2

        _mhc_head_bwd_kernel[(BL,)](
            x_flat_c, w_phi_c,
            alpha_f, bias.float(),
            inv_rms, mix_raw,
            grad_out_c,
            grad_x, grad_mix_raw,
            grad_alpha_per_tok, grad_mix_act_per_tok,
            ctx.rms_eps,
            NC=NC, C=c, N=n, BLOCK_NC=BLOCK_NC,
            USE_BLOCK_PTR=_USE_BLOCK_PTR_W,
        )

        # grad_W = x_norm.T @ grad_mix_raw （单次 cuBLAS GEMM）
        x_norm = x_flat_c.float() * inv_rms.unsqueeze(-1)                   # (BL, NC) fp32
        grad_w_phi = (x_norm.T @ grad_mix_raw).to(w_phi_c.dtype)            # (NC, n)

        # α/β grad：单次 PyTorch reduce 替代 BL 个 atomic_add（零竞争）
        grad_alpha = grad_alpha_per_tok.sum().view(1)                       # (1,)
        grad_bias  = grad_mix_act_per_tok.sum(dim=0)                        # (n,)

        # grad_alpha 必须保持 (1,) 形状（对齐 alpha = nn.Parameter(torch.full((1,), ...))）
        return (grad_x.view(B, L, NC), grad_w_phi,
                grad_alpha.to(alpha_f.dtype), grad_bias.to(bias.dtype),
                None, None, None, None)


class MHCHead(nn.Module):
    """(B, L, n·D) → (B, L, D)：mHC 可学加权 reduce（替代 StreamReduce，解决最后
    一层 H_res frozen 死锁，对齐官方 mhc_head 的 sigmoid·x sum 结构）。

    公式：
        x̃ = RMSNorm(vec(x))         # unweighted（简化，详见 module docstring）
        out = Σᵢ (σ(α·(x̃·φ) + β) + ε)ᵢ · xᵢ

    env: `MINIMIND_FUSED_MHC_HEAD=0` 可禁用 fused 路径
    """

    def __init__(self, n: int, dim: int, alpha_init: float = 0.01,
                 eps: float = 1e-6, pre_eps: float = 1e-6):
        super().__init__()
        assert _is_pow2(n) and n <= 16, f"n 必须 ∈ {{1,2,4,8,16}}，得到 {n}"
        self.n = n
        self.dim = dim
        self.nC = n * dim
        self.eps = eps
        self.pre_eps = pre_eps

        # phi: (NC, n) row-major（与 mhc_fused.py phi_weight 同风格，便于 Triton 索引）
        # 保留 autograd 对 phi_weight 的梯度连接
        self.phi_weight = nn.Parameter(torch.empty(self.nC, n))
        nn.init.kaiming_uniform_(self.phi_weight, a=math.sqrt(5))
        self.bias = nn.Parameter(torch.zeros(n))
        self.alpha = nn.Parameter(torch.full((1,), alpha_init))

    def extra_repr(self) -> str:
        return (f'n={self.n}, dim={self.dim}, fused=True, '
                f'input_norm=\'global\'(unweighted RMS, simplified param)')

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        B, L, _ = x_flat.shape
        n, C = self.n, self.dim
        if n == 1:
            # 单路退化：仅 RMSNorm
            x_f = x_flat.float()
            inv_rms = x_f.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
            return (x_f * inv_rms).to(x_flat.dtype).view(B, L, C)

        if _FUSED_MHC_HEAD_ENABLED and TRITON_AVAILABLE and x_flat.is_cuda:
            return _MHCHeadFn.apply(
                x_flat, self.phi_weight, self.alpha, self.bias,
                self.pre_eps, self.eps, n, C,
            )
        return _mhc_head_ref(x_flat, self.phi_weight, self.alpha, self.bias,
                             self.pre_eps, self.eps, n, C)
