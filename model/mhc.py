"""
model/mhc.py
==============================================================================
Manifold-Constrained Hyper-Connections (mHC), DeepSeek 2025, arXiv:2512.24880.

本实现参考并对齐 DeepSeek 官方 TileKernels 仓库的 PyTorch reference：
  https://github.com/deepseek-ai/TileKernels/blob/main/tile_kernels/torch/mhc.py
  https://github.com/deepseek-ai/TileKernels/blob/main/tile_kernels/modeling/mhc/functional.py

核心思想
------------------------------------------------------------------------------
把残差流加宽到 n 路 (x ∈ R^{n×C})，每个 sublayer F 的更新公式为
    x_{l+1} = H^res · x_l + H^post.T · F(H^pre · x_l)
其中三个系数矩阵 H 都是输入相关 dynamic + 全局 static：
    x̃ = RMSNorm(vec(x))                                # vec: (n,C) → (1, n·C)
    [m_pre | m_post | m_res] = α ⊙ (x̃ · φ) + b         # ∈ R^{1, 2n+n²}  (fused)
    H^pre  = σ(m_pre)              ∈ R^{1×n}            # 非负，每路 ∈ (0, 1+ε)
    H^post = post_mult · σ(m_post) ∈ R^{1×n}            # 非负，每路 ∈ (0, post_mult)
    H^res  = Sinkhorn-Knopp(m_res) ∈ R^{n×n}            # doubly-stochastic

H^res 落在 Birkhoff polytope 上是关键 —— doubly stochastic 矩阵的乘积仍是
doubly stochastic，因此 ∏_l H_l^res 永远保持谱范数 ≤ 1（mass-conserving），
解决了原版 HC 在大规模训练中信号无界放大/衰减的稳定性问题（论文 §3.1
报告 27B 模型 HC composite gain 峰值 ~3000，mHC 仅 ~1.6）。

本版本相对前版的优化（吸收自 DeepSeek 官方 TileKernels）
------------------------------------------------------------------------------
1. **Fused φ Linear**：3 个独立 Linear (phi_pre, phi_post, phi_res) → 1 个共享
   Linear，输出维度 (2n + n²)。GEMM 数量 3→1，CPU 上 launch overhead 降低，
   GPU 上访存吞吐和算子融合度提升。
2. **共享 α / b**：alpha 是 shape (3,) 的单一 Parameter（pre/post/res 各占一项），
   bias 是 shape (2n+n²,) 的单一 Parameter。前向时 alpha 用 expand 广播到对应段。
3. **Sinkhorn 用 softmax 起步**：第一次 row normalize 用 softmax 完成（合并 exp
   + 减 max + row norm），后续做 iters-1 次 col + row 交替；相比"exp + iters 次
   col/row"更简洁、数值更稳、op 数略低。
4. **StreamExpand 替代 StreamLift**（entry）：对称复制 n 份（对齐官方
   expand_to_mhc_ref）。phi 的 Kaiming init 使 H^pre/H^post 在 n 路上各异，
   layer 0 的 write_back 已经打破对称（mixed = x[0] 复制 n 次，但
   write_back[i] = H_post[i]·F_out 是 n 路不对称），后续层 H_res 自然驱动。
   ⚠️ 注意：b_res 仍然保留 eye(n) 初始化（非"打破对称"用途，而是让 SK 雅可比
   远离 uniform 附近的梯度压制区域，详见 MHCConnection.__init__ 注释）。
5. **MHCHead 替代 StreamReduce**（exit）：用 mHC 风格的可学加权 reduce 替代无参
   row-wise sum。这天然解开了前版"最后一层 H_res frozen"的固有死锁
   （StreamReduce 是 row-sum，doubly-stochastic H_res 列和=1 让最后一层 H_res
   梯度恒 0），改用可学权重后梯度路径打开。同时也对齐官方 mhc_head。

接口
------------------------------------------------------------------------------
MHCConnection(n, dim).forward(x, sublayer_fn):
    - x: (B, L, n·dim)            # flat 残差流形态（n 路 hyper-connection）
    - sublayer_fn: (B, L, dim) → (B, L, dim)
    - returns: (B, L, n·dim)

StreamExpand(n, dim).forward(x):   (B, L, dim) → (B, L, n·dim)  # 对称复制，无参数
MHCHead(n, dim).forward(x):        (B, L, n·dim) → (B, L, dim)  # 可学加权 reduce

兼容性：StreamLift / StreamReduce 已被替代，旧符号已不再导出（model_minimind.py
同步迁移到 StreamExpand / MHCHead）。

Ablation 开关
------------------------------------------------------------------------------
MHCConnection(..., disable_h_res=True)：复现论文 Table 1 中 H^res 的禁用对照
    （论文 Table 1 标题：H^res 禁用时用 "identity matrix" 替代）。该 flag 为
    True 时跳过 Sinkhorn-Knopp 投影，直接令 H^res = I_n（n×n 单位阵，按 batch
    维度零拷贝 expand）。等价于切断 n 路残差流之间的所有信息混合 —— 每一路只
    看自己的历史 + 写回。

    论文 Table 1 报告：H^res 单独贡献了约 81% 的总收益（-0.022 / -0.027），即
    所有三个映射中 H^res 最关键且不可替代；H^pre / H^post 提供的功能可被下游
    H^res 部分补偿，反之不成立。因此本 flag 是"功能性消融"中最锐利的对照。

    使用方式：MiniMindConfig(mhc_disable_h_res=True) → 经 MiniMindBlock 透传。

Sinkhorn 后端选择
------------------------------------------------------------------------------
  - CUDA + Triton + n 是 2 的幂（n ∈ {1,2,4,8,16}）时走 fused kernel：
    softmax 起步 + 全部 iters 次 row/col 交替合并为单个 program，n×n 全部在
    寄存器内迭代，规避 PyTorch 路径下 iters 次 op launch + 中间张量 HBM 往返。
    实测 n=4 / BL=8192 / iters=10 在 bf16 下 fwd 加速 ~10x，fwd+bwd 加速 ~2-3x
    （bwd 走 recompute，仍调 ref 实现，详见 _SinkhornFn 注释）。
  - 其他场景透明 fallback 到 PyTorch ref 实现。
  - 环境变量 MINIMIND_FUSED_SINKHORN=0 可一键禁用（用于排障 / A-B 对比）。
"""

import os
import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

# 环境变量 MINIMIND_FUSED_SINKHORN=0 可一键禁用 fused 路径
_FUSED_SINKHORN_ENABLED = os.environ.get("MINIMIND_FUSED_SINKHORN", "1") != "0"


def _is_pow2(n: int) -> bool:
    return n >= 1 and (n & (n - 1)) == 0


# ──────────────────────────────────────────────────────────────────────────────
# Sinkhorn-Knopp 投影
#   - _sinkhorn_ref       : PyTorch fp32 reference（对齐官方 sinkhorn_normalize_ref）
#   - _sinkhorn_fwd_kernel: Triton fused forward kernel
#   - _SinkhornFn         : autograd.Function（fused forward + recompute backward）
#   - sinkhorn_normalize  : 对外 API，按条件选择后端
# ──────────────────────────────────────────────────────────────────────────────

def _sinkhorn_ref(M: torch.Tensor, iters: int, eps: float) -> torch.Tensor:
    """PyTorch reference 实现（与原 sinkhorn_normalize 完全一致）。

    实现对齐 DeepSeek 官方 TileKernels (tile_kernels/torch/mhc.py)：
      1) 首次 row normalize 用 softmax(-1) 一步完成（exp + 减 max + 归一化），
         数值上等价于 "x = exp(x - row_max); x = x / row_sum"，但更简洁稳定。
      2) 然后做 1 次 col norm + (iters-1) 次 (row, col) 交替。
      3) eps 仅在 0/0 退化情形作 saturation 兜底；分子的小 eps 避免行/列 mass→0
         时除法爆炸（官方做法）。
    """
    x = M.softmax(dim=-1) + eps
    x = x / (x.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        x = x / (x.sum(dim=-1, keepdim=True) + eps)
        x = x / (x.sum(dim=-2, keepdim=True) + eps)
    return x


if TRITON_AVAILABLE:

    @triton.jit
    def _sinkhorn_fwd_kernel(
        M_ptr,                  # (BL, N, N) contiguous
        Y_ptr,                  # (BL, N, N) contiguous
        eps,                    # float scalar
        N: tl.constexpr,        # 矩阵阶数（必须是 2 的幂）
        ITERS: tl.constexpr,    # 迭代次数（编译期常量，循环全展开）
    ):
        """单 program 处理一个 N×N 矩阵的完整 SK 投影。

        关键点：
          - N×N 全部驻留寄存器（N ≤ 16 → ≤ 256 elements，寄存器充裕）
          - softmax 起步 + (ITERS-1) 次 row/col 交替全部展开到单 program
          - 全程 fp32 计算（外部 autocast 已 disable）
          - 写回时按 Y_ptr.dtype 强制转换（保持与 ref dtype 一致；
            内部 fp32 → bf16/fp16 输出时 fused 比 ref 精度更高，但对齐误差容忍内）
        """
        pid = tl.program_id(0)

        # N×N 索引（rows, cols），全展开
        rows = tl.arange(0, N)[:, None]                # (N, 1)
        cols = tl.arange(0, N)[None, :]                # (1, N)
        offs = pid * N * N + rows * N + cols           # (N, N)

        # 加载 N×N 矩阵到寄存器（fp32 计算）
        x = tl.load(M_ptr + offs).to(tl.float32)

        # ── 第 1 次 row normalize：softmax(dim=-1) + eps（含数值稳定的减 max）
        row_max = tl.max(x, axis=1)[:, None]           # (N, 1)
        x = tl.exp(x - row_max)                        # 减 max 后 exp，不溢出
        x = x / tl.sum(x, axis=1)[:, None]             # row sum 归一
        x = x + eps                                    # eps saturation 兜底

        # ── 第 1 次 col normalize
        x = x / (tl.sum(x, axis=0)[None, :] + eps)     # col sum + eps 兜底

        # ── 剩余 (ITERS-1) 次 row/col 交替（编译期全展开，无运行时循环开销）
        for _ in tl.static_range(0, ITERS - 1):
            x = x / (tl.sum(x, axis=1)[:, None] + eps)
            x = x / (tl.sum(x, axis=0)[None, :] + eps)

        # 写回（按 Y_ptr.dtype 转换；内部 fp32 → 输出 dtype）
        tl.store(Y_ptr + offs, x.to(Y_ptr.dtype.element_ty))


class _SinkhornFn(torch.autograd.Function):
    """Sinkhorn-Knopp：fused forward + recompute backward。

    forward：调用 Triton fused kernel —— softmax 起步 + iters 次 row/col 交替合并
    为单 program，N×N 全程驻留寄存器，规避 PyTorch 路径下 iters 次 op launch +
    中间张量 HBM 往返。

    backward：**recompute 策略**——只保存输入 M（小张量），反向时调 _sinkhorn_ref
    重跑 forward 并通过 autograd 求导，理由：
      1) 反向只需保存输入 M（O(N²·BL) 显存），无需保存 iters 次中间 trajectory
      2) ref 反向的 op 数和 forward 同量级，开销与未 fused 时持平（无额外代价）
      3) SK 在 mHC 训练中通常不是瓶颈（每层 attn/mlp 内部 GEMM 占绝对大头）
    若后续 profile 显示 backward 成为热点，可再扩展为手写 backward kernel（SK 每
    步反向 `dx = (dy - Σ(y·dy)) / s_eps` 可同样塞进单 program 全展开）。
    """

    @staticmethod
    def forward(ctx, M: torch.Tensor, iters: int, eps: float) -> torch.Tensor:
        # M: (..., N, N)
        orig_shape = M.shape
        N = M.shape[-1]
        assert M.shape[-2] == N, f"最后两维必须相等，得到 {tuple(M.shape)}"

        # flatten batch 维度到 BL
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
        # 在 grad-enabled + autocast disabled 上下文中重跑 ref forward + autograd
        with torch.enable_grad(), torch.amp.autocast("cuda", enabled=False):
            M_ = M.detach().requires_grad_(True)
            y_ = _sinkhorn_ref(M_, ctx.iters, ctx.eps)
            (dx,) = torch.autograd.grad(y_, M_, dy)
        return dx, None, None


@torch.amp.autocast("cuda", enabled=False)  # 强制 fp32 路径，规避 bf16 exp 溢出
def sinkhorn_normalize(M: torch.Tensor, iters: int = 10, eps: float = 1e-6) -> torch.Tensor:
    """把 raw 矩阵投影到 Birkhoff polytope（doubly stochastic 流形）。

    入参形状：(..., n, n)；最后两维做投影，前面所有维都是 batch。

    后端选择：
      - CUDA + Triton + n 是 2 的幂 时走 fused kernel（softmax + iters 次 row/col
        交替合并为单 program，n×n 全程驻留寄存器）
      - 否则透明 fallback 到 PyTorch ref 实现
      - 环境变量 MINIMIND_FUSED_SINKHORN=0 可禁用 fused 路径
    """
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


# 兼容旧调用方（如有外部脚本仍在 import sinkhorn_knopp）
sinkhorn_knopp = sinkhorn_normalize


# ──────────────────────────────────────────────────────────────────────────────
# PerStreamRMSNorm：n 路独立 weight 的 RMSNorm（mHC ablation 专用）
# ──────────────────────────────────────────────────────────────────────────────
class _PerStreamRMSNormFn(torch.autograd.Function):
    """PerStreamRMSNorm 的 memory-efficient autograd Function。

    **背景**：朴素实现 `F.rms_norm(x, ...) * weight` 的 mul 是独立 op，autograd 会保留
    中间 `x_normed` (B, L, n, D) bf16 用于计算 weight grad（∂L/∂w = Σ grad_out · x_normed）。
    32 层 × 2 sublayer = 64 个 PerStreamRMSNorm 实例时，仅这一项就占
    64 × sizeof(B·L·n·D, bf16) ≈ 8 GB 激活（B=32, L=512, n=4, D=1024）。
    这是开启 mhc_per_stream_norm 后 OOM 的根因。

    **优化**：
      - forward 只保存 (x, inv_rms, weight)；inv_rms 是 (..., n, 1) 极小张量；
        x_normed 作为 forward 内部临时变量，被 weight 乘法消费后立即释放
      - backward 重算 `x_normed = x * inv_rms`（一次 bf16 mul，O(BLnD)），
        计算 grad_weight 与 RMSNorm 标准反向 grad_x
      - 反向额外 mul kernel 相对 attn/mlp 几乎可忽略；省 ~8 GB 激活
    
    数值精度：RMS 累加与反向都在 fp32 下做（与 nn.RMSNorm 对齐），规避 bf16 平方和溢出。
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        # x: (..., n, D), weight: (n, D)
        # fp32 累加平方和 → rsqrt → 转回 x.dtype（inv_rms 张量极小，cast 开销可忽略）
        x_f = x.float()
        inv_rms_f = x_f.pow(2).mean(dim=-1, keepdim=True).add(eps).rsqrt()  # (..., n, 1) fp32
        inv_rms = inv_rms_f.to(x.dtype)                                      # 同 dtype，小张量
        # 临时 x_normed = x * inv_rms，被下一行 weight mul 消费后释放（不进 save_for_backward）
        out = (x * inv_rms) * weight                                         # (..., n, D), bf16
        # 只保存 3 个张量：x（上游本来就要存）+ inv_rms（小）+ weight（参数）
        ctx.save_for_backward(x, inv_rms, weight)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, inv_rms, weight = ctx.saved_tensors
        # 重算 x_normed —— 1 次 bf16 mul，O(BLnD)；反向 transient，函数返回后释放
        x_normed = x * inv_rms                                               # bf16, (..., n, D)
        # ∂L/∂weight = Σ_{batch dims} (grad_out * x_normed)；保留最后两维 (n, D)
        sum_dims = tuple(range(grad_out.dim() - 2))
        grad_weight = (grad_out * x_normed).sum(dim=sum_dims)                # (n, D)
        # RMSNorm 标准反向（fp32 数值稳定）：
        #   y = x · inv_rms, output = y · weight
        #   ∂L/∂y = grad_out · weight
        #   ∂L/∂x = inv_rms · (∂L/∂y - y · mean(∂L/∂y · y, dim=-1, keepdim=True))
        grad_y_f = (grad_out * weight).float()                               # fp32, (..., n, D)
        x_normed_f = x_normed.float()
        inv_rms_f = inv_rms.float()
        mean_term = (grad_y_f * x_normed_f).mean(dim=-1, keepdim=True)       # fp32, (..., n, 1)
        grad_x = (inv_rms_f * (grad_y_f - x_normed_f * mean_term)).to(x.dtype)
        return grad_x, grad_weight, None


class PerStreamRMSNorm(nn.Module):
    """每路独立 weight 的 per-stream RMSNorm（mHC per_stream_norm ablation 专用）。

    与 nn.RMSNorm(dim) 的差别：weight shape (n, D) vs (D,) → 每路独立 affine vs n 路共享。
    参数量 n·D 与整体 nn.RMSNorm(n·D) 一致，构成 strictly controlled ablation：切换
    per_stream_norm 不改参数量/init，只改 RMS grouping 与 affine 是否绑定到流。

    实现：通过 `_PerStreamRMSNormFn` 自定义 autograd Function 跑前后向，反向重算
    `x_normed = x · inv_rms`，避免保存 (B, L, n, D) bf16 中间张量。32 层网络下相比
    朴素 `F.rms_norm(x) * weight` 实现节省 ~8 GB 激活，时间几乎不增（反向多 1 次 mul）。
    详见 `_PerStreamRMSNormFn` 的 docstring。
    """

    def __init__(self, n: int, dim: int, eps: float = 1e-6):
        super().__init__()
        self.n = n
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(n, dim))  # (n, D)，broadcast 到 (..., n, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., n, D)；最后一维 D 上独立求 RMS（n 路彼此独立 = 每路独立做 RMSNorm(D)）
        return _PerStreamRMSNormFn.apply(x, self.weight, self.eps)

    def extra_repr(self) -> str:
        # 与 nn.RMSNorm 的 repr 风格一致：先 shape，再 eps
        return f'(n={self.n}, dim={self.dim}), eps={self.eps}'


# ──────────────────────────────────────────────────────────────────────────────
# MHCConnection：单 sublayer 的 mHC 残差更新单元
# ──────────────────────────────────────────────────────────────────────────────
class MHCConnection(nn.Module):
    """单 sublayer 的 mHC 残差更新单元（论文 Eq.3 + Eq.7 + Eq.8）。

    一个 MHCConnection 包裹一个 sublayer F，等价替换原本的
        x' = x + F(norm(x))
    为论文公式
        x' = H^res · x + H^post.T · F(H^pre · x)
    （内部还包含 RMSNorm + 三套 dynamic/static 系数生成）。

    参数量预算（n 通常 ≤ 16，C 即 hidden_size）：
      fused φ : nC × (2n + n²)        （单一 Linear 替代官方 3 个独立投影）
      合计 ≈ nC(n² + 2n) = n³C + 2n²C  （n=4, C=1024 → ~88K params）
      bias    : 2n + n²
      alpha   : 3 scalars
    相对 sublayer F 内部 attn/mlp 的 O(C²) 参数完全是噪音级。

    与 DeepSeek 官方 TileKernels 的字段对齐：
      self.phi.weight     ↔ fn        (mhc_mult*(mhc_mult+2), mhc_mult*hidden_size)
      self.alpha          ↔ scale     (3,)
      self.bias           ↔ base      (mhc_mult*(mhc_mult+2),)
      self.input_rms.weight ↔ norm_weight (mhc_mult*hidden_size,)
    """

    # H^pre 候选激活函数集合（class-level constant，校验/argparse choices 共用）
    H_PRE_ACTIVATIONS = ('sigmoid', 'softmax', 'identity', 'relu', 'tanh')

    def __init__(self, n: int, dim: int, alpha_init: float = 0.01,
                 sinkhorn_iters: int = 10, post_mult_value: float = 2.0,
                 eps: float = 1e-6, pre_eps: float = 1e-6, sinkhorn_eps: float = 1e-6,
                 disable_h_res: bool = False, per_stream_norm: bool = False,
                 h_pre_activation: str = 'sigmoid'):
        super().__init__()
        assert n >= 1 and isinstance(n, int), f"n 必须是 >=1 的整数，得到 {n}"
        assert h_pre_activation in self.H_PRE_ACTIVATIONS, \
            f"h_pre_activation 必须是 {self.H_PRE_ACTIVATIONS} 之一，得到 {h_pre_activation!r}"
        self.n = n
        self.dim = dim
        self.nC = n * dim
        self.sinkhorn_iters = sinkhorn_iters
        self.post_mult_value = post_mult_value
        self.pre_eps = pre_eps
        self.sinkhorn_eps = sinkhorn_eps
        # —— Ablation: H^pre 激活函数（默认 'sigmoid'，论文 Eq.8 & 官方 TileKernels）——
        # H^pre 是 sublayer 输入 mix 系数 (B, L, n): sublayer_in = Σ_i H^pre[i] * x[i]
        #   'sigmoid' : σ(raw)+ε ∈ (ε, 1+ε)  ← 默认，每路独立非负有界
        #   'softmax' : softmax(·,-1) Σ=1     ← 强制 normalized mixture
        #   'identity': raw + bias            ← 无约束，允许负 mix
        #   'relu'    : relu(raw)+ε ∈ [ε, ∞)  ← 稀疏激活
        #   'tanh'    : tanh(raw) ∈ (-1, 1)   ← 有界但允许负
        # 非 sigmoid 候选下 H^pre 可能 < 0，sublayer F 接收的是 raw linear combination。
        self.h_pre_activation = h_pre_activation
        # —— Ablation: 禁用 H^res（论文 Table 1 对照）——
        # True 时 forward 跳过 Sinkhorn-Knopp，令 H^res = I_n（切断 n 路信息混合）。
        # 论文 Table 1: H^res 单独贡献 ~81% 总收益，因此小模型上仍能观察明显劣化。
        # phi 的 fused_dim 不裁剪，ckpt 形状向后兼容；H^res_raw 段算出但不用。
        self.disable_h_res = disable_h_res

        # —— Ablation: per-stream RMSNorm（n 路独立 vs 论文/官方整体）——
        # 默认 False (nn.RMSNorm(n·C))：n 路共享 RMS scale，inter-stream 相对量级被保留。
        # 这是 mHC 信息保留的核心：φ 看到的 n 路保留原始量级 → H^pre/H^post 能学到"哪些
        # 路更重要"；SK doubly-stochastic 质量守恒在共享 mass scale 时 well-defined。
        # True (PerStreamRMSNorm)：每路独立归一化 + 每路独立 weight (n, C)，参数量与默
        # 认完全一致 → strictly controlled ablation，专用于验证整体 norm 必要性。
        self.per_stream_norm = per_stream_norm

        # RMSNorm：默认整体 norm（论文 Eq.7），ablation 下退化为 per-stream
        if per_stream_norm:
            self.input_rms = PerStreamRMSNorm(n, self.dim, eps=eps)        # (B,L,n,C) 最后一维
        else:
            self.input_rms = nn.RMSNorm(self.nC, eps=eps)                  # (B,L,n·C) 整体

        # ---------- Dynamic 部分：fused φ 投影 ----------
        # 单一 Linear 输出 (2n + n²)，按段拆分为 (pre_mix | post_mix | res_mix)：
        #   [0  : n     ] → pre  (H^pre  raw)
        #   [n  : 2n    ] → post (H^post raw)
        #   [2n : 2n+n²] → res  (H^res  raw，后续 view 成 (n, n))
        self.fused_dim = 2 * n + n * n
        self.phi = nn.Linear(self.nC, self.fused_dim, bias=False)

        # ---------- Static 部分：共享 bias / alpha（对齐官方 base / scale）----------
        # bias (2n+n²,) 分段 init：
        #   - pre/post 段  : zeros → init 时 σ(0)·1 = 0.5·1 / 2σ(0)·1 = 1·1（n 路均匀）
        #   - res 段       : eye(n).flatten() → H^res init = SK(exp(I)) ≈ 对角占优
        # ⚠️ res 段必须 eye init：SK 在 H_res ≈ (1/n)·11.T 处的雅可比把 rank-1 梯度投到
        #    zero-row/col-sum 切空间后压制 6~8 个数量级（实测 grad ≈ 1e-14 无法训练）；
        #    eye init 让 H_res 落在对角占优区，梯度 ≈ 1e-7~1e-10 健康（论文 §4.1 隐含要求）。
        bias_init = torch.zeros(self.fused_dim)
        bias_init[2 * n:] = torch.eye(n).flatten()
        self.bias = nn.Parameter(bias_init)

        # alpha (3,)，前向广播到 n / n / n² 三段（对齐官方 scale）
        self.alpha = nn.Parameter(torch.full((3,), alpha_init))

    def _expanded_alpha(self) -> torch.Tensor:
        """把共享 alpha (3,) 展开成 (2n+n²,) 的 per-element scale 向量。"""
        n = self.n
        # cat 操作非常便宜（小张量，常量化容易被 torch.compile 折叠）
        return torch.cat([
            self.alpha[0].expand(n),
            self.alpha[1].expand(n),
            self.alpha[2].expand(n * n),
        ])

    def forward(self, x_flat: torch.Tensor, sublayer_fn) -> torch.Tensor:
        """单 sublayer 的 mHC 残差更新。

        Args:
            x_flat: (B, L, n·dim) flat 残差流（与 MiniMindBlock 接口对齐）
            sublayer_fn: callable, (B, L, dim) → (B, L, dim)
                        对应论文里的 F（attn 或 mlp），不含残差连接

        Returns:
            (B, L, n·dim) 更新后的 flat 残差流
        """
        B, L, _ = x_flat.shape
        n, C = self.n, self.dim

        # 1) 把 flat 视图 reshape 成 (B, L, n, C) 多流视图（零成本 view）
        x = x_flat.view(B, L, n, C)

        # 2) RMSNorm：默认整体 norm（论文 Eq.7，保留 inter-stream 相对量级）；
        #    per_stream_norm=True 时退化为每路独立归一化（ablation 对照组）。
        if self.per_stream_norm:
            # 作用于 (B, L, n, C) 的最后一维，n 路独立归一化后 flatten 回 (B, L, n·C)
            x_norm = self.input_rms(x).reshape(B, L, n * C)
        else:
            x_norm = self.input_rms(x_flat)                             # (B, L, n·C)

        # 3) Fused φ：1 次 matmul 同时算出 pre/post/res 的 raw mix
        mixes_raw = self.phi(x_norm)                                    # (B, L, 2n + n²)

        # 4) 仿射：α ⊙ raw + b（α 广播 + flat bias 加法，对齐官方 base/scale）
        mixes = mixes_raw * self._expanded_alpha() + self.bias          # (B, L, 2n + n²)

        # 5) 分段取出三段 raw 系数
        H_pre_raw  = mixes[..., :n]                                     # (B, L, n)
        H_post_raw = mixes[..., n:2 * n]                                # (B, L, n)
        H_res_raw  = mixes[..., 2 * n:].view(B, L, n, n)                # (B, L, n, n)

        # 6) 流形投影
        # H^pre 的激活由 self.h_pre_activation 控制（默认 sigmoid，论文/官方标准）；
        # 各分支语义见 __init__ 中的注释。pre_eps 仅在 sigmoid/relu 下加（防止某路 → 0
        # 切断梯度），其它路径不加（softmax 严格正、identity/tanh 加 ε 物理无意义）。
        act = self.h_pre_activation
        if act == 'sigmoid':
            H_pre = torch.sigmoid(H_pre_raw) + self.pre_eps             # ∈ (ε, 1+ε)
        elif act == 'softmax':
            H_pre = torch.softmax(H_pre_raw, dim=-1)                    # ∈ (0,1), Σ=1
        elif act == 'identity':
            H_pre = H_pre_raw                                           # 无约束 raw
        elif act == 'relu':
            H_pre = torch.relu(H_pre_raw) + self.pre_eps                # ∈ [ε, ∞)
        else:  # 'tanh'  （校验已在 __init__ 完成，else 安全兜底）
            H_pre = torch.tanh(H_pre_raw)                               # ∈ (-1, 1)
        H_post = torch.sigmoid(H_post_raw) * self.post_mult_value       # ∈ (0, post_mult)
        if self.disable_h_res:
            # Ablation：H^res 冻结为 identity matrix（论文 Table 1 中 H^res 禁用时的替代）。
            # 跳过 Sinkhorn-Knopp 投影；H_res_raw 算出来但不使用，梯度链路对应段自然为 0。
            # 用 eye(n) 广播到 (B, L, n, n)，einsum 仍可正常工作（expand 零拷贝）。
            H_res = torch.eye(n, dtype=x.dtype, device=x.device).expand(B, L, n, n)
        else:
            H_res = sinkhorn_normalize(H_res_raw.float(),               # doubly stochastic
                                       iters=self.sinkhorn_iters,
                                       eps=self.sinkhorn_eps).to(x.dtype)

        # 7) 读入：H^pre · x  → (B, L, C)（n 路加权聚合送给 sublayer）
        #    x: (B,L,n,C), H_pre: (B,L,n) → unsqueeze(-1) broadcast 到 (B,L,n,1)
        sublayer_in = (H_pre.unsqueeze(-1) * x).sum(dim=-2)             # (B, L, C)

        # 8) 跑 sublayer F（attn 或 mlp）
        F_out = sublayer_fn(sublayer_in)                                # (B, L, C)

        # 9) 写回 + 残差混合（对齐官方 mhc_post_ref）
        #    write_back = H^post.T · F  → (B, L, n, C)
        #    mixed      = H^res · x     → (B, L, n, C)  via einsum 'blij,bljc->blic'
        #    H^res=I 时 mixed[..., i, :] = x[..., i, :]，即每路独立保留自己的残差。
        write_back = H_post.unsqueeze(-1) * F_out.unsqueeze(-2)         # (B, L, n, C)
        mixed = torch.einsum('blij,bljc->blic', H_res, x)               # (B, L, n, C)

        # 10) 输出 = mixed + write_back，再 flat 回去
        out = mixed + write_back                                        # (B, L, n, C)
        return out.reshape(B, L, n * C)


# ──────────────────────────────────────────────────────────────────────────────
# StreamExpand：对称复制 entry（对齐官方 expand_to_mhc_ref）
# ──────────────────────────────────────────────────────────────────────────────
class StreamExpand(nn.Module):
    """(B, L, D) → (B, L, n·D)：把 embedding 对称复制 n 份。

    无可学参数。对齐 DeepSeek 官方 expand_to_mhc_ref：
        x_0 = (embed, embed, ..., embed) ∈ R^{n×C}

    n 路对称下 H^res · x = x (因 H^res 行和=1)，看起来层 0 没动作；但同层的
    H^post 经 phi 的 Kaiming init 在各路上不同，write_back[i] = H_post[i]·F_out
    立即破坏 n 路对称，后续层 H^res 即可正常驱动梯度。

    相对前版 StreamLift（第 0 路独占，其他路 zero）的优势：
      - 与官方/论文对齐
      - 不需要 b_res 初始化为 I 的特殊处理
      - layer 0 的 sublayer_in 不被 H_pre 缩成"x[0] 的 scalar 缩放"那么单调
        （H_pre 对每路同等加权但 sum 后是单流，依然是 x[0] 的某个 scale，
         但 RMS norm 后的 phi 投影更稳定）
    """
    def __init__(self, n: int, dim: int):
        super().__init__()
        self.n = n
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.n == 1:
            return x
        # (B, L, D) → (B, L, 1, D) → (B, L, n, D) → (B, L, n*D)
        # expand 是 zero-copy view；reshape 触发一次 contiguous copy
        B, L, D = x.shape
        return x.unsqueeze(-2).expand(B, L, self.n, D).reshape(B, L, self.n * D)


# ──────────────────────────────────────────────────────────────────────────────
# MHCHead：可学加权 reduce（对齐官方 mhc_head，替代无参 StreamReduce）
# ──────────────────────────────────────────────────────────────────────────────
class MHCHead(nn.Module):
    """(B, L, n·D) → (B, L, D)：mHC 风格的可学习 weighted reduce（lm_head 之前）。

    对齐 DeepSeek 官方 mhc_head：用一个 fused Linear 计算 pre_mix（n 维），
    sigmoid + ε 后加权 reduce n 路得到 final D-dim representation。

    实现公式：
        x̃        = RMSNorm(vec(x))
        m_raw    = α · (x̃ · φ) + b              # ∈ R^{1×n}
        mix      = σ(m_raw) + ε                  # 非负，每路 ∈ (ε, 1+ε)
        out      = (mix.unsqueeze(-1) * x).sum(-2)

    相对前版 StreamReduce（无参 row-wise sum）的优势：
      - 解决"最后一层 H_res frozen"问题：之前因 H_res 列和=1 + StreamReduce 是
        row-sum，最后一层 H_res 梯度恒 0（数学性质，已确认 1e-15 噪声）。改用
        可学 sigmoid 加权后，∂loss/∂x[i] 在 n 路上各异，最后一层 H_res 梯度路径打开。
      - 表达力提升：模型可以学到"用哪几路最相关于 lm_head"，而不是简单平均。
      - 与官方 mhc_head 对齐。

    参数量：phi.weight (nC × n) + bias (n,) + alpha (1,) + RMSNorm.weight (nC,)
    n=4, C=1024 → ~20K params，相对 lm_head 的 vocab × hidden 完全是噪音级。
    """
    def __init__(self, n: int, dim: int, alpha_init: float = 0.01,
                 eps: float = 1e-6, pre_eps: float = 1e-6):
        super().__init__()
        self.n = n
        self.dim = dim
        self.nC = n * dim
        self.pre_eps = pre_eps

        # 与 MHCConnection 同款 RMSNorm（作用于 flatten 后的 nC 维）
        self.input_rms = nn.RMSNorm(self.nC, eps=eps)

        # φ : nC × n（输出维度 n，对应每路的 raw mix）
        self.phi = nn.Linear(self.nC, n, bias=False)

        # 共享 bias / alpha（与 MHCConnection 同款命名风格）
        self.bias = nn.Parameter(torch.zeros(n))
        self.alpha = nn.Parameter(torch.full((1,), alpha_init))

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        B, L, _ = x_flat.shape
        n, C = self.n, self.dim
        if n == 1:
            # 单路退化：直接 RMSNorm 后返回（不需要 mix）
            return self.input_rms(x_flat)
        x = x_flat.view(B, L, n, C)                                     # (B, L, n, C)
        x_norm = self.input_rms(x_flat)                                 # (B, L, n·C)
        raw = self.phi(x_norm) * self.alpha + self.bias                 # (B, L, n)
        mix = torch.sigmoid(raw) + self.pre_eps                         # (B, L, n)
        return (mix.unsqueeze(-1) * x).sum(dim=-2)                      # (B, L, C)


# =============================================================================
# Sinkhorn fused kernel：correctness & benchmark
# =============================================================================

def _sk_check(name, ref, fused, atol, rtol):
    diff = (ref - fused).abs().max().item()
    rel = diff / max(ref.abs().max().item(), 1e-8)
    flag = "✓" if (diff < atol or rel < rtol) else "✗"
    print(f"  {flag} {name:>18s}: max|diff|={diff:.3e}  rel={rel:.3e}")
    assert diff < atol or rel < rtol, f"{name} 不对齐: abs={diff}, rel={rel}"


def _test_sinkhorn(BL, n, iters, dtype, atol, rtol, eps=1e-6):
    """SK fused vs ref：fwd y / dx 数值对齐。

    fp32 输入下 fused 内部和 ref 都是 fp32，应严格对齐到 ~1e-5。
    bf16 输入下 ref 也是 bf16 计算，fused 内部 fp32 反而更精确，
    误差容忍按 bf16 dynamic range 给 abs+rel 双判据。
    """
    torch.manual_seed(0)
    device = 'cuda'
    M_base = torch.randn(BL, n, n, device=device, dtype=dtype)

    # ref
    M_ref = M_base.detach().clone().requires_grad_(True)
    with torch.amp.autocast("cuda", enabled=False):
        y_ref = _sinkhorn_ref(M_ref, iters, eps)

    # fused（通过 sinkhorn_normalize 公开 API，确保后端选择走 fused 分支）
    M_fused = M_base.detach().clone().requires_grad_(True)
    y_fused = sinkhorn_normalize(M_fused, iters=iters, eps=eps)

    print(f"[Sinkhorn BL={BL} n={n} iters={iters} dtype={str(dtype).replace('torch.', '')}]")
    _sk_check("fwd y", y_ref, y_fused, atol, rtol)

    # 检查 doubly-stochastic 性质（容忍 eps 量级偏差）
    rs = y_fused.sum(dim=-1)
    cs = y_fused.sum(dim=-2)
    rs_dev = (rs - 1.0).abs().max().item()
    cs_dev = (cs - 1.0).abs().max().item()
    print(f"  • row_sum max|·-1|={rs_dev:.3e}   col_sum max|·-1|={cs_dev:.3e}")

    # backward
    g = torch.randn_like(y_ref)
    y_ref.backward(g)
    y_fused.backward(g)
    _sk_check("dM", M_ref.grad, M_fused.grad, atol, rtol)
    print()


def _bench_sinkhorn(BL, n, iters, dtype, warmup=20, runs=100):
    import time
    device = 'cuda'

    print(f"  --- BL={BL} n={n} iters={iters} dtype={str(dtype).replace('torch.', '')} ---")
    for tag, fn_factory in [("ref  ", lambda: _sinkhorn_ref),
                            ("fused", lambda: _SinkhornFn.apply)]:
        fn = fn_factory()
        # 预热
        for _ in range(warmup):
            M = torch.randn(BL, n, n, device=device, dtype=dtype, requires_grad=True)
            y = fn(M, iters, 1e-6)
        torch.cuda.synchronize()

        # fwd 计时
        M = torch.randn(BL, n, n, device=device, dtype=dtype)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(runs):
            y = fn(M, iters, 1e-6)
        torch.cuda.synchronize()
        fwd_ms = (time.perf_counter() - t0) / runs * 1000

        # fwd+bwd 计时
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(runs):
            M = torch.randn(BL, n, n, device=device, dtype=dtype, requires_grad=True)
            y = fn(M, iters, 1e-6)
            y.sum().backward()
        torch.cuda.synchronize()
        fwdbwd_ms = (time.perf_counter() - t0) / runs * 1000
        print(f"    {tag}  fwd: {fwd_ms:7.3f} ms   fwd+bwd: {fwdbwd_ms:7.3f} ms")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("此模块需要 CUDA")
    if not TRITON_AVAILABLE:
        raise RuntimeError("请先 pip install triton")

    # 关闭 TF32，确保 fp32 严格对齐
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    print("=" * 70)
    print("Sinkhorn 数值正确性测试 (fp32, TF32 disabled)")
    print("=" * 70)
    # mHC 典型配置：n ∈ {4, 8}, iters=10
    # BL = B * L，覆盖小/中/大 batch
    for BL, n in [(256, 4), (4096, 4), (4096, 8), (1024, 16), (32, 2)]:
        _test_sinkhorn(BL=BL, n=n, iters=10, dtype=torch.float32,
                       atol=1e-5, rtol=1e-4)

    print("=" * 70)
    print("Sinkhorn 数值正确性测试 (bf16, abs OR rel)")
    print("=" * 70)
    for BL, n in [(4096, 4), (4096, 8)]:
        # bf16：fused 内部 fp32 通常比 ref 更准，给宽容差（dM 容差再放宽）
        _test_sinkhorn(BL=BL, n=n, iters=10, dtype=torch.bfloat16,
                       atol=2e-2, rtol=2e-2)

    # benchmark 恢复 TF32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("\n" + "=" * 70)
    print("Sinkhorn 性能 benchmark (fwd / fwd+bwd)")
    print("=" * 70)
    # 典型 mHC 训练 shape：B=2, L=1024~2048, n=4, 每个 sublayer 调一次
    for BL, n, dtype in [
        (2048, 4, torch.bfloat16),    # B=2, L=1024
        (4096, 4, torch.bfloat16),    # B=2, L=2048
        (8192, 4, torch.bfloat16),    # B=4, L=2048
        (4096, 8, torch.bfloat16),
        (4096, 4, torch.float32),
    ]:
        _bench_sinkhorn(BL=BL, n=n, iters=10, dtype=dtype)
