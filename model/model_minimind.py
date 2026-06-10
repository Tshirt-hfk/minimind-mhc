import math
import torch, torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

# Manifold-Constrained Hyper-Connections (mHC, DeepSeek 2025, arXiv:2512.24880)。
# use_mhc=1 → MHCConnection_Fused（完整 mHC，含 H^res Sinkhorn-Knopp 投影）
# use_mhc=2 → MHCConnection_FusedNoHres（H^res = I_n，跳过 SK）
from .mhc_common import StreamExpand, MHCHead
from .mhc_fused import MHCConnection_Fused
from .mhc_fused_no_hres import MHCConnection_FusedNoHres

# Optional Dao-AILab flash-attn 库（CUDA-only，比 PyTorch SDPA 更快、显存更省）。
# 未安装时自动回退到 SDPA / 慢路径，不影响功能。
# H800/H100 强烈推荐 flash-attn ≥ 3.0（专为 Hopper 优化 WGMMA + TMA，比 v2 快约 1.5-2x）；
# rank 0 进程会打印 1 次版本日志，便于训练脚本启动时确认是否启用 Hopper 优化路径。
try:
    from flash_attn import flash_attn_func as _flash_attn_func   # type: ignore
    _FLASH_ATTN_LIB_AVAILABLE = True
    try:
        import flash_attn as _flash_attn_module                  # type: ignore
        _FLASH_ATTN_VERSION = getattr(_flash_attn_module, "__version__", "unknown")
    except Exception:
        _FLASH_ATTN_VERSION = "unknown"
except Exception:
    _flash_attn_func = None
    _FLASH_ATTN_LIB_AVAILABLE = False
    _FLASH_ATTN_VERSION = None


def _log_flash_attn_status_once():
    """在 rank 0（或非分布式）首次构建模型时打印一次 flash-attn 状态。
    H800 上未装 v3+ 会有显著性能损失，便于训练时立刻发现配置问题。"""
    if getattr(_log_flash_attn_status_once, "_done", False):
        return
    _log_flash_attn_status_once._done = True
    # 仅 rank 0 打印（无分布式环境时也打）
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return
    except Exception:
        pass
    if not _FLASH_ATTN_LIB_AVAILABLE:
        print("[mHC/attn] flash-attn lib 未安装 → 走 PyTorch SDPA。"
              " H800/H100 建议 `pip install flash-attn>=3.0` 启用 Hopper 优化路径。")
        return
    v = str(_FLASH_ATTN_VERSION)
    try:
        major = int(v.split(".")[0])
    except Exception:
        major = -1
    if major >= 3:
        print(f"[mHC/attn] flash-attn v{v} 已启用（Hopper 优化：WGMMA + TMA）")
    elif major >= 0:
        # H800 上 v2 也能跑，但比 v3 慢 1.5-2x；提示用户升级
        try:
            import torch
            cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
            cc = cap[0] * 10 + cap[1]
        except Exception:
            cc = 0
        if cc >= 90:
            print(f"[mHC/attn] flash-attn v{v} 已启用，但当前 GPU (sm_{cc}) 为 Hopper；"
                  " 建议升级到 flash-attn>=3.0 获得 WGMMA+TMA 加速（约 1.5-2x）。")
        else:
            print(f"[mHC/attn] flash-attn v{v} 已启用。")
    else:
        print(f"[mHC/attn] flash-attn 已加载（版本未知）。")

# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     MiniMind Config
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"
    def __init__(self, hidden_size=1024, num_hidden_layers=24, use_moe=False,
                 use_mhc=0, mhc_residual_expansion=4,
                 **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe
        self.dropout = kwargs.get("dropout", 0.0)
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        self.flash_attn = kwargs.get("flash_attn", True)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.hidden_act = kwargs.get("hidden_act", 'silu')
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 1e6)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        ### MoE specific configs (ignored if use_moe = False)
        self.num_experts = kwargs.get("num_experts", 4)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)

        # use_mhc: 0=plain pre-Norm; 1=完整 mHC (含 H^res + SK); 2=mHC 无 H^res (H^res=I_n)
        # mhc_residual_expansion (n): 残差流路数；总维度 = hidden_size * n
        #   fused kernel 约束：n ∈ {1,2,4,8,16}，n·hidden_size 是 2 的幂
        # 其他 mHC 超参（alpha_init / sinkhorn_iters / post_mult_value）固化在底层 ctor 默认值
        self.use_mhc                = use_mhc
        self.mhc_residual_expansion = mhc_residual_expansion
        assert use_mhc in (0, 1, 2), \
            f"use_mhc 必须 ∈ {{0,1,2}}（0=plain, 1=mHC完整版, 2=mHC无H^res版），得到 {use_mhc}"
        if self.use_mhc:
            assert isinstance(mhc_residual_expansion, int) and mhc_residual_expansion >= 1, \
                f"mhc_residual_expansion 必须是 >=1 的整数，得到 {mhc_residual_expansion}"

# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     MiniMind Model
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
# - 标准 RMSNorm：直接复用 PyTorch 2.4+ 自带的 nn.RMSNorm

def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: dict = None):
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0
    if rope_scaling is not None: # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)
        )
        if end / orig_max > 1.0:
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    def rotate_half(x): return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1: return x
    return (x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim))

class Attention(nn.Module):
    """三档自动 dispatch：flash-attn lib > PyTorch SDPA > 手写慢路径。
    - flash-attn lib：CUDA + bf16/fp16 + 无任意 attn_mask（GQA 原生支持，省 repeat_kv 与 transpose）
    - SDPA：覆盖训练、KV cache decode（q_len=1）、chunked prefill（1<q_len<kv_len，显式右对齐 causal mask）、padding mask（避免 D2H 同步）
    - 慢路径：CPU 或最终兜底
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.head_dim
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        # flash_attn=True → 启用快路径自动 dispatch（flash-attn lib > SDPA）；False → 仅慢路径
        self.use_fast_attn      = bool(config.flash_attn)
        self.use_flash_attn_lib = _FLASH_ATTN_LIB_AVAILABLE and self.use_fast_attn
        self.use_sdpa           = hasattr(F, 'scaled_dot_product_attention') and self.use_fast_attn

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xq, xk = self.q_norm(xq), self.k_norm(xk)
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None
        q_len, kv_len = xq.shape[1], xk.shape[1]
        drop_p = self.dropout if self.training else 0.0

        # ── Backend 1: Dao-AILab flash-attn lib（最优）──────────────────────────
        # 条件：CUDA + bf16/fp16 + 无 padding mask；(B,S,H,D) layout 直传；GQA 原生支持。
        # causal=True 时 flash_attn 自动应用右对齐 mask，q_len < kv_len（KV cache decode/prefill）也正确。
        if (self.use_flash_attn_lib
                and xq.is_cuda
                and xq.dtype in (torch.float16, torch.bfloat16)
                and attention_mask is None):
            out_4d = _flash_attn_func(xq, xk, xv, dropout_p=drop_p, causal=self.is_causal)
            output = out_4d.reshape(bsz, q_len, -1)
        # ── Backend 2: PyTorch SDPA（覆盖训练 / decode / prefill / padding）─────
        elif self.use_sdpa:
            xq_t = xq.transpose(1, 2)
            xk_t = repeat_kv(xk, self.n_rep).transpose(1, 2)
            xv_t = repeat_kv(xv, self.n_rep).transpose(1, 2)
            if attention_mask is None:
                # 无 padding：按 q_len/kv_len 选 is_causal 或建显式右对齐 causal mask
                # PyTorch SDPA 的 is_causal=True 在 q_len != kv_len 时是 upper-left（错误语义），
                # 故对 KV cache 场景显式构造 mask 而非依赖 is_causal flag。
                if not self.is_causal or q_len == 1:
                    output = F.scaled_dot_product_attention(xq_t, xk_t, xv_t, dropout_p=drop_p, is_causal=False)
                elif q_len == kv_len:
                    output = F.scaled_dot_product_attention(xq_t, xk_t, xv_t, dropout_p=drop_p, is_causal=True)
                else:  # 1 < q_len < kv_len: chunked prefill，右对齐 causal
                    offset = kv_len - q_len
                    m = torch.full((q_len, kv_len), float("-inf"), device=xq_t.device, dtype=xq_t.dtype).triu(offset + 1)
                    output = F.scaled_dot_product_attention(xq_t, xk_t, xv_t, attn_mask=m, dropout_p=drop_p, is_causal=False)
            else:
                # 有 padding mask：合并 padding mask + 右对齐 causal mask（避免 torch.all D2H 同步）
                neg_inf = torch.finfo(xq_t.dtype).min
                attn_mask = (1.0 - attention_mask.to(xq_t.dtype)).unsqueeze(1).unsqueeze(2) * neg_inf  # (B,1,1,kv_len)
                if self.is_causal and q_len > 1:
                    offset = kv_len - q_len
                    causal_m = torch.full((q_len, kv_len), float("-inf"), device=xq_t.device, dtype=xq_t.dtype).triu(offset + 1)
                    attn_mask = attn_mask + causal_m  # 广播为 (B,1,q_len,kv_len)
                output = F.scaled_dot_product_attention(xq_t, xk_t, xv_t, attn_mask=attn_mask, dropout_p=drop_p, is_causal=False)
            output = output.transpose(1, 2).reshape(bsz, q_len, -1)
        # ── Backend 3: 手写慢路径（CPU / 无 SDPA / 兜底）──────────────────────
        else:
            xq_t = xq.transpose(1, 2)
            xk_t = repeat_kv(xk, self.n_rep).transpose(1, 2)
            xv_t = repeat_kv(xv, self.n_rep).transpose(1, 2)
            scores = (xq_t @ xk_t.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if self.is_causal and q_len > 1:  # q_len=1（decode）无需 causal mask
                offset = kv_len - q_len
                scores = scores + torch.full((q_len, kv_len), float("-inf"), device=scores.device).triu(offset + 1)
            if attention_mask is not None:
                scores = scores + (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq_t)) @ xv_t
            output = output.transpose(1, 2).reshape(bsz, q_len, -1)

        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv

class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig, intermediate_size: int = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class MOEFeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList([FeedForward(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.num_experts)])
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)
        scores = F.softmax(self.gate(x_flat), dim=-1)
        topk_weight, topk_idx = torch.topk(scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False)
        if self.config.norm_topk_prob: topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        y = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (topk_idx == i)
            if mask.any():
                token_idx = mask.any(dim=-1).nonzero().flatten()
                weight = topk_weight[mask].view(-1, 1)
                y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
            elif self.training:
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())
        if self.training and self.config.router_aux_loss_coef > 0:
            load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)
            self.aux_loss = (load * scores.mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()
        return y.view(batch_size, seq_len, hidden_dim)

class MiniMindBlock(nn.Module):
    """三档残差结构（plain / mHC 完整版 / mHC 无 H^res）。

    mHC 路径下 sublayer F 固定 pre-Norm：F(z) = sublayer(in_norm(z))，由调用者在 F
    闭包里加（fused kernel 把 F 视为黑盒）。
    """
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)
        self.use_mhc = config.use_mhc
        eps = config.rms_norm_eps
        D = config.hidden_size
        self.attn_in_norm = nn.RMSNorm(D, eps=eps)
        self.ffn_in_norm  = nn.RMSNorm(D, eps=eps)
        if config.use_mhc:
            n = config.mhc_residual_expansion
            MHCConnCls = MHCConnection_Fused if config.use_mhc == 1 else MHCConnection_FusedNoHres
            self.attn_mhc = MHCConnCls(n, D, eps=eps)
            self.mlp_mhc  = MHCConnCls(n, D, eps=eps)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        # hidden_states: (b, l, D) [plain] 或 (b, l, n·D) [mhc]
        if self.use_mhc:
            # 把 sublayer F 包成闭包传给 MHCConnection（论文 Eq.3：x' = H^res·x + H^post·F(H^pre·x)）
            # attn 的 past_kv 是状态副作用，用闭包变量捕获。
            present_kv_holder = [None]

            def attn_fn(z):
                out, present_kv = self.self_attn(
                    self.attn_in_norm(z), position_embeddings, past_key_value, use_cache, attention_mask
                )
                present_kv_holder[0] = present_kv
                return out

            def mlp_fn(z):
                return self.mlp(self.ffn_in_norm(z))

            hidden_states = self.attn_mhc(hidden_states, attn_fn)
            hidden_states = self.mlp_mhc(hidden_states, mlp_fn)
            return hidden_states, present_kv_holder[0]

        # plain pre-Norm
        residual = hidden_states
        attn_out, present_key_value = self.self_attn(
            self.attn_in_norm(hidden_states), position_embeddings, past_key_value, use_cache, attention_mask
        )
        hidden_states = residual + attn_out
        residual = hidden_states
        hidden_states = residual + self.mlp(self.ffn_in_norm(hidden_states))
        return hidden_states, present_key_value


class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        eps = config.rms_norm_eps
        D = config.hidden_size
        if config.use_mhc:
            n = config.mhc_residual_expansion
            # entry: RMSNorm + StreamExpand (D → n·D)；exit: MHCHead (n·D → D) + RMSNorm
            self.embed_up_norm = nn.Sequential(nn.RMSNorm(D, eps=eps), StreamExpand(n, D))
            self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
            self.norm = nn.Sequential(
                MHCHead(n, D, eps=eps),
                nn.RMSNorm(D, eps=eps),
            )
        else:
            self.embed_up_norm = nn.Identity()
            self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
            self.norm = nn.RMSNorm(D, eps=eps)
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.head_dim, end=config.max_position_embeddings, rope_base=config.rope_theta, rope_scaling=config.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        hidden_states = self.dropout(self.embed_tokens(input_ids))  # (b, l, D)
        hidden_states = self.embed_up_norm(hidden_states)           # (b, l, n·D) [mhc] 或恒等 [plain]
        # Recompute RoPE buffers lost during meta-device init (transformers>=5.x)
        if self.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(dim=self.config.head_dim, end=self.config.max_position_embeddings, rope_base=self.config.rope_theta, rope_scaling=self.config.rope_scaling)
            self.freqs_cos, self.freqs_sin = freqs_cos.to(hidden_states.device), freqs_sin.to(hidden_states.device)
        position_embeddings = (self.freqs_cos[start_pos:start_pos + seq_length], self.freqs_sin[start_pos:start_pos + seq_length])
        presents = []
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)
        hidden_states = self.norm(hidden_states)  # (b, l, D)
        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        return hidden_states, presents, aux_loss

class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        if self.config.tie_word_embeddings: self.model.embed_tokens.weight = self.lm_head.weight
        self.post_init()
        # rank0 进程首次构建模型时打印 flash-attn 版本，提示 H800 是否启用 v3 优化路径
        _log_flash_attn_status_once()

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, labels=None, **kwargs):
        hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache, **kwargs)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
        return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)
    
    # https://github.com/jingyaogong/minimind/discussions/611
    @torch.inference_mode()
    def generate(self, inputs=None, attention_mask=None, max_new_tokens=8192, temperature=0.85, top_p=0.85, top_k=50, eos_token_id=2, streamer=None, use_cache=True, num_return_sequences=1, do_sample=True, repetition_penalty=1.0, **kwargs):
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
        attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
        past_key_values = kwargs.pop("past_key_values", None)
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        if streamer: streamer.put(input_ids.cpu())
        for _ in range(max_new_tokens):
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0
            outputs = self.forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache, **kwargs)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1) if attention_mask is not None else None
            logits = outputs.logits[:, -1, :] / temperature
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]):
                    seen = torch.unique(input_ids[i]); score = logits[i, seen]; logits[i, seen] = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)
            if top_k > 0: 
                logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
                logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(logits, dim=-1, keepdim=True)
            if eos_token_id is not None: next_token = torch.where(finished.unsqueeze(-1), next_token.new_full((next_token.shape[0], 1), eos_token_id), next_token)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values if use_cache else None
            if streamer: streamer.put(next_token.cpu())
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all(): break
        if streamer: streamer.end()
        if kwargs.get("return_kv"): return {'generated_ids': input_ids, 'past_kv': past_key_values}
        return input_ids