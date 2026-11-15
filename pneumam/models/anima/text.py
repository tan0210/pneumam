import json
import os
from collections.abc import Sequence
from typing import cast

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn

from ...graphs import Graphed

QWEN_PAD = 151643  # <|endoftext|>
BUCKET = (
    32  # token counts are padded up to a multiple of this, so few shapes get captured
)


def pad_context(ctx: torch.Tensor, length: int) -> torch.Tensor:
    """Zero-pad text embeddings along the sequence.

    Args:
        ctx: Text embeddings, shape ``[B, S, text_dim]``.
        length: Target sequence length, at least ``S``.

    Returns:
        Tensor of shape ``[B, length, text_dim]``.
    """
    return F.pad(ctx, (0, 0, 0, length - ctx.shape[1]))


def _rope(
    length: int, dim: int, theta: float, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / theta ** (
        torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim
    )
    freqs = torch.arange(length, device=device, dtype=torch.float32)[:, None] * inv
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x is [B, H, S, D] and cos, sin are [S, D], with the second half of each head rotated
    half = x.shape[-1] // 2
    turned = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cos.to(x.dtype) + turned * sin.to(x.dtype)


def _rms(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return F.rms_norm(x.float(), (x.shape[-1],), eps=eps).to(x.dtype) * weight


def _w(shape: tuple[int, ...]) -> nn.Parameter:
    return nn.Parameter(torch.empty(shape), requires_grad=False)


class _QwenLayer(nn.Module):
    def __init__(
        self, dim: int, heads: int, kv_heads: int, head_dim: int, mlp: int
    ) -> None:
        super().__init__()
        self.heads, self.kv_heads, self.head_dim, self.mlp = (
            heads,
            kv_heads,
            head_dim,
            mlp,
        )
        self.ln1, self.ln2 = _w((dim,)), _w((dim,))
        self.qkv = _w(((heads + 2 * kv_heads) * head_dim, dim))
        self.q_norm, self.k_norm = _w((head_dim,)), _w((head_dim,))
        self.o = _w((dim, heads * head_dim))
        self.gate_up = _w((2 * mlp, dim))
        self.down = _w((dim, mlp))

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        B, S, _ = x.shape
        H, KV, D = self.heads, self.kv_heads, self.head_dim
        qkv = F.linear(_rms(x, self.ln1), self.qkv)
        q, k, v = qkv.split((H * D, KV * D, KV * D), dim=-1)
        q = _rms(q.view(B, S, H, D), self.q_norm).transpose(1, 2)
        k = _rms(k.view(B, S, KV, D), self.k_norm).transpose(1, 2)
        v = v.view(B, S, KV, D).transpose(1, 2)
        q, k = _rotate(q, cos, sin), _rotate(k, cos, sin)
        att = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        x = x + F.linear(att.transpose(1, 2).reshape(B, S, H * D), self.o)
        gate, up = F.linear(_rms(x, self.ln2), self.gate_up).split(self.mlp, dim=-1)
        return x + F.linear(F.silu(gate) * up, self.down)


class Qwen3(nn.Module):
    """The Qwen3 language model, without its output head.

    Args:
        cfg: Its ``config.json`` as a dict.
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        dim, self.head_dim = cfg["hidden_size"], cfg["head_dim"]
        self.theta = float(cfg.get("rope_parameters", cfg).get("rope_theta", 1e6))
        self.eps = cfg["rms_norm_eps"]
        self.embed = _w((cfg["vocab_size"], dim))
        self.layers = nn.ModuleList(
            _QwenLayer(
                dim,
                cfg["num_attention_heads"],
                cfg["num_key_value_heads"],
                self.head_dim,
                cfg["intermediate_size"],
            )
            for _ in range(cfg["num_hidden_layers"])
        )
        self.norm = _w((dim,))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Compute the last hidden states.

        Args:
            ids: Token ids, shape ``[B, S]``, padded on the right. Padding does not change the
                states of the tokens before it, since attention is causal.

        Returns:
            Hidden states, shape ``[B, S, hidden_size]``.
        """
        x = F.embedding(ids, self.embed)
        cos, sin = _rope(ids.shape[1], self.head_dim, self.theta, ids.device)
        for layer in self.layers:
            x = layer(x, cos, sin)
        return _rms(x, self.norm, self.eps)

    def load(self, sd: dict[str, torch.Tensor]) -> None:
        """Copy weights in the layout of ``transformers``' ``Qwen3Model``.

        Args:
            sd: The state dict.
        """
        with torch.no_grad():
            self.embed.copy_(sd["embed_tokens.weight"])
            self.norm.copy_(sd["norm.weight"])
            for i, layer in enumerate(cast(list[_QwenLayer], list(self.layers))):
                p = f"layers.{i}."
                a = p + "self_attn."
                layer.ln1.copy_(sd[p + "input_layernorm.weight"])
                layer.ln2.copy_(sd[p + "post_attention_layernorm.weight"])
                layer.qkv.copy_(torch.cat([sd[a + n + "_proj.weight"] for n in "qkv"]))
                layer.q_norm.copy_(sd[a + "q_norm.weight"])
                layer.k_norm.copy_(sd[a + "k_norm.weight"])
                layer.o.copy_(sd[a + "o_proj.weight"])
                gate, up = sd[p + "mlp.gate_proj.weight"], sd[p + "mlp.up_proj.weight"]
                layer.gate_up.copy_(torch.cat([gate, up]))
                layer.down.copy_(sd[p + "mlp.down_proj.weight"])


def _attend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    norms: tuple[torch.Tensor, torch.Tensor],
    rope_q: tuple[torch.Tensor, torch.Tensor],
    rope_k: tuple[torch.Tensor, torch.Tensor],
    mask: torch.Tensor,
) -> torch.Tensor:
    B, S, C = q.shape
    T, D = k.shape[1], C // heads
    q = _rms(q.view(B, S, heads, D), norms[0]).transpose(1, 2)
    k = _rms(k.view(B, T, heads, D), norms[1]).transpose(1, 2)
    v = v.view(B, T, heads, D).transpose(1, 2)
    q, k = _rotate(q, *rope_q), _rotate(k, *rope_k)
    att = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None, None])
    return att.transpose(1, 2).reshape(B, S, C)


class _ConditionerBlock(nn.Module):
    def __init__(self, dim: int, source_dim: int, heads: int, mlp: int) -> None:
        super().__init__()
        self.heads = heads
        self.n_self, self.n_cross, self.n_mlp = _w((dim,)), _w((dim,)), _w((dim,))
        self.self_qkv, self.self_o = _w((3 * dim, dim)), _w((dim, dim))
        self.self_qn, self.self_kn = _w((dim // heads,)), _w((dim // heads,))
        self.cross_q, self.cross_kv, self.cross_o = (
            _w((dim, dim)),
            _w((2 * dim, source_dim)),
            _w((dim, dim)),
        )
        self.cross_qn, self.cross_kn = _w((dim // heads,)), _w((dim // heads,))
        self.up, self.up_b = _w((mlp, dim)), _w((mlp,))
        self.down, self.down_b = _w((dim, mlp)), _w((dim,))

    def forward(self, x, source, rope_t, rope_s, mask_t, mask_s):
        h = F.linear(_rms(x, self.n_self), self.self_qkv)
        q, k, v = h.chunk(3, dim=-1)
        norms = (self.self_qn, self.self_kn)
        att = _attend(q, k, v, self.heads, norms, rope_t, rope_t, mask_t)
        x = x + F.linear(att, self.self_o)
        q = F.linear(_rms(x, self.n_cross), self.cross_q)
        k, v = F.linear(source, self.cross_kv).chunk(2, dim=-1)
        norms = (self.cross_qn, self.cross_kn)
        att = _attend(q, k, v, self.heads, norms, rope_t, rope_s, mask_s)
        x = x + F.linear(att, self.cross_o)
        h = F.gelu(F.linear(_rms(x, self.n_mlp), self.up, self.up_b))
        return x + F.linear(h, self.down, self.down_b)


class Conditioner(nn.Module):
    """Maps Qwen3 hidden states onto a prompt's T5 token ids, giving the model's text context.

    Args:
        cfg: Its ``config.json`` as a dict.
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        dim, heads = cfg["model_dim"], cfg["num_attention_heads"]
        self.head_dim, self.min_length = dim // heads, cfg["min_sequence_length"]
        self.embed = _w((cfg["target_vocab_size"], cfg["target_dim"]))
        mlp = int(dim * cfg["mlp_ratio"])
        self.blocks = nn.ModuleList(
            _ConditionerBlock(dim, cfg["source_dim"], heads, mlp)
            for _ in range(cfg["num_layers"])
        )
        self.out, self.out_b, self.norm = (
            _w((cfg["target_dim"], dim)),
            _w((cfg["target_dim"],)),
            _w((cfg["target_dim"],)),
        )

    def forward(
        self,
        source: torch.Tensor,
        source_mask: torch.Tensor,
        ids: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the text context.

        Args:
            source: Qwen3 hidden states, zero at padding, shape ``[B, S, source_dim]``.
            source_mask: Which source tokens are real, bool ``[B, S]``.
            ids: T5 token ids, shape ``[B, T]``, padded on the right.
            mask: Which T5 tokens are real, bool ``[B, T]``.

        Returns:
            Context, shape ``[B, max(T, min_sequence_length), target_dim]``, zero past each
            prompt's tokens.
        """
        x = F.embedding(ids, self.embed).to(source.dtype)
        rope_t = _rope(ids.shape[1], self.head_dim, 1e4, ids.device)
        rope_s = _rope(source.shape[1], self.head_dim, 1e4, ids.device)
        for block in self.blocks:
            x = block(x, source, rope_t, rope_s, mask, source_mask)
        x = _rms(F.linear(x, self.out, self.out_b), self.norm) * mask[..., None].to(
            x.dtype
        )
        return pad_context(x, max(x.shape[1], self.min_length))

    def load(self, sd: dict[str, torch.Tensor]) -> None:
        """Copy weights in the layout of ``diffusers``' ``AnimaTextConditioner``.

        Args:
            sd: The state dict.
        """
        with torch.no_grad():
            self.embed.copy_(sd["embed.weight"])
            self.out.copy_(sd["out_proj.weight"])
            self.out_b.copy_(sd["out_proj.bias"])
            self.norm.copy_(sd["norm.weight"])
            for i, b in enumerate(cast(list[_ConditionerBlock], list(self.blocks))):
                p = f"blocks.{i}."
                s, c = p + "self_attn.", p + "cross_attn."
                b.n_self.copy_(sd[p + "norm_self_attn.weight"])
                b.n_cross.copy_(sd[p + "norm_cross_attn.weight"])
                b.n_mlp.copy_(sd[p + "norm_mlp.weight"])
                b.self_qkv.copy_(torch.cat([sd[s + n + "_proj.weight"] for n in "qkv"]))
                b.self_o.copy_(sd[s + "o_proj.weight"])
                b.self_qn.copy_(sd[s + "q_norm.weight"])
                b.self_kn.copy_(sd[s + "k_norm.weight"])
                b.cross_q.copy_(sd[c + "q_proj.weight"])
                b.cross_kv.copy_(
                    torch.cat([sd[c + "k_proj.weight"], sd[c + "v_proj.weight"]])
                )
                b.cross_o.copy_(sd[c + "o_proj.weight"])
                b.cross_qn.copy_(sd[c + "q_norm.weight"])
                b.cross_kn.copy_(sd[c + "k_norm.weight"])
                b.up.copy_(sd[p + "mlp.0.weight"])
                b.up_b.copy_(sd[p + "mlp.0.bias"])
                b.down.copy_(sd[p + "mlp.2.weight"])
                b.down_b.copy_(sd[p + "mlp.2.bias"])


def _tokenize(
    tok, prompts: Sequence[str], pad: int, device: torch.device, bucket: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = [e.ids for e in tok.encode_batch(list(prompts))]
    length = max(1, max(len(r) for r in rows))  # the empty prompt gets one masked token
    length = -(-length // bucket) * bucket
    ids = torch.full((len(rows), length), pad, dtype=torch.long)
    mask = torch.zeros(len(rows), length, dtype=torch.bool)
    for i, r in enumerate(rows):
        ids[i, : len(r)] = torch.tensor(r, dtype=torch.long)
        mask[i, : len(r)] = True
    if not any(rows):
        ids.zero_()
    return ids.to(device), mask.to(device)


class TextEncoder:
    """Turns prompts into the model's text context.

    A prompt is encoded by Qwen3, and its hidden states are then mapped onto the prompt's T5
    token ids by the conditioner.

    Args:
        qwen: The language model.
        conditioner: The conditioner.
        tokenizer: Qwen3's ``tokenizers.Tokenizer``.
        t5_tokenizer: T5's ``tokenizers.Tokenizer``. Only its token ids are used.
        max_length: Prompts are truncated to this many tokens by both tokenizers.
    """

    def __init__(
        self,
        qwen: Qwen3,
        conditioner: Conditioner,
        tokenizer,
        t5_tokenizer,
        max_length: int = 512,
        graph: bool = True,
    ) -> None:
        self.qwen, self.conditioner, self.max_length = qwen, conditioner, max_length
        self.tokenizer, self.t5_tokenizer = tokenizer, t5_tokenizer
        self.graph = graph
        self.graphs: dict[tuple, Graphed] = {}
        for t in (tokenizer, t5_tokenizer):
            t.enable_truncation(max_length)
            t.no_padding()

    @classmethod
    def from_repo(
        cls,
        repo: str,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> "TextEncoder":
        """Load the text components of a Diffusers Anima repository.

        Args:
            repo: Local folder or Hugging Face repository id.
            device: Device to load them onto.
            dtype: Dtype of their weights.

        Returns:
            The encoder, with frozen weights.
        """
        from tokenizers import Tokenizer

        def path(name: str) -> str:
            local = os.path.join(repo, name)
            if os.path.exists(local):
                return local
            from huggingface_hub import hf_hub_download

            return hf_hub_download(repo, name)

        def config(name: str) -> dict:
            with open(path(name), encoding="utf-8") as f:
                return json.load(f)

        qwen = Qwen3(config("text_encoder/config.json"))
        qwen.load(load_file(path("text_encoder/model.safetensors")))
        cond = Conditioner(config("text_conditioner/config.json"))
        cond.load(
            load_file(path("text_conditioner/diffusion_pytorch_model.safetensors"))
        )
        return cls(
            qwen.to(device, dtype).eval(),
            cond.to(device, dtype).eval(),
            Tokenizer.from_file(path("tokenizer/tokenizer.json")),
            Tokenizer.from_file(path("t5_tokenizer/tokenizer.json")),
        )

    @torch.no_grad()
    def encode(self, prompts: str | Sequence[str], dtype: torch.dtype) -> torch.Tensor:
        """Encode prompts into text embeddings.

        Args:
            prompts: One prompt or several.
            dtype: Dtype of the result. Use the model's compute dtype.

        Returns:
            Text embeddings, shape ``[B, S, text_dim]``, on the encoder's device, zero past each
            prompt's tokens. The prompts are encoded together in one pass.
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        device = self.qwen.embed.device
        graph = self.graph and device.type == "cuda"
        bucket = BUCKET if graph else 1
        ids, mask = _tokenize(self.tokenizer, prompts, QWEN_PAD, device, bucket)
        t5_ids, t5_mask = _tokenize(self.t5_tokenizer, prompts, 0, device, bucket)
        inputs = (ids, mask, t5_ids, t5_mask)
        if not graph:
            return self._run(*inputs).to(dtype)
        key = tuple(t.shape for t in inputs)
        runner = self.graphs.get(key)
        if runner is None:
            runner = self.graphs[key] = Graphed(self._run, inputs)
        return runner(*inputs).to(dtype)

    def _run(self, ids, mask, t5_ids, t5_mask) -> torch.Tensor:
        h = self.qwen(ids) * mask[..., None].to(self.qwen.embed.dtype)
        return self.conditioner(h, mask, t5_ids, t5_mask)
