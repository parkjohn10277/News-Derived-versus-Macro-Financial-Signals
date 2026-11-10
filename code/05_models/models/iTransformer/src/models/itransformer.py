"""
iTransformer -- inverted Transformer for multivariate forecasting.

Paper : Liu, Y., Hu, T., Zhang, H., Wu, H., Wang, S., Ma, L. & Long, M. (2024),
        "iTransformer: Inverted Transformers Are Effective for Time Series
        Forecasting", ICLR 2024.  arXiv:2310.06625
Code  : https://github.com/thuml/iTransformer  (MIT)

Self-contained port of the official `model/iTransformer.py` plus the pieces of
`layers/Embed.py`, `layers/Transformer_EncDec.py` and
`layers/SelfAttention_Family.py` it uses, so the folder does not depend on the
upstream package (which also pulls in Reformer/Flowformer variants).  Verified
component by component against upstream:

  * DataEmbeddingInverted  == DataEmbedding_inverted  (Linear(seq_len, d_model)
                              applied to x.permute(0, 2, 1); no time tokens)
  * FullAttention          == FullAttention(mask_flag=False)   1/sqrt(E) scaling
  * AttentionLayer         == AttentionLayer
  * EncoderLayer           == EncoderLayer   (residual -> norm1 -> conv1d FFN ->
                              norm2, gelu/relu)
  * Encoder                == Encoder(norm_layer=LayerNorm)
  * projector              == Linear(d_model, pred_len)
  * use_norm               == the non-stationary per-window normalisation and
                              re-denormalisation

The inversion is the whole point: each *variate's* full lookback window becomes
one token, and self-attention runs across variates rather than across time.

Deliberate changes for this study
---------------------------------
* `pred_len = 1` -- one-step-ahead, matching every other model here.
* Explicit `target_index` instead of the upstream MS convention that the target
  is the last column.  `experiment_features` always emits the KRX target first,
  so the default is 0.
* No timestamp tokens.  Upstream's inverted embedding can append `x_mark`
  covariates as extra tokens; calendar variables are not part of the E0-E7
  information sets, so adding them would smuggle in another block.
* The returned `alpha` is the target token's attention over the variate tokens,
  averaged over encoder layers and heads.  It is a *diagnostic*: attention mass
  is not a causal or predictive importance measure.  Use the ablation /
  permutation analyses for importance claims.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ITransformerCfg:
    seq_len: int = 30
    pred_len: int = 1
    d_model: int = 128
    n_heads: int = 4
    e_layers: int = 2
    d_ff: int = 256
    dropout: float = 0.1
    activation: str = "gelu"
    use_norm: bool = True
    output_attention: bool = True

    def validate(self) -> None:
        if self.seq_len <= 0 or self.pred_len <= 0:
            raise ValueError("seq_len and pred_len must be positive")
        if self.d_model <= 0 or self.n_heads <= 0:
            raise ValueError("d_model and n_heads must be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.e_layers <= 0 or self.d_ff <= 0:
            raise ValueError("e_layers and d_ff must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.activation not in {"relu", "gelu"}:
            raise ValueError("activation must be 'relu' or 'gelu'")


class DataEmbeddingInverted(nn.Module):
    """One token per variate: its whole lookback window -> d_model."""

    def __init__(self, seq_len: int, d_model: int, dropout: float):
        super().__init__()
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.value_embedding(x.permute(0, 2, 1)))   # [B,T,N] -> [B,N,D]


class FullAttention(nn.Module):
    """Unmasked full self-attention across variate tokens."""

    def __init__(self, dropout: float = 0.1, output_attention: bool = False):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.output_attention = bool(output_attention)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        scale = 1.0 / sqrt(q.shape[-1])
        scores = torch.einsum("bnhe,bmhe->bhnm", q, k)
        attn = self.dropout(torch.softmax(scale * scores, dim=-1))
        out = torch.einsum("bhnm,bmhd->bnhd", attn, v)
        return out.contiguous(), (attn if self.output_attention else None)


class AttentionLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, output_attention: bool):
        super().__init__()
        d_head = d_model // n_heads
        self.n_heads = n_heads
        self.query_projection = nn.Linear(d_model, d_head * n_heads)
        self.key_projection = nn.Linear(d_model, d_head * n_heads)
        self.value_projection = nn.Linear(d_model, d_head * n_heads)
        self.out_projection = nn.Linear(d_head * n_heads, d_model)
        self.inner_attention = FullAttention(dropout, output_attention)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        B, N, _ = q.shape
        H = self.n_heads
        q = self.query_projection(q).view(B, N, H, -1)
        k = self.key_projection(k).view(B, N, H, -1)
        v = self.value_projection(v).view(B, N, H, -1)
        out, attn = self.inner_attention(q, k, v)
        return self.out_projection(out.view(B, N, -1)), attn


class EncoderLayer(nn.Module):
    def __init__(self, cfg: ITransformerCfg):
        super().__init__()
        self.attention = AttentionLayer(cfg.d_model, cfg.n_heads, cfg.dropout, cfg.output_attention)
        self.conv1 = nn.Conv1d(cfg.d_model, cfg.d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(cfg.d_ff, cfg.d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.activation = F.relu if cfg.activation == "relu" else F.gelu

    def forward(self, x: torch.Tensor):
        new_x, attn = self.attention(x, x, x)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(1, 2))))
        y = self.dropout(self.conv2(y).transpose(1, 2))
        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, cfg: ITransformerCfg):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(cfg) for _ in range(cfg.e_layers)])
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, x: torch.Tensor):
        attns = []
        for layer in self.layers:
            x, attn = layer(x)
            attns.append(attn)
        return self.norm(x), attns


class ITransformer(nn.Module):
    """Inverted Transformer with an explicit target token.

    forward(X, y_hist) -> (y_hat, alpha)
        X      [B, T, n]  window of driving series (scaled)
        y_hist [B, T]     accepted for template compatibility and unused -- the
                          target's own history is already column `target_index`
                          of X, exactly as upstream treats the target variate
        y_hat  [B]        next-day prediction in the scaled target space
        alpha  [B, n]     target token's attention over variate tokens
    """

    extra_names = ("alpha",)

    def __init__(self, n_features: int, cfg: ITransformerCfg, target_index: int = 0):
        super().__init__()
        cfg.validate()
        if cfg.pred_len != 1:
            raise ValueError("this adapter is one-step-ahead; set pred_len=1")
        if not 0 <= target_index < n_features:
            raise ValueError(f"target_index={target_index} invalid for n_features={n_features}")
        self.n_features = int(n_features)
        self.target_index = int(target_index)
        self.cfg = cfg
        self.enc_embedding = DataEmbeddingInverted(cfg.seq_len, cfg.d_model, cfg.dropout)
        self.encoder = Encoder(cfg)
        self.projector = nn.Linear(cfg.d_model, cfg.pred_len, bias=True)

    def forward(self, X: torch.Tensor, y_hist: torch.Tensor | None = None):
        if X.ndim != 3 or X.shape[1] != self.cfg.seq_len or X.shape[2] != self.n_features:
            raise ValueError(f"expected X [B,{self.cfg.seq_len},{self.n_features}], got {tuple(X.shape)}")

        if self.cfg.use_norm:                       # non-stationary normalisation (upstream)
            means = X.mean(dim=1, keepdim=True).detach()
            centered = X - means
            stdev = torch.sqrt(torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_model = centered / stdev
        else:
            means = stdev = None
            x_model = X

        enc_out, attns = self.encoder(self.enc_embedding(x_model))      # [B,N,D]
        dec_out = self.projector(enc_out).permute(0, 2, 1)              # [B,N,S] -> [B,S,N]
        if self.cfg.use_norm:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1) + means[:, 0, :].unsqueeze(1)

        pred = dec_out[:, -1, self.target_index]
        if not self.cfg.output_attention:
            return pred
        usable = [a for a in attns if a is not None]
        if not usable:
            alpha = torch.full((X.shape[0], self.n_features), float("nan"),
                               device=X.device, dtype=X.dtype)
        else:
            # each layer [B,H,N,N] -> mean over layers then heads -> [B,N,N]
            alpha = torch.stack(usable, dim=0).mean(dim=0).mean(dim=1)[:, self.target_index, :]
        return pred, alpha


def build(n_features: int, T: int, params: dict) -> ITransformer:
    p = dict(params or {})
    target_index = int(p.pop("target_index", 0))     # experiment_features emits KRX first
    return ITransformer(n_features, ITransformerCfg(seq_len=T, **p), target_index)
