"""
DA-RNN  --  Dual-Stage Attention-Based Recurrent Neural Network.

Reference architecture:
    Yao Qin, Dongjin Song, Haifeng Chen, Wei Cheng, Guofei Jiang, Garrison W. Cottrell,
    "A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction",
    IJCAI 2017.  arXiv:1704.02971.   Equation numbers below refer to that paper.

Adapted from Zhenye-Na/DA-RNN (PyTorch), with these deliberate changes:
    * faithful  v^T tanh( W[h;s] + U x )  scoring in BOTH stages
      (the reference collapses stage 1 to a single Linear)
    * LayerNorm on each recurrent step   (NOT BatchNorm -- Wen & Li 2023 use BN,
      which is unstable on small non-stationary financial batches)
    * dropout on the encoded sequence, the context vector and the final concat
      (Sebastian & Tantia 2025, dropout 0.2)
    * strictly linear output head (eq. 22) -- no ReLU (Wen & Li's ReLU head would
      zero every negative return)
    * per-step LSTMCell so the two attention stages have explicit hooks
    * returns the attention tensors so alpha_kt / beta_it can be logged

Shapes:  B = batch, T = window length, n = #driving series,
         m = encoder hidden, p = decoder hidden.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class InputAttentionEncoder(nn.Module):
    """Stage 1 -- input (feature) attention + LSTM encoder.  DA-RNN eqs. 8-11."""

    def __init__(self, n_features: int, T: int, hidden: int, dropout: float = 0.2):
        super().__init__()
        self.n_features = n_features
        self.T = T
        self.hidden = hidden

        self.lstm = nn.LSTMCell(n_features, hidden)
        # e^k_t = v_e^T tanh( W_e [h_{t-1}; s_{t-1}]  +  U_e x^k )          (eq. 8)
        self.W_e = nn.Linear(2 * hidden, T, bias=False)
        self.U_e = nn.Linear(T, T, bias=False)
        self.v_e = nn.Linear(T, 1, bias=False)

        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)

    def forward(self, X: torch.Tensor):
        # X: (B, T, n)
        B = X.size(0)
        h = X.new_zeros(B, self.hidden)
        s = X.new_zeros(B, self.hidden)

        Ux = self.U_e(X.permute(0, 2, 1))                       # (B, n, T)
        encoded = X.new_zeros(B, self.T, self.hidden)
        alphas = X.new_zeros(B, self.T, self.n_features)

        for t in range(self.T):
            q = self.W_e(torch.cat([h, s], dim=1))              # (B, T)
            e = self.v_e(torch.tanh(q.unsqueeze(1) + Ux)).squeeze(-1)   # (B, n)
            alpha = torch.softmax(e, dim=1)                     # (B, n)  -- weights over driving series
            x_tilde = alpha * X[:, t, :]                        # (B, n)  (eq. 10)
            h, s = self.lstm(x_tilde, (h, s))                   # (eq. 11)
            h = self.norm(h)
            encoded[:, t, :] = h
            alphas[:, t, :] = alpha

        return self.drop(encoded), alphas


class TemporalAttentionDecoder(nn.Module):
    """Stage 2 -- temporal attention + LSTM decoder + linear head.  DA-RNN eqs. 12-22."""

    def __init__(self, enc_hidden: int, dec_hidden: int, T: int, dropout: float = 0.2):
        super().__init__()
        self.enc_hidden = enc_hidden
        self.dec_hidden = dec_hidden
        self.T = T

        # l^i_t = v_d^T tanh( W_d [d_{t-1}; s'_{t-1}]  +  U_d h_i )         (eq. 12)
        self.W_d = nn.Linear(2 * dec_hidden, enc_hidden, bias=False)
        self.U_d = nn.Linear(enc_hidden, enc_hidden, bias=False)
        self.v_d = nn.Linear(enc_hidden, 1, bias=False)

        self.y_fuse = nn.Linear(enc_hidden + 1, 1)              # y~_{t-1} (eq. 15)
        self.lstm = nn.LSTMCell(1, dec_hidden)                  # (eq. 16)
        self.norm = nn.LayerNorm(dec_hidden)
        self.drop = nn.Dropout(dropout)

        # y_hat = v_y^T ( W_y [d_T; c_T] + b_w ) + b_v  -- linear, no activation (eq. 22)
        self.head = nn.Sequential(
            nn.Linear(dec_hidden + enc_hidden, dec_hidden),
            nn.Linear(dec_hidden, 1),
        )

    def forward(self, encoded: torch.Tensor, y_hist: torch.Tensor):
        # encoded: (B, T, m) ; y_hist: (B, T)  -- realized past returns through day tau,
        # target is the return on day tau+1 (strict 1-step-ahead, no gap).
        B = encoded.size(0)
        d = encoded.new_zeros(B, self.dec_hidden)
        s = encoded.new_zeros(B, self.dec_hidden)

        Uh = self.U_d(encoded)                                  # (B, T, m)
        context = encoded.new_zeros(B, self.enc_hidden)
        beta = encoded.new_zeros(B, self.T)

        for t in range(self.T):
            q = self.W_d(torch.cat([d, s], dim=1))              # (B, m)
            l = self.v_d(torch.tanh(q.unsqueeze(1) + Uh)).squeeze(-1)   # (B, T)
            beta = torch.softmax(l, dim=1)                      # (B, T)  -- weights over encoder steps
            context = torch.bmm(beta.unsqueeze(1), encoded).squeeze(1)  # (B, m)  (eq. 14)
            y_tilde = self.y_fuse(torch.cat([context, y_hist[:, t:t + 1]], dim=1))  # (B, 1)  (eq. 15)
            d, s = self.lstm(y_tilde, (d, s))                   # (eq. 16)
            d = self.norm(d)

        y_hat = self.head(self.drop(torch.cat([d, context], dim=1))).squeeze(-1)   # (B,)  (eq. 22)
        return y_hat, beta            # beta: (B, T) temporal weights from the final step


class DARNN(nn.Module):
    def __init__(
        self,
        n_features: int,
        T: int,
        enc_hidden: int = 32,
        dec_hidden: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = InputAttentionEncoder(n_features, T, enc_hidden, dropout)
        self.decoder = TemporalAttentionDecoder(enc_hidden, dec_hidden, T, dropout)

    def forward(self, X: torch.Tensor, y_hist: torch.Tensor):
        encoded, alphas = self.encoder(X)
        y_hat, beta = self.decoder(encoded, y_hist)
        return y_hat, alphas, beta

    @staticmethod
    def attention_entropy(alphas: torch.Tensor) -> torch.Tensor:
        """Mean entropy of the stage-1 weights; add lambda*this to the loss to
        push the encoder toward using a few driving series rather than smearing."""
        p = alphas.clamp_min(1e-9)
        return -(p * p.log()).sum(-1).mean()


DARNN.extra_names = ("alpha", "beta")     # alpha: (B, T, n) input attention, beta: (B, T) temporal attention


def build(n_features: int, T: int, params: dict) -> DARNN:
    return DARNN(n_features, T, **params)
