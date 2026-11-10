"""
BiLSTM -- the bidirectional counterpart of the LSTM baseline.

Reference: Han, J. J. & Kim, H.-j. (2021/2023), "Prediction of Investor-Specific Trading
Trends in South Korean Stock Markets Using a BiLSTM Prediction Model Based on Sentiment
Analysis of Financial News Articles", Journal of Behavioral Finance 24(4), 398-410.
They predict next-day net buy/sell of foreign, individual and institutional investors on
KOSPI/KOSDAQ from daily LDA-topic occurrence rates with a BiLSTM (vs MLP, LSTM), evaluated
by AUC. The paper defines the BiLSTM only as "two layers running in direct and reverse
order" (= bidirectional=True) and reports no hidden size, depth, window, optimizer or
epochs; those are therefore taken from the shared protocol and matched to the LSTM
baseline, so the LSTM/BiLSTM contrast isolates bidirectionality. Architecture origin:
Graves & Schmidhuber (2005), Neural Networks 18, 602-610.

    
    BiLSTM(hidden per direction, num_layers) -> last time step (fwd ++ bwd) -> Dropout -> Linear(1)
Same hidden size per direction as the LSTM, so the only difference is bidirectionality.
The target's own history enters through X (the KRX column); y_hist is unused.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BiLSTMModel(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=num_layers, batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden, 1)

    def forward(self, X: torch.Tensor, y_hist: torch.Tensor | None = None) -> torch.Tensor:
        out, _ = self.lstm(X)                 # (B, T, 2*hidden): forward and backward states concatenated
        return self.fc(self.dropout(out[:, -1, :])).squeeze(-1)


def build(n_features: int, T: int, params: dict) -> BiLSTMModel:
    return BiLSTMModel(n_features, **params)
