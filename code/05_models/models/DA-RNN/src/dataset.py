"""Sliding-window dataset for strict 1-step-ahead direction classification.

Sample i:
    X       = driving series over days [i, i+T)      shape (T, n)
    y_hist  = realised 0/1 direction  [i, i+T)       shape (T,)
    target  = direction on day i+T                   scalar, 0.0 or 1.0
    weight  = sample weight for day i+T              scalar

Every value in the window is dated strictly before the predicted day, and the
sentiment columns were already lagged in the pipeline, so there is no
contemporaneous leakage.

The target is float32, not long: BCEWithLogitsLoss expects float targets.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, T: int, weights: np.ndarray | None = None,
                 y_hist: np.ndarray | None = None):
        self.X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        self.y = torch.as_tensor(np.asarray(y), dtype=torch.float32)   # 0/1, float for BCE
        # The history fed to the model is NOT the 0/1 target.  A NARX decoder
        # (DA-RNN eq. 15) fuses the previous target value into every timestep;
        # handing it a binary sequence throws away the magnitude the regression
        # variant had and the model collapses to one class.  `y_hist` carries
        # the continuous arcsinh-scaled flow instead.  Models that ignore the
        # argument (BiLSTM, iTransformer) are unaffected.
        yh = self.y if y_hist is None else torch.as_tensor(np.asarray(y_hist), dtype=torch.float32)
        self.y_hist = yh
        w = np.ones(len(self.y), dtype=np.float32) if weights is None else np.asarray(weights)
        self.w = torch.as_tensor(w, dtype=torch.float32)
        self.T = int(T)
        if len(self.X) <= self.T:
            raise ValueError(f"need > T={self.T} rows, got {len(self.X)}")

    def __len__(self) -> int:
        return self.X.size(0) - self.T

    def __getitem__(self, i: int):
        sl = slice(i, i + self.T)
        return self.X[sl], self.y_hist[sl], self.y[i + self.T], self.w[i + self.T]

    def target_positions(self) -> np.ndarray:
        """Row index (into X/y) of each sample's prediction target."""
        return np.arange(self.T, self.X.size(0))
