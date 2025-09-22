
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn.functional as F

@dataclass
class TemperatureCalibrator:
    T: float = 1.0

    def apply(self, logits):
        Tt = torch.tensor(self.T, dtype=logits.dtype, device=logits.device)
        return logits / Tt

    @staticmethod
    def fit_from_logits(logits: torch.Tensor, labels: torch.Tensor, init_T: float = 1.0, epochs: int = 200, lr: float = 0.01) -> "TemperatureCalibrator":
        T = torch.nn.Parameter(torch.tensor(init_T, dtype=logits.dtype, device=logits.device))
        opt = torch.optim.LBFGS([T], lr=lr, max_iter=epochs, line_search_fn="strong_wolfe")

        def nll():
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(logits / T, labels)
            loss.backward()
            return loss

        opt.step(nll)
        with torch.no_grad():
            T_clamped = float(torch.clamp(T, 1e-3, 100.0).item())
        return TemperatureCalibrator(T=T_clamped)

    def state_dict(self):
        return {"T": self.T}

    @staticmethod
    def from_state_dict(sd: dict) -> "TemperatureCalibrator":
        return TemperatureCalibrator(T=float(sd.get("T", 1.0)))
