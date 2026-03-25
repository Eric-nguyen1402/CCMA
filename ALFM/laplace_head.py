# laplace_head.py
from __future__ import annotations
from typing import Optional, Tuple
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearSoftmaxHead(nn.Module):
    """
    Linear softmax classifier for frozen features.
    W: (C, d), b: (C,)
    """
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

    @torch.no_grad()
    def get_probs(self, x: np.ndarray | torch.Tensor, batch_size: int = 8192) -> torch.Tensor:
        self.eval()
        if isinstance(x, np.ndarray):
            X = torch.from_numpy(x).float()
        else:
            X = x
        probs = []
        for i in range(0, X.size(0), batch_size):
            logits = self.forward(X[i : i + batch_size])
            probs.append(F.softmax(logits, dim=1))
        return torch.cat(probs, dim=0)

class LastLayerLaplace:
    """
    Diagonal (one-vs-rest) Laplace approx for linear softmax head.
    Posterior per-class: w_c ~ N(μ_c, Σ_c), with Σ_c diagonal in weight space.
    """
    def __init__(self, head: LinearSoftmaxHead, weight_decay: float = 1e-4, device: Optional[torch.device]=None):
        self.head = head
        self.weight_decay = float(weight_decay)
        self.device = device or next(head.parameters()).device
        self.mu_W: Optional[torch.Tensor] = None    # (C, d)
        self.mu_b: Optional[torch.Tensor] = None    # (C,)
        self.var_W: Optional[torch.Tensor] = None   # (C, d) diagonal
        self.var_b: Optional[torch.Tensor] = None   # (C,)

    @torch.no_grad()
    def fit(self, feats: np.ndarray | torch.Tensor, labels: np.ndarray | torch.Tensor, batch_size: int = 8192):
        """
        Fit diagonal Gaussian around MAP for last layer.
        Uses one-vs-rest diagonal Hessian approx: H_c ≈ Σ_i p_c(1-p_c) * x_i^2 + λ
        where x_i^2 is elementwise square of features.
        """
        self.head.eval()
        # cache MAP
        W = self.head.fc.weight.detach().clone()  # (C,d)
        b = self.head.fc.bias.detach().clone()    # (C,)
        self.mu_W, self.mu_b = W, b

        if isinstance(feats, np.ndarray):
            X = torch.from_numpy(feats).float().to(self.device)
        else:
            X = feats.to(self.device)
        if isinstance(labels, np.ndarray):
            y = torch.from_numpy(labels.reshape(-1)).long().to(self.device)
        else:
            y = labels.reshape(-1).long().to(self.device)

        C, d = W.shape
        # accumulate diagonal Hessian per class
        H_W = torch.zeros((C, d), device=self.device)
        H_b = torch.zeros((C,), device=self.device)

        for i in range(0, X.size(0), batch_size):
            xb = X[i : i + batch_size]
            logits = xb @ W.T + b  # (B,C)
            p = F.softmax(logits, dim=1)  # (B,C)
            # For multinomial logistic regression, diagonal blocks approx:
            # diag(H_c) ≈ Σ_i p_ic(1-p_ic) * (x_i^2), and for bias: Σ_i p_ic(1-p_ic)
            w_c = p * (1.0 - p)  # (B,C)
            x2 = xb.pow(2)       # (B,d)
            # accumulate
            H_b += w_c.sum(dim=0)
            # (C,d): sum over batch of outer diag; do as matmul of w_c^T @ x2
            H_W += w_c.T @ x2

        # Add weight decay λI
        H_W += self.weight_decay
        H_b += self.weight_decay

        # Diagonal covariance = inverse diagonal Hessian
        self.var_W = 1.0 / H_W.clamp_min(1e-9)
        self.var_b = 1.0 / H_b.clamp_min(1e-9)

    @torch.no_grad()
    def predictive_probs(
        self,
        feats: np.ndarray | torch.Tensor,
        n_samples: int = 20,
        batch_size: int = 8192,
    ) -> torch.Tensor:
        """
        Monte Carlo over last-layer Gaussian (no dropout).
        """
        assert self.mu_W is not None and self.var_W is not None, "Call fit() first"
        if isinstance(feats, np.ndarray):
            X = torch.from_numpy(feats).float().to(self.device)
        else:
            X = feats.to(self.device)

        C, d = self.mu_W.shape
        probs_accum = torch.zeros((X.size(0), C), device=self.device)
        # Precompute std
        std_W = torch.sqrt(self.var_W)
        std_b = torch.sqrt(self.var_b)

        for s in range(n_samples):
            eps_W = torch.randn_like(self.mu_W)
            eps_b = torch.randn_like(self.mu_b)
            Ws = self.mu_W + eps_W * std_W
            bs = self.mu_b + eps_b * std_b
            out = []
            for i in range(0, X.size(0), batch_size):
                logits = X[i : i + batch_size] @ Ws.T + bs
                out.append(F.softmax(logits, dim=1))
            probs_accum += torch.cat(out, dim=0)

        return probs_accum / float(n_samples)

    @torch.no_grad()
    def epistemic_variance(
        self, feats: np.ndarray | torch.Tensor, n_samples: int = 20, batch_size: int = 8192
    ) -> torch.Tensor:
        """
        Variance of predictive probabilities across weight samples (per class).
        """
        if isinstance(feats, np.ndarray):
            X = torch.from_numpy(feats).float().to(self.device)
        else:
            X = feats.to(self.device)
        C = self.mu_W.size(0)
        probs_samples = []
        for s in range(n_samples):
            eps_W = torch.randn_like(self.mu_W)
            eps_b = torch.randn_like(self.mu_b)
            Ws = self.mu_W + eps_W * torch.sqrt(self.var_W)
            bs = self.mu_b + eps_b * torch.sqrt(self.var_b)
            out = []
            for i in range(0, X.size(0), batch_size):
                logits = X[i : i + batch_size] @ Ws.T + bs
                out.append(F.softmax(logits, dim=1))
            probs_samples.append(torch.cat(out, dim=0))
        P = torch.stack(probs_samples, dim=0)  # (S, N, C)
        return P.var(dim=0)                    # (N, C)
