"""Knowledge-distillation losses for Module 2 (CCKD).

Ported faithfully from the verified ssvep_ensemble/models/kd_losses.py.

Framing: privileged-information distillation. Teacher observes 9 channels
(available only at train time); student observes 1/3 channels of the SAME
trial. Feature-based terms below all operate on the SAME flat penultimate
feature vectors exposed by `model(x, return_feat=True)`, so that different
KD mechanisms are compared on strictly equal footing (identical feature
access point, only the loss formula differs):
  (1) logit_kd_loss  -- response-based (Hinton et al., 2015). Matches the
      40-class output distribution.
  (2) relational_kd_loss -- similarity-preserving KD (Tung & Mori, ICCV
      2019). Matches the *structure* of pairwise sample similarity (Gram
      matrix, Frobenius-normalized) instead of forcing absolute feature
      equality -- appropriate when teacher/student observe different input
      spaces. This is the mechanism CCKD (the proposed method) uses.
  (3) fitnets_loss -- feature imitation (Romero et al., 2015). Forces
      absolute (L2-normalized) feature equality; no projection head is
      needed here because every backbone in this study collapses channels
      early, so teacher/student penultimate feature dims already match.
  (4) attention_transfer_loss -- activation-based attention transfer
      (Zagoruyko & Komodakis, 2017), adapted to operate on flat feature
      vectors (no explicit spatial map) for architecture-agnostic
      comparability with the other three losses above.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def logit_kd_loss(z_s: torch.Tensor, z_t: torch.Tensor, tau: float = 4.0) -> torch.Tensor:
    """KL(softmax(z_t/tau) || softmax(z_s/tau)) * tau^2."""
    p_t = F.softmax(z_t / tau, dim=1)
    log_p_s = F.log_softmax(z_s / tau, dim=1)
    return F.kl_div(log_p_s, p_t, reduction="batchmean") * (tau ** 2)


def relational_kd_loss(f_s: torch.Tensor, f_t: torch.Tensor) -> torch.Tensor:
    """Similarity-preserving / relational KD.

    f_*: [B, d] (or [B, ...], flattened). G = normalize_row(F F^T);
    loss = ||G_s - G_t||_F^2 / B^2.
    """
    f_s = f_s.flatten(1)
    f_t = f_t.flatten(1)
    G_s = F.normalize(f_s @ f_s.t(), p=2, dim=1)
    G_t = F.normalize(f_t @ f_t.t(), p=2, dim=1)
    b = f_s.size(0)
    return (G_s - G_t).pow(2).sum() / (b * b)


def fitnets_loss(f_s: torch.Tensor, f_t: torch.Tensor) -> torch.Tensor:
    """Feature-imitation KD (Romero et al., 2015, "FitNets").

    Forces the student's feature vector to match the teacher's absolute
    representation (per-sample L2-normalized first, since raw feature scale
    is arbitrary and would otherwise dominate the loss): MSE(f_s/||f_s||,
    f_t/||f_t||). No learned projection head, since feature dims already
    match across teacher/student for every backbone studied here.
    """
    f_s = F.normalize(f_s.flatten(1), p=2, dim=1)
    f_t = F.normalize(f_t.flatten(1), p=2, dim=1)
    return F.mse_loss(f_s, f_t)


def attention_transfer_loss(f_s: torch.Tensor, f_t: torch.Tensor) -> torch.Tensor:
    """Activation-based attention transfer (Zagoruyko & Komodakis, 2017),
    adapted to flat feature vectors: attention = L2-normalized elementwise
    squared activation ("what the network attends to along the feature
    axis"); loss = MSE between student and teacher attention vectors.
    """
    a_s = f_s.flatten(1).pow(2)
    a_t = f_t.flatten(1).pow(2)
    a_s = F.normalize(a_s, p=2, dim=1)
    a_t = F.normalize(a_t, p=2, dim=1)
    return F.mse_loss(a_s, a_t)


_FEATURE_LOSSES = {
    "relational": relational_kd_loss,
    "fitnets": fitnets_loss,
    "attention": attention_transfer_loss,
}


class CCKDLoss(nn.Module):
    """Total student loss: CE + lambda_kd * logit_kd + lambda_sp * feature_loss.

        crit = CCKDLoss(tau=4, lambda_kd=1.0, lambda_sp=100.0, feature_loss="relational")
        z_s, f_s = student(x_student, return_feat=True)
        with torch.no_grad():
            z_t, f_t = teacher(x_teacher, return_feat=True)  # teacher frozen
        loss, parts = crit(z_s, f_s, z_t, f_t, y)

    mode selects which terms are active (matches the paper's comparison methods):
      "logit"      -> Response-based KD only
      "relational" -> Proposed CCKD (multi-level relational) only
      "hybrid"     -> Response-based + Relational KD combined
      "fitnets"    -> Feature-imitation KD only (baseline, Romero et al. 2015)
      "attention"  -> Attention-transfer KD only (baseline, Zagoruyko & Komodakis 2017)
    `feature_loss` picks which function fills the lambda_sp slot; defaults to
    the proposed relational mechanism and is only overridden for the two new
    baselines above.
    """

    def __init__(self, tau: float = 4.0, lambda_kd: float = 1.0, lambda_sp: float = 100.0,
                 feature_loss: str = "relational") -> None:
        super().__init__()
        self.tau, self.lkd, self.lsp = tau, lambda_kd, lambda_sp
        self.feature_loss_name = feature_loss
        self.feature_loss_fn = _FEATURE_LOSSES[feature_loss]

    def forward(self, z_s, feats_s, z_t, feats_t, y):
        l_ce = F.cross_entropy(z_s, y)
        l_kd = logit_kd_loss(z_s, z_t, self.tau)
        if isinstance(feats_s, (list, tuple)):
            l_sp = sum(self.feature_loss_fn(a, b) for a, b in zip(feats_s, feats_t)) / len(feats_s)
        else:
            l_sp = self.feature_loss_fn(feats_s, feats_t)
        total = l_ce + self.lkd * l_kd + self.lsp * l_sp
        return total, {"ce": l_ce.item(), "kd": l_kd.item(), "sp": l_sp.item()}
