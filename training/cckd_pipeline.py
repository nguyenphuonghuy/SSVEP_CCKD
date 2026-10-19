"""Engine for Module 2 (Cross-Channel Relational KD) experiments.

Ported faithfully from the verified ssvep_ensemble/cckd_pipeline.py, adapted to
plug into this project's datasets.public_ssvep.PublicSSVEPDataset loader.

Fixed design decisions (matching the paper's Experimental Setup):
  - ONE fixed subject-independent split (more test subjects, no re-splitting).
    Variance is estimated over multiple init seeds, not over the split.
  - KD = privileged-information distillation: teacher sees 9 channels
    (train-time only) -> student sees 1/3 channels of the SAME trial.
  - Relational loss = similarity-preserving (Gram matrix), not raw cosine
    alignment against absolute features.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset

from models.cckd_backbones import build_backbone, apply_filter_bank
from training.kd_losses import CCKDLoss

# Position of each channel WITHIN the 9-channel teacher array
# (Pz, PO5, PO3, POz, PO4, PO6, O1, Oz, O2) -- matches dataset.teacher_channel_names.
CH_POS = {
    "9C": list(range(9)),   # teacher
    "3C": [6, 7, 8],         # O1, Oz, O2
    "1C": [7],                # Oz
    # Alternative student montages (montage-sensitivity study, plan N3).
    "3C_PPO": [2, 3, 4],     # PO3, POz, PO4 (shifted to parieto-occipital)
    "3C_POO": [3, 6, 8],     # POz, O1, O2
    "1C_POz": [3],            # POz
    "1C_O1": [6],             # O1
}
CH_NAMES = ["Pz", "PO5", "PO3", "POz", "PO4", "PO6", "O1", "Oz", "O2"]

# Feature levels distilled for each --kd_layers value (see models/cckd_backbones.py).
# "legacy" reproduces the Module 2 / ICERA behaviour exactly: FB-tCNN distills
# [mid736, last], every other backbone distills [last] only.
KD_LAYERS = {
    "last": ["last"],
    "mid": ["mid"],
    "both": ["mid", "last"],
}
CKA_LAYERS = ("mid", "last")
FS_PHYS = 250
GAZE_SHIFT = 0.5
N_CLASSES = 40


def fixed_split(subject_ids, n_train, n_val, n_test, split_seed=2026):
    """ONE fixed (train, val, test) subject split, independent of init seed."""
    subj = np.unique(subject_ids)
    rng = np.random.RandomState(split_seed)
    perm = rng.permutation(subj)
    tr = perm[:n_train]
    va = perm[n_train:n_train + n_val]
    te = perm[n_train + n_val:n_train + n_val + n_test]
    return np.sort(tr), np.sort(va), np.sort(te)


def masks_for(subject_ids, subj_set):
    return np.isin(subject_ids, subj_set)


def prep_input(X, backbone, n_bands=3):
    """X: [N, C, T] (already channel-subset). fbtcnn -> [N, n_bands, C, T]."""
    if backbone == "fbtcnn":
        return apply_filter_bank(X, fs=FS_PHYS, bands=_bands(n_bands))
    return X.astype(np.float32)


def _bands(n_bands):
    full = [(6, 50), (14, 50), (22, 50), (30, 50)]
    return tuple(full[:n_bands])


def subset(X, cfg):
    return X[:, CH_POS[cfg], :]


def _loader(X, y, batch, shuffle):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=2, pin_memory=True)


class PairedDS(Dataset):
    """Returns ((x_student, x_teacher), y) for the SAME trial."""

    def __init__(self, Xs, Xt, y):
        self.Xs = torch.from_numpy(Xs)
        self.Xt = torch.from_numpy(Xt)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (self.Xs[i], self.Xt[i]), self.y[i]


@torch.no_grad()
def eval_per_subject(model, X, y, s, device):
    model.eval()
    xb = torch.from_numpy(X)
    pred = []
    for i in range(0, len(xb), 512):
        pred.append(model(xb[i:i + 512].to(device)).argmax(1).cpu())
    pred = torch.cat(pred).numpy()
    accs = {int(sub): float((pred[s == sub] == y[s == sub]).mean()) for sub in np.unique(s)}
    return accs


def train_ce(model, Xtr, ytr, Xva, yva, sva, cfg, device):
    """Train with cross-entropy only; select best checkpoint by mean per-subject val acc."""
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    crit = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    tr = _loader(Xtr, ytr, cfg["batch_size"], True)
    best, state = -1, None
    for ep in range(cfg["epochs"]):
        model.train()
        for xb, yb in tr:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            if scaler:
                with torch.cuda.amp.autocast():
                    loss = crit(model(xb), yb)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                loss = crit(model(xb), yb); loss.backward(); opt.step()
        sched.step()
        va = np.mean(list(eval_per_subject(model, Xva, yva, sva, device).values()))
        if va > best:
            best, state = va, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    return model, best


def _kd_feats(model, x, kd_layers):
    if kd_layers == "legacy":
        return model(x, return_feat=True)
    z, d = model(x, return_feat="all")
    return z, [d[n] for n in KD_LAYERS[kd_layers]]


def train_kd(student, teacher, Xs_tr, Xt_tr, ytr, Xs_va, yva, sva, cfg, device, mode="relational",
             kd_layers="legacy"):
    """mode: 'logit' (response-based only) | 'relational' (proposed CCKD only)
    | 'hybrid' (response + relational) | 'fitnets' (feature-imitation baseline,
    Romero et al. 2015) | 'attention' (attention-transfer baseline, Zagoruyko
    & Komodakis 2017). The two new baselines are feature-only (no logit term),
    matching how 'relational' is treated, and use their own lambda_sp slot
    value (cfg["lambda_sp"]) since MSE-based losses have a different natural
    scale from the Gram-matrix relational loss.

    kd_layers: 'legacy' | 'last' | 'mid' | 'both' (see KD_LAYERS). With several
    levels, the feature loss is the mean over levels (CCKDLoss)."""
    feature_loss = mode if mode in ("fitnets", "attention") else "relational"
    lkd = 0.0 if mode in ("relational", "fitnets", "attention") else cfg["lambda_kd"]
    lsp = 0.0 if mode == "logit" else cfg["lambda_sp"]
    crit = CCKDLoss(tau=cfg["tau"], lambda_kd=lkd, lambda_sp=lsp, feature_loss=feature_loss)
    student.to(device); teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    opt = optim.Adam(student.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    tr = DataLoader(PairedDS(Xs_tr, Xt_tr, ytr), batch_size=cfg["batch_size"],
                     shuffle=True, num_workers=2, pin_memory=True)
    best, state = -1, None
    for ep in range(cfg["epochs"]):
        student.train()
        agg = {"ce": 0.0, "kd": 0.0, "sp": 0.0}; nb = 0
        for (xs, xt), yb in tr:
            xs, xt, yb = xs.to(device), xt.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            zs, fs = _kd_feats(student, xs, kd_layers)
            with torch.no_grad():
                zt, ft = _kd_feats(teacher, xt, kd_layers)
            loss, parts = crit(zs, fs, zt, ft, yb)
            loss.backward(); opt.step()
            for k in agg:
                agg[k] += parts[k]
            nb += 1
        sched.step()
        va = np.mean(list(eval_per_subject(student, Xs_va, yva, sva, device).values()))
        if va > best:
            best, state = va, {k: v.clone() for k, v in student.state_dict().items()}
    student.load_state_dict(state)
    return student, best


@torch.no_grad()
def extract_named_feats(model, X, device, names=CKA_LAYERS, batch=256):
    """Named feature levels on X (eval mode), concatenated, float64, kept on `device`."""
    model.eval()
    xb = torch.from_numpy(X)
    out = {n: [] for n in names}
    for i in range(0, len(xb), batch):
        _, d = model(xb[i:i + batch].to(device), return_feat="all")
        for n in names:
            out[n].append(d[n].flatten(1).double())
    return {n: torch.cat(v) for n, v in out.items()}


@torch.no_grad()
def linear_cka(X, Y):
    """Linear CKA (Kornblith et al., 2019) in Gram form, X: [n, d1], Y: [n, d2].
    Equal to ||Xc^T Yc||_F^2 / (||Xc^T Xc||_F ||Yc^T Yc||_F) but needs only n x n
    memory, which matters for the wide "mid" levels (up to 24000-d)."""
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    K, L = X @ X.T, Y @ Y.T
    den = torch.linalg.norm(K) * torch.linalg.norm(L)
    if den < 1e-12:
        return float("nan")
    return float((K * L).sum() / den)


def cka_vs_teacher(teacher_feats, student, X, device):
    sf = extract_named_feats(student, X, device, names=tuple(teacher_feats))
    return {n: linear_cka(teacher_feats[n], sf[n]) for n in teacher_feats}


def itr(acc, window_s, n_classes=N_CLASSES, gaze=GAZE_SHIFT):
    """Wolpaw ITR (bits/min)."""
    p = min(max(acc, 1e-6), 1 - 1e-6)
    N = n_classes
    bits = np.log2(N) + p * np.log2(p) + (1 - p) * np.log2((1 - p) / (N - 1))
    T = window_s + gaze
    return float(bits * 60.0 / T)


def summarize(per_subject_list, window_s):
    """per_subject_list: list of {subj: acc} dicts across seeds -> mean/std + ITR."""
    subs = sorted(per_subject_list[0].keys())
    per_sub_mean = {s: np.mean([d[s] for d in per_subject_list]) for s in subs}
    accs = np.array(list(per_sub_mean.values()))
    mean = float(accs.mean())
    return {
        "mean_acc": mean,
        "std_acc": float(accs.std()),
        "itr": itr(mean, window_s),
        "per_subject": {int(s): float(a) for s, a in per_sub_mean.items()},
        "n_test_subjects": len(subs),
        "n_seeds": len(per_subject_list),
    }


def paired_stats(base_list, mode_list):
    """Paired per-subject comparison (removes between-subject variance).
    Wilcoxon signed-rank over subjects (seed-averaged per subject first)."""
    from scipy.stats import wilcoxon
    subs = sorted(base_list[0].keys())
    b = np.array([np.mean([d[s] for d in base_list]) for s in subs])
    m = np.array([np.mean([d[s] for d in mode_list]) for s in subs])
    delta = m - b
    try:
        _, p = wilcoxon(m, b)
    except ValueError:
        p = float("nan")
    return {
        "mean_delta_pp": float(delta.mean() * 100),
        "median_delta_pp": float(np.median(delta) * 100),
        "n_improved": int((delta > 0).sum()),
        "n_subjects": len(subs),
        "wilcoxon_p": float(p),
        "per_subject_delta_pp": {int(s): float(d * 100) for s, d in zip(subs, delta)},
    }
