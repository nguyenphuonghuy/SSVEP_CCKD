"""
cckd_pipeline.py
===================================================================
Loi (engine) cho thi nghiem Cross-Channel KD. Plug thang vao:
    data.benchmark_loader.BenchmarkDataset   (tra ve [N, 9, T], thu tu kenh:
        Pz,PO5,PO3,POz,PO4,PO6,O1,Oz,O2)
    models.backbones.build_backbone / apply_filter_bank
    models.kd_losses.CCKDLoss

QUYET DINH THIET KE da chot:
  - Chia subject-independent CO DINH 1 lan, test tren NHIEU subject hon.
    Bien thien lay tu nhieu seed KHOI TAO (khong doi phep chia).
  - KD = privileged-information distillation: teacher 9C (dac quyen) -> student 1C/3C.
  - Feature loss = similarity-preserving (relational), khong phai cosine cung.
"""
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset

from models.backbones import build_backbone, apply_filter_bank
from models.kd_losses import CCKDLoss

# --- Vi tri kenh TRONG mang 9 kenh loader tra ve (Pz,PO5,PO3,POz,PO4,PO6,O1,Oz,O2) ---
CH_POS = {
    "9C": list(range(9)),   # teacher
    "3C": [6, 7, 8],        # O1, Oz, O2
    "1C": [7],              # Oz   <<< XAC NHAN vi tri 7 == Oz voi montage cua ban
}
FS_PHYS = 250               # Hz vat ly (cho duration & ITR)
GAZE_SHIFT = 0.5            # s, cong vao T khi tinh ITR
N_CLASSES = 40


# ===================================================================
# Chia subject CO DINH (1 phep chia duy nhat)
# ===================================================================
def fixed_split(subject_ids, n_train, n_val, n_test, split_seed=2026):
    """Tra ve (train_subj, val_subj, test_subj). Phep chia CO DINH theo split_seed,
    KHONG doi giua cac model / init-seed."""
    subj = np.unique(subject_ids)
    rng = np.random.RandomState(split_seed)
    perm = rng.permutation(subj)
    tr = perm[:n_train]
    va = perm[n_train:n_train + n_val]
    te = perm[n_train + n_val:n_train + n_val + n_test]
    return np.sort(tr), np.sort(va), np.sort(te)


def masks_for(subject_ids, subj_set):
    return np.isin(subject_ids, subj_set)


# ===================================================================
# Chuan bi input theo backbone (filter bank cho FB-tCNN)
# ===================================================================
def prep_input(X, backbone, n_bands=3):
    """X: [N, C, T] (da subset kenh). Voi fbtcnn -> [N, n_bands, C, T] (filter bank).
    tcnn/eegnet -> giu [N, C, T]."""
    if backbone == "fbtcnn":
        return apply_filter_bank(X, fs=FS_PHYS, bands=_bands(n_bands))
    return X.astype(np.float32)


def _bands(n_bands):
    full = [(6, 50), (14, 50), (22, 50), (30, 50)]
    return tuple(full[:n_bands])


def subset(X, cfg):
    return X[:, CH_POS[cfg], :]


# ===================================================================
# DataLoaders
# ===================================================================
def _loader(X, y, batch, shuffle):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=2, pin_memory=True)


class PairedDS(Dataset):
    """Tra ve ((x_student, x_teacher), y) — CUNG trial."""
    def __init__(self, Xs, Xt, y):
        self.Xs = torch.from_numpy(Xs); self.Xt = torch.from_numpy(Xt); self.y = torch.from_numpy(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return (self.Xs[i], self.Xt[i]), self.y[i]


# ===================================================================
# Eval theo subject
# ===================================================================
@torch.no_grad()
def eval_per_subject(model, X, y, s, device):
    model.eval()
    xb = torch.from_numpy(X).to(device)
    pred = []
    for i in range(0, len(xb), 512):
        pred.append(model(xb[i:i + 512]).argmax(1).cpu())
    pred = torch.cat(pred).numpy()
    accs = {int(sub): float((pred[s == sub] == y[s == sub]).mean()) for sub in np.unique(s)}
    return accs


# ===================================================================
# Train baseline (CE)  — chon best theo val subject-acc
# ===================================================================
def train_ce(model, Xtr, ytr, Xva, yva, sva, cfg, device):
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


# ===================================================================
# Train student voi KD (teacher dong bang) — privileged-information
# ===================================================================
def train_kd(student, teacher, Xs_tr, Xt_tr, ytr, Xs_va, yva, sva, cfg, device,
             mode="hybrid"):
    """mode: 'logit' (chi logit KD) | 'sp' (chi similarity-preserving) | 'hybrid' (ca hai)."""
    lkd = 0.0 if mode == "sp" else cfg["lambda_kd"]
    lsp = 0.0 if mode == "logit" else cfg["lambda_sp"]
    crit = CCKDLoss(tau=cfg["tau"], lambda_kd=lkd, lambda_sp=lsp)
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
            zs, fs = student(xs, return_feat=True)
            with torch.no_grad():
                zt, ft = teacher(xt, return_feat=True)
            loss, parts = crit(zs, fs, zt, ft, yb)
            loss.backward(); opt.step()
            for k in agg:
                agg[k] += parts[k]
            nb += 1
        sched.step()
        va = np.mean(list(eval_per_subject(student, Xs_va, yva, sva, device).values()))
        if va > best:
            best, state = va, {k: v.clone() for k, v in student.state_dict().items()}
        # In DO LON CO TRONG SO cua tung so hang (de so voi CE) o epoch dau va cuoi
        if ep in (0, cfg["epochs"] - 1):
            print(f"      [{mode}] ep{ep}: CE={agg['ce']/nb:.3f} | "
                  f"lam_kd*KD={lkd * agg['kd']/nb:.3f} | lam_sp*SP={lsp * agg['sp']/nb:.3f}",
                  flush=True)
    student.load_state_dict(state)
    return student, best


# ===================================================================
# ITR (Wolpaw), T = cua so vat ly + gaze shift
# ===================================================================
def itr(acc, window_s, n_classes=N_CLASSES, gaze=GAZE_SHIFT):
    p = min(max(acc, 1e-6), 1 - 1e-6)
    N = n_classes
    bits = np.log2(N) + p * np.log2(p) + (1 - p) * np.log2((1 - p) / (N - 1))
    T = window_s + gaze
    return float(bits * 60.0 / T)


def summarize(per_subject_list, window_s):
    """per_subject_list: list cac dict {subj: acc} qua nhieu seed. Tra ve mean/std + ITR."""
    subs = sorted(per_subject_list[0].keys())
    # trung binh theo seed cho tung subject
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
    """So SANH THEO CAP tren tung subject (huy bien thien giua-subject).
    base_list/mode_list: list per-seed dict {subj: acc}. Wilcoxon signed-rank tren cac subject."""
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
