#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.public_ssvep import PublicSSVEPDataset
from models.fbtcnn import FBT_CNN
from training.filterbank import load_or_create_cache
from utils.config import (
    apply_cli_overrides,
    deep_get,
    load_yaml,
    resolve_project_path,
)

LOG = logging.getLogger("cckd.teacher")


def parser():
    p = argparse.ArgumentParser(
        description="Module 1: FB-tCNN 9-channel teacher baseline"
    )
    p.add_argument("--config", required=True)
    p.add_argument("--data_root")
    p.add_argument("--output_dir")
    p.add_argument("--cache_dir")
    p.add_argument("--n_subjects", type=int)
    p.add_argument("--n_folds", type=int)
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--device")
    p.add_argument("--workers", type=int)
    p.add_argument("--seed", type=int)
    return p


def format_duration(seconds: float) -> str:
    """Convert seconds to HH:MM:SS."""
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_subjects(remaining, frac, seed):
    subjects = np.array(sorted(remaining.tolist()))
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)

    n_val = max(
        1,
        min(
            len(subjects) - 1,
            int(round(len(subjects) * frac)),
        ),
    )

    return subjects[n_val:], subjects[:n_val]


def make_loader(X, y, mask, batch, shuffle, workers):
    idx = np.flatnonzero(mask)

    dataset = TensorDataset(
        torch.from_numpy(
            X[idx].astype(np.float32, copy=False)
        ),
        torch.from_numpy(
            y[idx].astype(np.int64, copy=False)
        ),
    )

    return DataLoader(
        dataset,
        batch_size=batch,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def epoch(model, loader, device, opt=None, scaler=None):
    train = opt is not None
    model.train(train)

    total = 0
    correct = 0
    loss_sum = 0.0

    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            if train:
                opt.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                enabled=scaler is not None,
            ):
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)

            if train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    opt.step()

            loss_sum += loss.item() * yb.numel()
            correct += (logits.argmax(1) == yb).sum().item()
            total += yb.numel()

    return {
        "loss": loss_sum / max(total, 1),
        "acc": correct / max(total, 1),
    }


def main():
    args = parser().parse_args()

    cfg = apply_cli_overrides(
        load_yaml(args.config),
        args,
    )

    out = resolve_project_path(
        deep_get(cfg, "output.output_dir"),
        PROJECT_ROOT,
    )
    cache = resolve_project_path(
        deep_get(cfg, "output.cache_dir"),
        PROJECT_ROOT,
    )

    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                out / "training.log",
                encoding="utf-8",
            ),
        ],
        force=True,
    )

    run_start = time.time()

    seed = int(deep_get(cfg, "runtime.seed", 2026))
    seed_all(seed)

    devname = deep_get(cfg, "runtime.device", "auto")

    if devname == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(devname)

    data_root = Path(
        deep_get(cfg, "dataset.data_root")
    ).expanduser().resolve()

    configured_epochs = int(
        deep_get(cfg, "training.epochs", 150)
    )

    LOG.info("=" * 72)
    LOG.info("CCKD-SSVEP — FB-tCNN TEACHER TRAINING")
    LOG.info("=" * 72)
    LOG.info("config       : %s", Path(args.config).resolve())
    LOG.info("data_root    : %s", data_root)
    LOG.info("output_dir   : %s", out)
    LOG.info("cache_dir    : %s", cache)
    LOG.info("device       : %s", device)
    LOG.info("seed         : %d", seed)
    LOG.info("max_epochs   : %d", configured_epochs)

    LOG.info("")
    LOG.info("Loading dataset...")

    dataset = PublicSSVEPDataset(
        data_root,
        deep_get(cfg, "dataset.name"),
        deep_get(cfg, "dataset.sfreq", 250),
        deep_get(cfg, "dataset.t_start", 0.14),
        deep_get(cfg, "dataset.window_seconds", 1.0),
        deep_get(cfg, "dataset.mat_key", "data"),
        verbose=False,
    )

    X, y, sid = dataset.load_all_subjects(
        int(deep_get(cfg, "dataset.n_subjects"))
    )

    channels = list(
        map(
            int,
            deep_get(
                cfg,
                "dataset.teacher_channel_indices",
            ),
        )
    )

    if max(channels) >= X.shape[1]:
        raise IndexError(
            "teacher channel index exceeds raw channels: "
            f"max={max(channels)}, raw={X.shape[1]}"
        )

    X = X[:, channels, :]

    bands = [
        tuple(map(float, band))
        for band in deep_get(cfg, "filterbank.bands_hz")
    ]

    LOG.info(
        "Raw selected shape : %s",
        X.shape,
    )
    LOG.info(
        "Preparing/loading filter-bank cache..."
    )

    Xfb = load_or_create_cache(
        X,
        cache,
        deep_get(cfg, "dataset.name"),
        float(deep_get(cfg, "dataset.sfreq")),
        bands,
        channels,
        order=int(deep_get(cfg, "filterbank.order", 4)),
        ripple_db=float(deep_get(cfg, "filterbank.ripple_db", 1.0)),
        normalize=bool(deep_get(cfg, "filterbank.normalize_per_trial_channel", True)),
    )

    LOG.info(
        "Dataset ready       : raw_selected=%s | filterbank=%s | "
        "labels=%d | subjects=%d",
        X.shape,
        Xfb.shape,
        len(np.unique(y)),
        len(np.unique(sid)),
    )

    subjects = np.unique(sid)
    n_folds_override = deep_get(cfg, "evaluation.n_folds")

    if n_folds_override is not None:
        subjects = subjects[: int(n_folds_override)]

    total_folds = len(subjects)

    LOG.info(
        "Evaluation folds    : %d",
        total_folds,
    )

    results = []
    params = None
    feature_dim = None
    completed_fold_times = []

    for fold_index, test_subject in enumerate(subjects, 1):
        fold_start = time.time()

        remaining_subjects = np.unique(sid)
        remaining_subjects = remaining_subjects[
            remaining_subjects != test_subject
        ]

        train_subjects, val_subjects = split_subjects(
            remaining_subjects,
            float(
                deep_get(
                    cfg,
                    "training.val_fraction_subjects",
                    0.15,
                )
            ),
            seed + fold_index,
        )

        train_mask = np.isin(sid, train_subjects)
        val_mask = np.isin(sid, val_subjects)
        test_mask = sid == test_subject

        assert not np.any(train_mask & val_mask)
        assert not np.any(train_mask & test_mask)
        assert not np.any(val_mask & test_mask)

        batch_size = int(
            deep_get(cfg, "training.batch_size", 64)
        )
        workers = int(
            deep_get(cfg, "runtime.workers", 2)
        )

        train_loader = make_loader(
            Xfb,
            y,
            train_mask,
            batch_size,
            True,
            workers,
        )
        val_loader = make_loader(
            Xfb,
            y,
            val_mask,
            batch_size,
            False,
            workers,
        )
        test_loader = make_loader(
            Xfb,
            y,
            test_mask,
            batch_size,
            False,
            workers,
        )

        LOG.info("")
        LOG.info("=" * 72)
        LOG.info(
            "FOLD %d / %d | TEST SUBJECT S%02d",
            fold_index,
            total_folds,
            int(test_subject),
        )
        LOG.info("=" * 72)
        LOG.info(
            "train subjects : %d | samples: %d | batches: %d",
            len(train_subjects),
            int(train_mask.sum()),
            len(train_loader),
        )
        LOG.info(
            "val subjects   : %d | samples: %d | batches: %d",
            len(val_subjects),
            int(val_mask.sum()),
            len(val_loader),
        )
        LOG.info(
            "test subjects  : 1 | samples: %d | batches: %d",
            int(test_mask.sum()),
            len(test_loader),
        )
        LOG.info(
            "batch_size     : %d | workers: %d | max_epochs: %d",
            batch_size,
            workers,
            configured_epochs,
        )

        model = FBT_CNN(
            n_channels=Xfb.shape[2],
            n_samples=Xfb.shape[3],
            n_classes=int(
                deep_get(cfg, "dataset.n_classes", 40)
            ),
            n_subbands=Xfb.shape[1],
            branch_filters=int(
                deep_get(cfg, "model.branch_filters", 16)
            ),
            fusion_filters=int(
                deep_get(cfg, "model.fusion_filters", 32)
            ),
            temporal_stride=int(
                deep_get(cfg, "model.temporal_stride", 5)
            ),
            local_kernel=int(
                deep_get(cfg, "model.local_kernel", 5)
            ),
            dropout=float(
                deep_get(cfg, "model.dropout", 0.4)
            ),
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(
                deep_get(
                    cfg,
                    "training.learning_rate",
                    1e-3,
                )
            ),
            weight_decay=float(
                deep_get(
                    cfg,
                    "training.weight_decay",
                    1e-4,
                )
            ),
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=int(
                deep_get(
                    cfg,
                    "training.reduce_lr_patience",
                    7,
                )
            ),
            min_lr=float(
                deep_get(
                    cfg,
                    "training.min_lr",
                    1e-6,
                )
            ),
        )

        amp = bool(
            deep_get(cfg, "training.amp", True)
        )

        scaler = (
            torch.amp.GradScaler("cuda")
            if device.type == "cuda" and amp
            else None
        )

        best_val_acc = -1.0
        best_state = None
        best_epoch = 0
        wait = 0
        history = []
        epoch_times = []

        early_stopping_patience = int(
            deep_get(
                cfg,
                "training.early_stopping_patience",
                20,
            )
        )

        min_delta = float(
            deep_get(
                cfg,
                "training.min_delta",
                1e-4,
            )
        )

        for epoch_index in range(
            1,
            configured_epochs + 1,
        ):
            epoch_start = time.time()

            train_metrics = epoch(
                model,
                train_loader,
                device,
                optimizer,
                scaler,
            )

            val_metrics = epoch(
                model,
                val_loader,
                device,
            )

            scheduler.step(val_metrics["acc"])

            history.append(
                {
                    "epoch": epoch_index,
                    "train": train_metrics,
                    "val": val_metrics,
                }
            )

            improved = (
                val_metrics["acc"]
                > best_val_acc + min_delta
            )

            if improved:
                best_val_acc = val_metrics["acc"]
                best_epoch = epoch_index
                best_state = copy.deepcopy(
                    model.state_dict()
                )
                wait = 0
            else:
                wait += 1

            epoch_seconds = time.time() - epoch_start
            epoch_times.append(epoch_seconds)

            avg_epoch_seconds = float(
                np.mean(epoch_times)
            )
            remaining_epochs = (
                configured_epochs - epoch_index
            )
            fold_eta_seconds = (
                avg_epoch_seconds * remaining_epochs
            )

            current_lr = optimizer.param_groups[0]["lr"]
            elapsed_fold = time.time() - fold_start

            LOG.info(
                "[Fold %02d/%02d | S%02d] "
                "Epoch %03d/%03d | "
                "train loss=%.4f acc=%.4f | "
                "val loss=%.4f acc=%.4f | "
                "best=%.4f@%d | "
                "lr=%.2e | "
                "epoch=%s | elapsed=%s | ETA<=%s | "
                "wait=%d/%d%s",
                fold_index,
                total_folds,
                int(test_subject),
                epoch_index,
                configured_epochs,
                train_metrics["loss"],
                train_metrics["acc"],
                val_metrics["loss"],
                val_metrics["acc"],
                best_val_acc,
                best_epoch,
                current_lr,
                format_duration(epoch_seconds),
                format_duration(elapsed_fold),
                format_duration(fold_eta_seconds),
                wait,
                early_stopping_patience,
                " | NEW BEST" if improved else "",
            )

            if wait >= early_stopping_patience:
                LOG.info(
                    "[Fold %02d/%02d | S%02d] "
                    "EARLY STOPPING at epoch %d; "
                    "best epoch=%d, best val acc=%.4f",
                    fold_index,
                    total_folds,
                    int(test_subject),
                    epoch_index,
                    best_epoch,
                    best_val_acc,
                )
                break

        if best_state is None:
            raise RuntimeError(
                f"No valid model state was saved in fold {fold_index}"
            )

        model.load_state_dict(best_state)

        LOG.info(
            "[Fold %02d/%02d | S%02d] "
            "Evaluating restored best checkpoint from epoch %d...",
            fold_index,
            total_folds,
            int(test_subject),
            best_epoch,
        )

        test_metrics = epoch(
            model,
            test_loader,
            device,
        )

        checkpoint = (
            out / f"fbtcnn_teacher_fold{fold_index:02d}.pt"
        )

        torch.save(
            {
                "model_state": model.state_dict(),
                "config": cfg,
                "fold": fold_index,
                "test_subject": int(test_subject),
                "train_subjects": train_subjects.tolist(),
                "val_subjects": val_subjects.tolist(),
                "best_epoch": best_epoch,
                "best_val_acc": best_val_acc,
                "test": test_metrics,
            },
            checkpoint,
        )

        fold_seconds = time.time() - fold_start
        completed_fold_times.append(fold_seconds)

        row = {
            "fold": fold_index,
            "test_subject": int(test_subject),
            "best_epoch": best_epoch,
            "best_val_acc": float(best_val_acc),
            "test_acc": float(test_metrics["acc"]),
            "test_loss": float(test_metrics["loss"]),
            "time_sec": fold_seconds,
            "checkpoint": str(checkpoint),
        }

        if bool(
            deep_get(
                cfg,
                "evaluation.save_history",
                False,
            )
        ):
            row["history"] = history

        results.append(row)

        params = model.count_params()
        feature_dim = model.get_feature_dim()

        remaining_folds = total_folds - fold_index
        avg_fold_seconds = float(
            np.mean(completed_fold_times)
        )
        overall_eta_seconds = (
            avg_fold_seconds * remaining_folds
        )

        LOG.info(
            "[Fold %02d/%02d | S%02d] DONE | "
            "test loss=%.4f acc=%.4f | "
            "best epoch=%d | "
            "fold time=%s | "
            "overall elapsed=%s | "
            "remaining folds=%d | ETA~%s | "
            "checkpoint=%s",
            fold_index,
            total_folds,
            int(test_subject),
            test_metrics["loss"],
            test_metrics["acc"],
            best_epoch,
            format_duration(fold_seconds),
            format_duration(time.time() - run_start),
            remaining_folds,
            format_duration(overall_eta_seconds),
            checkpoint,
        )

    accuracies = np.array(
        [row["test_acc"] for row in results]
    )

    summary = {
        "project": "CCKD-SSVEP",
        "version": "0.1.0",
        "module": "FB-tCNN teacher baseline",
        "dataset": deep_get(cfg, "dataset.name"),
        "data_root": str(data_root),
        "mean_test_acc": float(accuracies.mean()),
        "std_test_acc": float(accuracies.std()),
        "min_test_acc": float(accuracies.min()),
        "max_test_acc": float(accuracies.max()),
        "n_folds": len(results),
        "params": params,
        "feature_dim": feature_dim,
        "folds": results,
        "config": cfg,
    }

    result_path = out / "teacher_results.json"

    with result_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    LOG.info("")
    LOG.info("=" * 72)
    LOG.info("TRAINING COMPLETED")
    LOG.info("=" * 72)
    LOG.info(
        "dataset       : %s",
        deep_get(cfg, "dataset.name"),
    )
    LOG.info(
        "folds         : %d",
        len(results),
    )
    LOG.info(
        "mean test acc : %.4f",
        accuracies.mean(),
    )
    LOG.info(
        "std test acc  : %.4f",
        accuracies.std(),
    )
    LOG.info(
        "min test acc  : %.4f",
        accuracies.min(),
    )
    LOG.info(
        "max test acc  : %.4f",
        accuracies.max(),
    )
    LOG.info(
        "total time    : %s",
        format_duration(time.time() - run_start),
    )
    LOG.info(
        "results       : %s",
        result_path,
    )


if __name__ == "__main__":
    main()
