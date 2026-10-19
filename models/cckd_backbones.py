"""Backbones for Module 2 (CCKD): teacher(9C) -> student(3C/1C) distillation.

Ported faithfully from the verified ssvep_ensemble/models/backbones.py implementation
(the one whose aggregate numbers match the public CCKD GitHub repo README), so that
numbers produced here are directly comparable/reproducible.

  - FBTCNN : shared-weight filter-bank tCNN. First conv has kernel (C,1) -> collapses
             channels to height 1, so penultimate feature dim is independent of C.
             Exposes TWO feature levels for multi-level relational distillation:
             feat_mid (pre-fusion, ~736-d @ 250 samples) and feat (post-fusion, 32-d).
  - EEGNet : EEGNet-8,2 (depthwise spatial conv also collapses channels to height 1).

Feature API (every backbone):
  - forward(x, return_feat=True)  -> (logits, list)  LEGACY, unchanged: FB-tCNN returns
    [feat_mid(736-d), feat(32-d)], every other backbone returns [feat(penultimate)].
  - forward(x, return_feat="all") -> (logits, dict) with named levels:
      "mid"  : output of the spatial (C,1) convolution block right after its BN +
               nonlinearity, before pooling/dropout (the point where the electrode
               dimension has just been collapsed); defined the same way for all backbones.
      "last" : penultimate feature (the input of the classifier).
    FB-tCNN additionally returns "mid736" (the legacy pre-conv4 fused map).
Computation is identical in both modes, so logits do not depend on return_feat.
"""
from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===================================================================
# Filter bank (Butterworth, matches the verified reference EXACTLY --
# no post-filter normalization). Module 2 trains its own teacher from
# scratch (train_ce), it never loads a Module 1 checkpoint, so there is
# no reason to match Module 1's (unrelated) Chebyshev+normalize
# preprocessing in training/filterbank.py. Keep this faithful to
# ssvep_ensemble/models/backbones.py, whose aggregate numbers are
# independently confirmed by the public CCKD GitHub repo README and by
# the private result JSONs -- do not add normalization here without
# re-validating against that reference.
# ===================================================================
def apply_filter_bank(X: np.ndarray, fs: float = 250,
                       bands: Sequence[Tuple[float, float]] = ((6, 50), (14, 50), (22, 50))) -> np.ndarray:
    """[N, C, T] -> [N, n_bands, C, T] (float32). No normalization."""
    from scipy.signal import butter, filtfilt
    N, C, T = X.shape
    out = np.empty((N, len(bands), C, T), dtype=np.float32)
    nyq = fs / 2.0
    for bi, (lo, hi) in enumerate(bands):
        b, a = butter(4, [lo / nyq, min(hi, nyq - 1) / nyq], btype="band")
        out[:, bi] = filtfilt(b, a, X, axis=-1).astype(np.float32)
    return out


class _SameConvStride(nn.Module):
    """Conv2d over time, TF-'same' padding + stride. Kernel (1, kw)."""

    def __init__(self, cin: int, cout: int, kw: int, stride: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, (1, kw), stride=(1, stride))
        self.kw, self.stride = kw, stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = x.shape[-1]
        out_w = math.ceil(W / self.stride)
        pad = max((out_w - 1) * self.stride + self.kw - W, 0)
        return self.conv(F.pad(x, (pad // 2, pad - pad // 2)))


class FBTCNN(nn.Module):
    """Filter-bank tCNN, shared conv weights across sub-bands.

    Penultimate feature = 32-d (post conv4), independent of n_channels.
    Intermediate feature = pre-conv4 fused map flattened (~736-d @ 250 samples).
    """

    def __init__(self, n_channels: int, n_samples: int, n_classes: int, n_bands: int = 3,
                 out_channel: int = 16, lenth: int = 5, drop: float = 0.4) -> None:
        super().__init__()
        self.n_bands = n_bands
        self.expects_filterbank = n_bands > 1
        self.conv1 = nn.Conv2d(1, out_channel, (n_channels, 1))
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = _SameConvStride(out_channel, out_channel, kw=n_samples, stride=5)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(out_channel, out_channel, (1, lenth))
        self.bn3 = nn.BatchNorm2d(out_channel)
        w2 = math.ceil(n_samples / 5)
        w3 = w2 - lenth + 1
        if w3 < 1:
            raise ValueError(f"n_samples={n_samples} too short for FB-tCNN")
        self.conv4 = nn.Conv2d(out_channel, 32, (1, w3))
        self.bn4 = nn.BatchNorm2d(32)
        self.drop = nn.Dropout(drop)
        self.feat_mid_dim = out_channel * w3
        self.feat_dim = 32
        self.classifier = nn.Linear(32, n_classes)

    def _branch(self, xb: torch.Tensor):
        """Returns (spatial-block output, branch output)."""
        sp = F.elu(self.bn1(self.conv1(xb)))
        x = self.drop(sp)
        x = F.elu(self.bn2(self.conv2(x)))
        x = self.drop(x)
        x = F.elu(self.bn3(self.conv3(x)))
        return sp, x

    def forward(self, x: torch.Tensor, return_feat=False):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if self.n_bands > 1:
            assert x.shape[1] == self.n_bands, (
                f"FBTCNN expects [B,{self.n_bands},C,T], got {tuple(x.shape)} "
                f"(apply_filter_bank first)"
            )
        outs = [self._branch(x[:, b:b + 1]) for b in range(x.shape[1])]
        x = torch.stack([o[1] for o in outs], 0).sum(0)
        feat_mid = x.flatten(1)
        x = self.drop(x)
        x = F.elu(self.bn4(self.conv4(x)))
        feat = x.flatten(1)
        logits = self.classifier(self.drop(feat))
        if return_feat == "all":
            sp = torch.cat([o[0].flatten(1) for o in outs], 1)  # per-band spatial maps
            return logits, {"mid": sp, "mid736": feat_mid, "last": feat}
        return (logits, [feat_mid, feat]) if return_feat else logits


class EEGNet(nn.Module):
    """EEGNet-8,2. Depthwise spatial conv collapses channels -> single feature level."""

    def __init__(self, n_channels: int, n_samples: int, n_classes: int,
                 F1: int = 8, D: int = 2, F2: int = 16, kern_length: int = 64,
                 drop: float = 0.5) -> None:
        super().__init__()
        self.expects_filterbank = False
        self.n_bands = 1
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kern_length), padding=(0, kern_length // 2), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(drop),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(drop),
        )
        with torch.no_grad():
            d = self._features(torch.zeros(1, 1, n_channels, n_samples))
            self.feat_dim = d.shape[1]
        self.classifier = nn.Linear(self.feat_dim, n_classes)

    # Index of the ELU that follows the depthwise spatial conv + BN in block1.
    _SPATIAL_ACT_IDX = 4

    def _features(self, x: torch.Tensor, with_mid: bool = False):
        sp = None
        for i, layer in enumerate(self.block1):
            x = layer(x)
            if i == self._SPATIAL_ACT_IDX:
                sp = x
        x = self.block2(x)
        return (sp.flatten(1), x.flatten(1)) if with_mid else x.flatten(1)

    def forward(self, x: torch.Tensor, return_feat=False):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        sp, feat = self._features(x, with_mid=True)
        logits = self.classifier(feat)
        if return_feat == "all":
            return logits, {"mid": sp, "last": feat}
        return (logits, [feat]) if return_feat else logits


class DeepConvNet(nn.Module):
    """DeepConvNet (Schirrmeister et al., 2017), adapted for a 1.0 s (250-sample)
    window (3 conv-pool blocks instead of the original 4, which would collapse
    the temporal dimension to <=0 at this short window length).

    Unlike EEGNet's depthwise/separable spatial conv, DeepConvNet's spatial
    conv (kernel (C,1)) is an ORDINARY (non-depthwise) convolution -- the same
    channel-collapse mechanism family as FB-tCNN's first conv layer. Used to
    test whether CCKD's benefit is tied to this "ordinary channel-collapsing
    conv" mechanism rather than being specific to FB-tCNN itself.
    """

    def __init__(self, n_channels: int, n_samples: int, n_classes: int,
                 drop: float = 0.5) -> None:
        super().__init__()
        self.expects_filterbank = False
        self.n_bands = 1

        self.temp_conv1 = _SameConvStride(1, 25, kw=10, stride=1)
        self.spat_conv1 = nn.Conv2d(25, 25, (n_channels, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(25)
        self.pool1 = nn.MaxPool2d((1, 3), stride=(1, 3))
        self.drop1 = nn.Dropout(drop)

        self.conv2 = _SameConvStride(25, 50, kw=10, stride=1)
        self.bn2 = nn.BatchNorm2d(50)
        self.pool2 = nn.MaxPool2d((1, 3), stride=(1, 3))
        self.drop2 = nn.Dropout(drop)

        self.conv3 = _SameConvStride(50, 100, kw=10, stride=1)
        self.bn3 = nn.BatchNorm2d(100)
        self.pool3 = nn.MaxPool2d((1, 3), stride=(1, 3))
        self.drop3 = nn.Dropout(drop)

        with torch.no_grad():
            d = self._features(torch.zeros(1, 1, n_channels, n_samples))
            self.feat_dim = d.shape[1]
        self.classifier = nn.Linear(self.feat_dim, n_classes)

    def _features(self, x: torch.Tensor, with_mid: bool = False):
        sp = F.elu(self.bn1(self.spat_conv1(self.temp_conv1(x))))
        x = self.drop1(self.pool1(sp))
        x = self.drop2(self.pool2(F.elu(self.bn2(self.conv2(x)))))
        x = self.drop3(self.pool3(F.elu(self.bn3(self.conv3(x)))))
        return (sp.flatten(1), x.flatten(1)) if with_mid else x.flatten(1)

    def forward(self, x: torch.Tensor, return_feat=False):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        sp, feat = self._features(x, with_mid=True)
        logits = self.classifier(feat)
        if return_feat == "all":
            return logits, {"mid": sp, "last": feat}
        return (logits, [feat]) if return_feat else logits


class CompactCNN(EEGNet):
    """Compact-CNN (Waytowich et al., 2018), for SSVEP classification.

    Same depthwise+separable block structure as EEGNet (temporal conv -> BN ->
    DepthwiseConv2D (C,1) -> BN -> ELU -> AvgPool -> Dropout -> SeparableConv2D
    -> BN -> ELU -> AvgPool -> Dropout), but with the paper's distinguishing
    SSVEP configuration: F1=96, D=1, F2=96 (wide, single spatial filter per
    temporal filter) vs. EEGNet's F1=8, D=2, F2=16. kern_length is reduced
    from the paper's 256 to 64 (matching this project's EEGNet instance) to
    fit the 250-sample (1.0 s) window used throughout this study -- 256 would
    be longer than the window itself.

    Same depthwise/separable channel-collapse mechanism family as EEGNet;
    used as a second member of that architectural group (EEGNet alone is a
    sample of one) when testing the "ordinary vs. depthwise channel collapse"
    mechanism hypothesis for CCKD's backbone-dependent generalization.
    """

    def __init__(self, n_channels: int, n_samples: int, n_classes: int) -> None:
        super().__init__(n_channels, n_samples, n_classes, F1=96, D=1, F2=96, kern_length=64)


class ShallowConvNet(nn.Module):
    """ShallowConvNet (Schirrmeister et al., 2017), the "sibling" architecture
    to DeepConvNet from the same paper -- designed to mimic the Filter Bank
    Common Spatial Pattern (FBCSP) pipeline end-to-end: temporal conv -> spatial
    conv (ORDINARY, non-depthwise, kernel (C,1)) -> BN -> square nonlinearity ->
    average pooling -> log -> classifier.

    Same "ordinary channel-collapsing conv" mechanism family as FB-tCNN /
    DeepConvNet, but with a markedly different nonlinearity/pooling strategy
    (square-log-mean, mimicking band-power features, vs. DeepConvNet's
    ELU + max-pooling VGG-style stack) -- genuine architectural diversity
    within the same mechanism category. Pooling size/stride reduced from the
    paper's defaults (kernel 75, stride 15, designed for multi-second windows)
    to fit the 250-sample (1.0 s) window used throughout this study.
    """

    def __init__(self, n_channels: int, n_samples: int, n_classes: int,
                 F1: int = 40, pool_size: int = 35, pool_stride: int = 7,
                 drop: float = 0.5) -> None:
        super().__init__()
        self.expects_filterbank = False
        self.n_bands = 1
        self.temp_conv = _SameConvStride(1, F1, kw=25, stride=1)
        self.spat_conv = nn.Conv2d(F1, F1, (n_channels, 1), bias=False)
        self.bn = nn.BatchNorm2d(F1)
        self.pool = nn.AvgPool2d((1, pool_size), stride=(1, pool_stride))
        self.drop = nn.Dropout(drop)
        with torch.no_grad():
            d = self._features(torch.zeros(1, 1, n_channels, n_samples))
            self.feat_dim = d.shape[1]
        self.classifier = nn.Linear(self.feat_dim, n_classes)

    def _features(self, x: torch.Tensor, with_mid: bool = False):
        x = self.bn(self.spat_conv(self.temp_conv(x)))
        sp = x.pow(2)  # square is this backbone's nonlinearity
        x = self.pool(sp)
        x = torch.log(torch.clamp(x, min=1e-6))
        x = self.drop(x)
        return (sp.flatten(1), x.flatten(1)) if with_mid else x.flatten(1)

    def forward(self, x: torch.Tensor, return_feat=False):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        sp, feat = self._features(x, with_mid=True)
        logits = self.classifier(feat)
        if return_feat == "all":
            return logits, {"mid": sp, "last": feat}
        return (logits, [feat]) if return_feat else logits


BACKBONE_REGISTRY = {
    "fbtcnn": lambda C, T, n, nb: FBTCNN(C, T, n, n_bands=nb),
    "tcnn": lambda C, T, n, nb: FBTCNN(C, T, n, n_bands=1),
    "eegnet": lambda C, T, n, nb: EEGNet(C, T, n),
    "deepconvnet": lambda C, T, n, nb: DeepConvNet(C, T, n),
    "compactcnn": lambda C, T, n, nb: CompactCNN(C, T, n),
    "shallowconvnet": lambda C, T, n, nb: ShallowConvNet(C, T, n),
}


def build_backbone(name: str, n_channels: int, n_samples: int, n_classes: int, n_bands: int = 3):
    name = name.lower()
    if name not in BACKBONE_REGISTRY:
        raise ValueError(f"backbone '{name}' unknown. Choose from: {list(BACKBONE_REGISTRY)}")
    return BACKBONE_REGISTRY[name](n_channels, n_samples, n_classes, n_bands)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
