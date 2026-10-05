"""Built-in loaders for the public Benchmark and BETA SSVEP datasets.

Expected public MAT tensor is normally [channels, time, targets, blocks].
The loader also detects common axis permutations and supports MATLAB v7.3 files.
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Tuple
import numpy as np


def _h5_to_python(obj):
    import h5py
    if isinstance(obj, h5py.Dataset):
        return np.array(obj)
    return {k: _h5_to_python(v) for k, v in obj.items()}


def _read_mat(path: Path) -> dict:
    try:
        from scipy.io import loadmat
        return loadmat(path)
    except NotImplementedError:
        import h5py
        with h5py.File(path, "r") as f:
            return {k: _h5_to_python(f[k]) for k in f.keys()}


def _collect_4d(obj: dict) -> list:
    found = []
    for k, v in obj.items():
        if k.startswith("__"):
            continue
        if isinstance(v, dict):
            found.extend(_collect_4d(v))
        elif isinstance(v, np.ndarray) and v.ndim == 4:
            found.append((k, np.asarray(v)))
    return found


def _find_subject_file(root: Path, subject_id: int) -> Path:
    candidates = [
        root / f"S{subject_id}.mat", root / f"S{subject_id:02d}.mat",
        root / f"s{subject_id}.mat", root / f"s{subject_id:02d}.mat",
    ]
    for p in candidates:
        if p.is_file():
            return p
    patterns = [f"**/S{subject_id}.mat", f"**/S{subject_id:02d}.mat",
                f"**/s{subject_id}.mat", f"**/s{subject_id:02d}.mat"]
    hits = []
    for pat in patterns:
        hits.extend(root.glob(pat))
    hits = sorted(set(hits))
    if not hits:
        raise FileNotFoundError(f"No MAT file found for subject {subject_id} under {root}")
    return hits[0]


def _find_4d_in_struct(arr: np.ndarray, _depth: int = 0) -> np.ndarray | None:
    """Recursively search a MATLAB struct (loaded by scipy.io.loadmat) for a 4-D numeric array."""
    if _depth > 4 or arr.size == 0:
        return None
    scalar = arr.reshape(-1)[0]
    names = getattr(scalar.dtype if hasattr(scalar, "dtype") else arr.dtype, "names", None)
    if not names:
        return None
    for name in names:
        field = scalar[name] if hasattr(scalar, "dtype") else arr[0, 0][name]
        field = np.asarray(field)
        if field.dtype == object:
            if field.ndim == 0:
                field = field.reshape(1, 1)
            found = _find_4d_in_struct(field, _depth + 1)
            if found is not None:
                return found
        elif field.ndim == 4:
            return field
    return None


def _select_4d_array(mat: dict, preferred_key: str = "data") -> np.ndarray:
    if preferred_key in mat and isinstance(mat[preferred_key], np.ndarray) and mat[preferred_key].ndim == 4:
        return np.asarray(mat[preferred_key])
    if preferred_key in mat and isinstance(mat[preferred_key], np.ndarray) and mat[preferred_key].dtype.names:
        found = _find_4d_in_struct(mat[preferred_key])
        if found is not None:
            return found
    if preferred_key in mat and isinstance(mat[preferred_key], dict):
        nested = _collect_4d(mat[preferred_key])
        if nested:
            return max(nested, key=lambda kv: kv[1].size)[1]
    arrays = _collect_4d(mat)
    if not arrays:
        shapes = {k: getattr(v, "shape", None) for k, v in mat.items() if not k.startswith("__")}
        raise ValueError(f"No 4-D EEG tensor found. MAT variables: {shapes}")
    arrays.sort(key=lambda kv: kv[1].size, reverse=True)
    return arrays[0][1]


def _canonicalize(x: np.ndarray, n_channels: int = 64, n_targets: int = 40,
                  expected_blocks: int | None = None) -> np.ndarray:
    """Return tensor as [channels, time, targets, blocks]."""
    shape = x.shape
    axes = list(range(4))
    ch_axes = [i for i, n in enumerate(shape) if n == n_channels]
    target_axes = [i for i, n in enumerate(shape) if n == n_targets]
    if not ch_axes or not target_axes:
        raise ValueError(f"Cannot identify channel/target axes in shape {shape}")
    ch_axis = ch_axes[0]
    target_axis = next((i for i in target_axes if i != ch_axis), target_axes[0])
    remaining = [i for i in axes if i not in (ch_axis, target_axis)]
    if expected_blocks is not None and any(shape[i] == expected_blocks for i in remaining):
        block_axis = next(i for i in remaining if shape[i] == expected_blocks)
    else:
        block_axis = min(remaining, key=lambda i: shape[i])
    time_axis = next(i for i in remaining if i != block_axis)
    out = np.transpose(x, (ch_axis, time_axis, target_axis, block_axis))
    return np.asarray(out, dtype=np.float32)


class PublicSSVEPDataset:
    # Both Benchmark (Wang et al. 2017) and BETA (Liu et al. 2020) epoch each
    # trial starting 0.5s BEFORE stimulus onset (confirmed for BETA in its
    # official Description.pdf; Benchmark follows the same 0.5s pre-stim +
    # 5s stim + 0.5s post convention throughout the SSVEP literature).
    # t_start is a visual-latency offset measured from the ONSET, not from
    # the start of the raw trial array, so it must be added on top of this.
    PRE_STIM_SECONDS = 0.5

    def __init__(self, data_root: str | Path, dataset: str, sfreq: float = 250.0,
                 t_start: float = 0.14, window_seconds: float = 1.0,
                 mat_key: str = "data", verbose: bool = True):
        self.root = Path(data_root).expanduser().resolve()
        self.dataset = dataset.lower()
        self.sfreq = float(sfreq)
        self.t_start = float(t_start)
        self.window_seconds = float(window_seconds)
        self.mat_key = mat_key
        self.verbose = verbose
        if not self.root.is_dir():
            raise FileNotFoundError(f"data_root does not exist: {self.root}")
        if self.dataset not in {"benchmark", "beta"}:
            raise ValueError("dataset must be 'benchmark' or 'beta'")

    @property
    def expected_blocks(self) -> int:
        return 6 if self.dataset == "benchmark" else 4

    def load_subject(self, subject_id: int) -> Tuple[np.ndarray, np.ndarray]:
        path = _find_subject_file(self.root, subject_id)
        mat = _read_mat(path)
        raw = _select_4d_array(mat, self.mat_key)
        eeg = _canonicalize(raw, expected_blocks=self.expected_blocks)
        start = int(round((self.PRE_STIM_SECONDS + self.t_start) * self.sfreq))
        n_samples = int(round(self.window_seconds * self.sfreq))
        stop = start + n_samples
        if stop > eeg.shape[1]:
            raise ValueError(
                f"Requested samples [{start}:{stop}] exceed trial length {eeg.shape[1]} in {path}"
            )
        seg = eeg[:, start:stop, :, :]
        # [C,T,K,B] -> [K,B,C,T] -> [trials,C,T]
        X = np.transpose(seg, (2, 3, 0, 1)).reshape(-1, eeg.shape[0], n_samples)
        y = np.repeat(np.arange(eeg.shape[2], dtype=np.int64), eeg.shape[3])
        if self.verbose:
            print(f"S{subject_id:02d}: {path.name} raw={tuple(raw.shape)} canonical={tuple(eeg.shape)} "
                  f"window=[{start/self.sfreq:.2f}s,{stop/self.sfreq:.2f}s) X={tuple(X.shape)}")
        return X.astype(np.float32, copy=False), y

    def load_all_subjects(self, n_subjects: int):
        Xs, ys, sids = [], [], []
        for sid in range(1, int(n_subjects) + 1):
            X, y = self.load_subject(sid)
            Xs.append(X); ys.append(y); sids.append(np.full(len(y), sid, dtype=np.int64))
        return np.concatenate(Xs), np.concatenate(ys), np.concatenate(sids)
