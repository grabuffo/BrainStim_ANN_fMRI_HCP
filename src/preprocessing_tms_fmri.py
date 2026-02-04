
"""
preprocessing_tms_fmri.py

Utilities to preprocess parcel-level TMS-fMRI time series and prepare ANN inputs/targets.

Designed to mirror the lightweight API of `preprocessing_hcp.py`, but with:
- TR default = 2.4 s (TMS-fMRI effective TR)
- optional initial time-point trimming (default: drop first 30 TRs *per run*)
- band-pass filtering in the same band as your HCP pipeline (0.008–0.08 Hz, 2nd order Butterworth)
- standardization (z-score per region)
- creation of sliding-window Inputs/Targets for multi-step-to-one-step prediction:
    X[t] = concat( signals[t : t+S] )  -> shape (S*N,)
    Y[t] = signals[t+S]               -> shape (N,)

Shapes:
- signals: (T, N)
- inputs:  (T-S, S*N)
- targets: (T-S, N)
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, detrend
from scipy import stats
from typing import Iterable, List, Tuple

# ----------------------------
# Defaults (match HCP pipeline)
# ----------------------------
TR = 2.4  # seconds (effective TR for TMS-fMRI)
LOW_HZ, HIGH_HZ = 0.008, 0.08
ORDER = 2
DROP_INITIAL = 30  # drop first 30 time points per run


def bandpass_filter_timeseries(
    X: np.ndarray,
    tr: float = TR,
    low: float = LOW_HZ,
    high: float = HIGH_HZ,
    order: int = ORDER,
    zscore: bool = True,
) -> np.ndarray:
    """
    Band-pass filter + detrend (linear) time series across time axis.

    Args:
        X: (T, N) array, timepoints × regions
        tr: repetition time in seconds
        low/high: band edges in Hz
        order: Butterworth order
        zscore: if True, z-score per region after filtering (ddof=0 via scipy.stats.zscore)

    Returns:
        Xf: (T, N) filtered (and optionally standardized) array
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (T,N). Got shape {X.shape}")

    fs = 1.0 / float(tr)
    nyq = fs / 2.0
    low_n = low / nyq
    high_n = high / nyq
    if not (0 < low_n < high_n < 1):
        raise ValueError(f"Bad band ({low}, {high}) for TR={tr}s (Nyquist={nyq} Hz).")

    b, a = butter(order, [low_n, high_n], btype="bandpass")

    # Detrend reduces edge/low-frequency artifacts
    Xd = detrend(X, axis=0, type="linear")

    # filtfilt pad length safeguard for short T
    padlen_default = 3 * max(len(a), len(b))
    padlen = min(padlen_default, max(0, Xd.shape[0] - 1))

    Xf = filtfilt(b, a, Xd, axis=0, padlen=padlen)

    if zscore:
        # keep NaNs (if any) stable: stats.zscore will propagate NaNs
        Xf = stats.zscore(Xf, axis=0)

    return Xf


def drop_initial_timepoints(X: np.ndarray, n_drop: int = DROP_INITIAL) -> np.ndarray:
    """
    Drop the first n_drop time points (rows).

    Args:
        X: (T,N)
        n_drop: int

    Returns:
        X_trim: (max(T-n_drop,0), N)
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (T,N). Got shape {X.shape}")
    if n_drop <= 0:
        return X
    if X.shape[0] <= n_drop:
        return X[0:0, :]
    return X[n_drop:, :]


def preprocess_run(
    X: np.ndarray,
    tr: float = TR,
    n_drop: int = DROP_INITIAL,
    low: float = LOW_HZ,
    high: float = HIGH_HZ,
    order: int = ORDER,
    zscore: bool = True,
) -> np.ndarray:
    """
    Preprocess a single run: drop initial points -> bandpass+detrend -> optional z-score.

    Args:
        X: (T,N) raw signals
        tr: TR in seconds
        n_drop: drop first n_drop TRs
        low/high/order: filter settings
        zscore: whether to standardize per region

    Returns:
        Xp: (T',N) preprocessed
    """
    Xt = drop_initial_timepoints(X, n_drop=n_drop)
    if Xt.shape[0] < 5:
        # too short to filter sensibly; return empty or the trimmed array
        return Xt.astype(np.float32, copy=False)
    Xp = bandpass_filter_timeseries(Xt, tr=tr, low=low, high=high, order=order, zscore=zscore)
    return Xp.astype(np.float32, copy=False)


def concat_runs(
    runs: Iterable[np.ndarray],
    tr: float = TR,
    n_drop: int = DROP_INITIAL,
    low: float = LOW_HZ,
    high: float = HIGH_HZ,
    order: int = ORDER,
    zscore: bool = True,
    filter_each_run: bool = True,
) -> np.ndarray:
    """
    Concatenate multiple runs for the SAME subject safely.

    Default behavior filters EACH run separately (recommended) and then concatenates.
    This avoids edge artifacts bleeding across run boundaries.

    Args:
        runs: iterable of (T_i,N) arrays
        filter_each_run: if True, preprocess each run independently then concat;
                         if False, drop+concat first then filter once (usually worse).

    Returns:
        signals: (sum_i T_i', N) concatenated preprocessed signals
    """
    runs = [np.asarray(r) for r in runs if r is not None and np.asarray(r).size > 0]
    if not runs:
        return np.zeros((0, 0), dtype=np.float32)

    # check consistent N
    Ns = [r.shape[1] for r in runs if r.ndim == 2]
    if len(set(Ns)) != 1:
        raise ValueError(f"All runs must have same #regions. Got {Ns}")

    if filter_each_run:
        proc = [
            preprocess_run(r, tr=tr, n_drop=n_drop, low=low, high=high, order=order, zscore=zscore)
            for r in runs
        ]
        proc = [p for p in proc if p.size > 0 and p.shape[0] > 0]
        if not proc:
            return np.zeros((0, Ns[0]), dtype=np.float32)
        return np.concatenate(proc, axis=0).astype(np.float32, copy=False)

    # alternative: concat then filter once
    concat = np.concatenate([drop_initial_timepoints(r, n_drop=n_drop) for r in runs], axis=0)
    if concat.shape[0] < 5:
        return concat.astype(np.float32, copy=False)
    return bandpass_filter_timeseries(concat, tr=tr, low=low, high=high, order=order, zscore=zscore).astype(np.float32, copy=False)


def make_inputs_targets(signals: np.ndarray, steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build sliding-window Inputs/Targets for one-step prediction.

    Args:
        signals: (T,N) array
        steps: S (window length)

    Returns:
        inputs:  (T-S, S*N)
        targets: (T-S, N)
    """
    X = np.asarray(signals, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"signals must be 2D (T,N). Got shape {X.shape}")
    T, N = X.shape
    S = int(steps)
    if S <= 0:
        raise ValueError("steps must be >= 1")
    if T <= S:
        return np.zeros((0, S * N), dtype=np.float32), np.zeros((0, N), dtype=np.float32)

    # Build windows: for each t in [0, T-S-1], input = X[t:t+S].flatten()
    # This is memory-safe enough for typical fMRI lengths; for huge arrays you could stride.
    inputs = np.empty((T - S, S * N), dtype=np.float32)
    for t in range(T - S):
        inputs[t] = X[t : t + S].reshape(-1)
    targets = X[S:, :].copy()
    return inputs, targets


def split_last_fraction(signals: np.ndarray, test_fraction: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split a single subject's signals into train/test by taking the LAST fraction as test.

    Args:
        signals: (T,N)
        test_fraction: e.g. 0.1

    Returns:
        train_signals, test_signals
    """
    X = np.asarray(signals)
    if X.ndim != 2:
        raise ValueError(f"signals must be 2D (T,N). Got shape {X.shape}")
    if not (0.0 < test_fraction < 1.0):
        raise ValueError("test_fraction must be between 0 and 1")

    T = X.shape[0]
    cut = int(np.floor((1.0 - test_fraction) * T))
    cut = max(1, min(T - 1, cut))  # ensure both non-empty
    return X[:cut].astype(np.float32, copy=False), X[cut:].astype(np.float32, copy=False)
