"""
metrics.py — Baseline & stimulation-effect metrics and utilities.

Shapes:
- W: number of windows (time points after windowing)
- N: number of regions

Inputs typically come from:
- xt_baseline: (W, N) — baseline state at time t (last step of each window)
- xt_next:     (W, N) — unperturbed next state at t+Δt
- xt_stim_next:(N, W, N) — for each target j, stimulated next state at t+Δt

This module provides:
- Baseline metrics (operate on x_t: (N,))
- Effect metrics:
    * "one-vector" metrics operate on a single vector v: (N,)
      (use v = e = x_stim - x_next OR v = x_stim)
    * "two-vector" metrics operate on a pair (a, b), each (N,)
      (use (x_stim, x_next) OR (e, x_next))
- A high-level function to compute a tidy DataFrame over all (w, tar):
    compute_metrics_table(...)

Author: Rina_Zelmann_AB_model helpers
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Callable, Dict, Tuple, Optional

try:
    from scipy.stats import entropy, skew, kurtosis, zscore
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

_EPS = 1e-12


# =============================================================================
# Helpers
# =============================================================================

def _safe_entropy_from_hist(x: np.ndarray, bins: int = 20) -> float:
    """Shannon entropy from histogram (handles empty bins)."""
    hist, _ = np.histogram(x, bins=bins, density=True)
    hist = hist[hist > 0]
    if hist.size == 0:
        return 0.0
    if _HAVE_SCIPY:
        return float(entropy(hist))
    # Fallback: natural log base
    p = hist / (hist.sum() + _EPS)
    return float(-np.sum(p * np.log(p + _EPS)))


def _gini_from_abs(x: np.ndarray) -> float:
    """Gini index of absolute values (0=equal, 1=unequal)."""
    a = np.sort(np.abs(x).astype(float) + _EPS)
    n = a.size
    cum = np.cumsum(a)
    return float((n + 1 - 2.0 * np.sum(cum) / cum[-1]) / n)


def _participation_ratio(x: np.ndarray) -> float:
    """(sum x^2)^2 / sum x^4; larger -> more distributed pattern."""
    num = float(np.sum(x**2)**2)
    den = float(np.sum(x**4) + _EPS)
    return num / den


def _l2(v: np.ndarray) -> float:
    return float(np.sqrt(np.sum(v * v)))


def _mean_abs(v: np.ndarray) -> float:
    return float(np.mean(np.abs(v)))


def _max_abs(v: np.ndarray) -> float:
    return float(np.max(np.abs(v)))


def _prob_from_abs(v: np.ndarray) -> np.ndarray:
    p = np.abs(v).astype(float)
    s = p.sum()
    return p / (s + _EPS)


# =============================================================================
# Baseline metrics (x_t -> scalar). x_t has shape (N,)
# Some metrics use context for z-scoring across time.
# =============================================================================

class BaselineContext:
    """Holds precomputed context for baseline metrics that need across-time stats."""
    def __init__(self, xt_baseline: np.ndarray):
        """
        xt_baseline: (W, N)
        Precompute per-region mean/std across W for z-scoring.
        """
        self.W, self.N = xt_baseline.shape
        self.mean = xt_baseline.mean(axis=0)     # (N,)
        self.std  = xt_baseline.std(axis=0, ddof=1) + _EPS  # (N,)
        # z-scored across time for convenience
        if _HAVE_SCIPY:
            self.Z = zscore(xt_baseline, axis=0, ddof=1)
        else:
            self.Z = (xt_baseline - self.mean) / self.std


def baseline_mean(x_t: np.ndarray, ctx: Optional[BaselineContext] = None) -> float:
    return float(np.mean(x_t))


def baseline_variance(x_t: np.ndarray, ctx: Optional[BaselineContext] = None) -> float:
    return float(np.var(x_t))


def baseline_std(x_t: np.ndarray, ctx: Optional[BaselineContext] = None) -> float:
    return float(np.std(x_t))


def baseline_entropy(x_t: np.ndarray, ctx: Optional[BaselineContext] = None, bins: int = 20) -> float:
    return _safe_entropy_from_hist(x_t, bins=bins)


def baseline_l2norm(x_t: np.ndarray, ctx: Optional[BaselineContext] = None) -> float:
    return _l2(x_t)


def baseline_mean_abs(x_t: np.ndarray, ctx: Optional[BaselineContext] = None) -> float:
    return _mean_abs(x_t)


def baseline_max_abs(x_t: np.ndarray, ctx: Optional[BaselineContext] = None) -> float:
    return _max_abs(x_t)


def baseline_gini(x_t: np.ndarray, ctx: Optional[BaselineContext] = None) -> float:
    return _gini_from_abs(x_t)


def baseline_participation_ratio(x_t: np.ndarray, ctx: Optional[BaselineContext] = None) -> float:
    return _participation_ratio(x_t)


def baseline_skewness(x_t: np.ndarray, ctx: Optional[BaselineContext] = None) -> float:
    if _HAVE_SCIPY:
        return float(skew(x_t))
    # simple fallback (biased for small N)
    m = float(np.mean(x_t))
    s = float(np.std(x_t) + _EPS)
    z3 = np.mean(((x_t - m) / s) ** 3)
    return float(z3)


def baseline_kurtosis(x_t: np.ndarray, ctx: Optional[BaselineContext] = None) -> float:
    if _HAVE_SCIPY:
        return float(kurtosis(x_t))
    m = float(np.mean(x_t))
    s = float(np.std(x_t) + _EPS)
    z4 = np.mean(((x_t - m) / s) ** 4) - 3.0
    return float(z4)


def baseline_mean_abs_z(x_t: np.ndarray, ctx: Optional[BaselineContext] = None) -> float:
    """
    Mean absolute z-scored activity across regions at time t.
    Requires ctx with across-time z-scoring.
    """
    if ctx is None:
        raise ValueError("baseline_mean_abs_z requires a BaselineContext (pass xt_baseline to compute_metrics_table).")
    # Find which row x_t corresponds to via pointer equality fallback — not reliable.
    # Instead, caller should pass the exact z-row. Here we recompute z_t from ctx stats:
    z_t = (x_t - ctx.mean) / ctx.std
    return float(np.mean(np.abs(z_t)))


# Registry of baseline metrics by name
BASELINE_METRICS: Dict[str, Callable[..., float]] = {
    "mean": baseline_mean,
    "variance": baseline_variance,
    "std": baseline_std,
    "entropy": baseline_entropy,
    "l2norm": baseline_l2norm,
    "mean_abs": baseline_mean_abs,
    "max_abs": baseline_max_abs,
    "gini": baseline_gini,
    "participation_ratio": baseline_participation_ratio,
    "skewness": baseline_skewness,
    "kurtosis": baseline_kurtosis,
    "mean_abs_z": baseline_mean_abs_z,
}


def list_baseline_metrics() -> Tuple[str, ...]:
    return tuple(BASELINE_METRICS.keys())


# =============================================================================
# Effect metrics
# One-vector metrics: f(v) where v is either e = x_stim - x_next or v = x_stim
# Two-vector metrics: f(a, b) where (a, b) is either (x_stim, x_next) or (e, x_next)
# =============================================================================

# ---- one-vector metrics: f(v) -> scalar ----
def effect_l2norm(v: np.ndarray) -> float:
    return _l2(v)

def effect_mean_abs(v: np.ndarray) -> float:
    return _mean_abs(v)

def effect_max_abs(v: np.ndarray) -> float:
    return _max_abs(v)

def effect_entropy_magnitude(v: np.ndarray, bins: int = 20) -> float:
    return _safe_entropy_from_hist(np.abs(v), bins=bins)

def effect_gini(v: np.ndarray) -> float:
    return _gini_from_abs(v)

def effect_participation_ratio(v: np.ndarray) -> float:
    return _participation_ratio(v)

def effect_fraction_suprathreshold(v: np.ndarray, k: float = 1.0) -> float:
    mag = np.abs(v)
    thr = k * np.std(mag) + _EPS
    return float(np.mean(mag > thr))

def effect_pos_neg_balance(v: np.ndarray) -> float:
    pos = v[v > 0].mean() if np.any(v > 0) else 0.0
    neg = v[v < 0].mean() if np.any(v < 0) else 0.0
    return float(pos - abs(neg))


# ---- two-vector metrics: f(a, b) -> scalar ----
def state_distance_l2(a: np.ndarray, b: np.ndarray) -> float:
    return _l2(a - b)

def state_mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))

def effect_alignment_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between a and b."""
    na = _l2(a); nb = _l2(b)
    if na * nb < _EPS:
        return 0.0
    return float(np.dot(a.ravel(), b.ravel()) / (na * nb))

def effect_js_divergence(a: np.ndarray, b: np.ndarray) -> float:
    """Jensen–Shannon divergence between |a| and |b| normalized to probabilities."""
    p = _prob_from_abs(a); q = _prob_from_abs(b)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * (np.log(p + _EPS) - np.log(m + _EPS)))
    kl_qm = np.sum(q * (np.log(q + _EPS) - np.log(m + _EPS)))
    return float(0.5 * (kl_pm + kl_qm))

def effect_corr_with_unperturbed(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between a and b."""
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return 0.0
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


# Registries
EFFECT_METRICS_ONE: Dict[str, Callable[..., float]] = {
    "l2norm": effect_l2norm,
    "mean_abs": effect_mean_abs,
    "max_abs": effect_max_abs,
    "entropy_magnitude": effect_entropy_magnitude,
    "gini": effect_gini,
    "participation_ratio": effect_participation_ratio,
    "fraction_suprathreshold": effect_fraction_suprathreshold,
    "pos_neg_balance": effect_pos_neg_balance,
}

EFFECT_METRICS_TWO: Dict[str, Callable[..., float]] = {
    "state_distance_l2": state_distance_l2,
    "state_mae": state_mae,
    "alignment_cosine": effect_alignment_cosine,
    "js_divergence": effect_js_divergence,
    "corr_with_unperturbed": effect_corr_with_unperturbed,
}

def list_effect_metrics() -> Tuple[str, ...]:
    """All effect metrics (both one- and two-vector) as a single tuple."""
    return tuple(list(EFFECT_METRICS_ONE.keys()) + list(EFFECT_METRICS_TWO.keys()))


# =============================================================================
# High-level table computation
# =============================================================================

def compute_metrics_table(
    xt_baseline: np.ndarray,      # (W, N)
    xt_next: np.ndarray,          # (W, N)
    xt_stim_next: np.ndarray,     # (N, W, N)
    baseline_metric_kind: str = "mean_abs_z",
    effect_metric_kind: str = "state_distance_l2",
    effect_input: str = "auto",
    # effect_input controls which vectors are fed into the effect metric:
    #   - "auto": choose based on metric arity (ONE -> use delta e; TWO -> use (x_stim, x_next))
    #   - "delta":       ONE-vector metric on e = x_stim - x_next
    #   - "stim":        ONE-vector metric on x_stim
    #   - "pair_state":  TWO-vector metric on (x_stim, x_next)
    #   - "pair_delta":  TWO-vector metric on (e, x_next)
    baseline_kwargs: Optional[dict] = None,
    effect_kwargs: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Build a tidy table with columns:
      ['w', 'tar', 'baseline_metric_kind', 'baseline_metric_value',
       'effect_metric_kind', 'effect_metric_value']

    Iterates over all windows w ∈ [0..W-1] and all targets tar ∈ [0..N-1].

    Returns:
        pd.DataFrame with (W * N) rows.
    """
    if baseline_kwargs is None: baseline_kwargs = {}
    if effect_kwargs is None: effect_kwargs = {}

    # Validate shapes
    W, N = xt_baseline.shape
    assert xt_next.shape == (W, N), f"xt_next must be (W, N), got {xt_next.shape}"
    assert xt_stim_next.shape == (N, W, N), f"xt_stim_next must be (N, W, N), got {xt_stim_next.shape}"

    # Resolve metrics
    if baseline_metric_kind not in BASELINE_METRICS:
        raise ValueError(f"Unknown baseline metric '{baseline_metric_kind}'. Available: {list_baseline_metrics()}")
    baseline_fn = BASELINE_METRICS[baseline_metric_kind]

    one_vec = EFFECT_METRICS_ONE.get(effect_metric_kind)
    two_vec = EFFECT_METRICS_TWO.get(effect_metric_kind)
    if (one_vec is None) and (two_vec is None):
        raise ValueError(f"Unknown effect metric '{effect_metric_kind}'. Available: {list_effect_metrics()}")

    # Prepare baseline context for metrics needing across-time stats
    ctx = BaselineContext(xt_baseline)

    # Decide effect input behavior
    if effect_input == "auto":
        effect_input = "delta" if one_vec is not None else "pair_state"
    if one_vec is not None and effect_input not in ("delta", "stim"):
        raise ValueError(f"Effect metric '{effect_metric_kind}' expects ONE vector; set effect_input='delta' or 'stim'.")
    if two_vec is not None and effect_input not in ("pair_state", "pair_delta"):
        raise ValueError(f"Effect metric '{effect_metric_kind}' expects TWO vectors; set effect_input='pair_state' or 'pair_delta'.")

    # Build table
    rows = []
    for w in range(W):
        x_t = xt_baseline[w]     # (N,)
        x_next = xt_next[w]      # (N,)

        # Baseline scalar
        try:
            b_val = float(baseline_fn(x_t, ctx=ctx, **baseline_kwargs))
        except TypeError:
            # Metric might not accept ctx (e.g., simple ones). Retry without it.
            b_val = float(baseline_fn(x_t, **baseline_kwargs))

        for tar in range(N):
            x_stim = xt_stim_next[tar, w]  # (N,)

            # Effect scalar
            if one_vec is not None:
                # Build v according to effect_input
                if effect_input == "delta":
                    v = x_stim - x_next
                elif effect_input == "stim":
                    v = x_stim
                else:
                    raise RuntimeError("Invalid effect_input for one-vector metric.")
                e_val = float(one_vec(v, **effect_kwargs))
            else:
                # two-vector
                if effect_input == "pair_state":
                    a, b = x_stim, x_next
                elif effect_input == "pair_delta":
                    a, b = x_stim - x_next, x_next
                else:
                    raise RuntimeError("Invalid effect_input for two-vector metric.")
                e_val = float(two_vec(a, b, **effect_kwargs))

            rows.append({
                "w": w,
                "tar": tar,
                "baseline_metric_kind": baseline_metric_kind,
                "baseline_metric_value": b_val,
                "effect_metric_kind": effect_metric_kind,
                "effect_metric_value": e_val,
            })

    df = pd.DataFrame(rows, columns=[
        "w", "tar", "baseline_metric_kind", "baseline_metric_value",
        "effect_metric_kind", "effect_metric_value"
    ])
    return df


# =============================================================================
# Brain state depencdence baseline vs effects
# =============================================================================

import numpy as np
import pandas as pd

def _cosine_distance_safe(u, v):
    """
    Cosine distance robust to zero vectors.
    u, v: arrays broadcastable to (..., N)
    returns: distance along last axis, shape (...)
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    dot = np.sum(u * v, axis=-1)
    uu = np.linalg.norm(u, axis=-1)
    vv = np.linalg.norm(v, axis=-1)
    denom = uu * vv

    cos_sim = np.zeros_like(dot, dtype=float)

    # safe mask: denom > 0
    mask = denom > 0
    cos_sim[mask] = dot[mask] / denom[mask]

    # if both zero -> define similarity=1 (distance=0)
    both_zero = (uu == 0) & (vv == 0)
    cos_sim[both_zero] = 1.0

    # numerical safety
    cos_sim = np.clip(cos_sim, -1.0, 1.0)

    return 1.0 - cos_sim


def make_effects_dataframe_simple(
    xt_baseline: np.ndarray,   # (W, N) baseline empirical state at t
    xt_next: np.ndarray,       # (W, N) unperturbed simulated state at t+dt
    xt_stim_next: np.ndarray,  # (J, W, N) stimulated simulated state at t+dt
    *,
    thresh: float = 1e-3,      # threshold for "affected region"
    relative: bool = False,    # if True, scale Δ by ROI std across windows
) -> pd.DataFrame:
    """
    Build a tidy dataframe with:
      window, target, total_energy_baseline,
      cosine_distance, l2_distance, l2_distance_sq, n_regions_affected

    Shapes expected:
      xt_baseline: (W, N)
      xt_next:     (W, N)
      xt_stim_next:(J, W, N)   where typically J == N (one target per region)
    """
    xt_baseline = np.asarray(xt_baseline, dtype=float)
    xt_next     = np.asarray(xt_next, dtype=float)
    xt_stim_next= np.asarray(xt_stim_next, dtype=float)

    W, N = xt_baseline.shape
    J = xt_stim_next.shape[0]
    assert xt_next.shape == (W, N), f"xt_next must be (W, N), got {xt_next.shape}"
    assert xt_stim_next.shape == (J, W, N), f"xt_stim_next must be (J, W, N), got {xt_stim_next.shape}"

    # --- Pre: total energy at baseline (||x_t||^2), per window
    total_energy = (xt_baseline** 2).sum(axis=1)  # (W,)

    # --- Post: differences
    delta = xt_stim_next - xt_next[None, :, :]     # (J, W, N)
    l2 = np.linalg.norm(delta, axis=-1)            # (J, W)
    l2_sq = (delta ** 2).sum(axis=-1)              # (J, W)
    cos_d = _cosine_distance_safe(xt_stim_next, xt_next[None, :, :])  # (J, W)

    # --- How many regions are "affected"
    delta_abs = np.abs(delta)                      # (J, W, N)
    if relative:
        # normalize by ROI-wise std across windows
        scale = xt_next.std(axis=0, keepdims=True)  # (1, N)
        delta_abs = delta_abs / (scale[None, None, :] + 1e-12)

    affected_counts = (delta_abs > thresh).sum(axis=-1)  # (J, W)

    # --- Build tidy dataframe
    windows = np.arange(W, dtype=int)
    targets = np.arange(J, dtype=int)  # label targets from first dim
    grid_w, grid_j = np.meshgrid(windows, targets, indexing='ij')  # (W, J)

    df = pd.DataFrame({
        "window": grid_w.ravel(),
        "target": grid_j.ravel(),
        "total_energy_baseline": np.repeat(total_energy, J),
        "cosine_distance": cos_d.T.ravel(),
        "l2_distance": l2.T.ravel(),
        "l2_distance_sq": l2_sq.T.ravel(),
        "n_regions_affected": affected_counts.T.ravel(),
    })

    return df

