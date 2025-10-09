import numpy as np
import time
import os, sys
from scipy.stats import pearsonr
sys.path.append(os.path.abspath('../src'))

from NPI import *

def train_models_for_groups(
    groups,
    steps=3,                       # past steps S for multi2one
    hidden_rule=lambda N: 2 * N,   # hidden size rule-of-thumb
    latent_rule=lambda N: int(0.8 * N),
    batch_size=50,
    train_prop=0.8,
    num_epochs=100,
    lr=1e-3,
    l2=5e-5,
    min_windows=50,                # skip subjects that are too short to train
    save_dir=None,                 # e.g. "../models"
    save_prefix="ANN_subject"      # filename prefix if saving
):
    """
    Trains an ANN_MLP per subject and returns a grouped dict:
      results[group] = [ { 'model', 'history', 'subject_idx', 'shapes', 'train_time_s' }, ... ]
    """
    results = {}
    os.makedirs(save_dir, exist_ok=True) if save_dir is not None else None

    for group_name, subject_list in groups.items():
        group_out = []
        print(f"\n=== Group: {group_name} — {len(subject_list)} subjects ===")
        for i, Z in enumerate(subject_list):
            # Z: (T, N)
            T, N = Z.shape
            # Prepare windows
            X, Y = multi2one(Z, steps=steps)  # X: (W, S*N), Y: (W, N)
            W = X.shape[0]

            if W < min_windows:
                print(f"[SKIP] {group_name}[{i}] — not enough windows (W={W} < {min_windows})")
                continue

            input_dim  = steps * N
            hidden_dim = hidden_rule(N)
            latent_dim = max(2, latent_rule(N))  # keep >=2
            output_dim = N

            model = ANN_MLP(input_dim, hidden_dim, latent_dim, output_dim)

            t0 = time.time()
            model, train_loss_hist, test_loss_hist = train_NN(
                model,
                X, Y,
                batch_size=batch_size,
                train_set_proportion=train_prop,
                num_epochs=num_epochs,
                lr=lr,
                l2=l2,
            )
            dt = time.time() - t0

            print(f"[OK]  {group_name}[{i}] — T={T}, N={N}, W={W} | "
                  f"final train={train_loss_hist[-1]:.6f}, test={test_loss_hist[-1]:.6f} "
                  f"({dt:.1f}s)")

            entry = {
                "model": model,
                "history": {
                    "train_loss": np.array(train_loss_hist),
                    "test_loss":  np.array(test_loss_hist),
                },
                "subject_idx": i,
                "shapes": {"T": T, "N": N, "W": W, "steps": steps},
                "train_time_s": dt,
            }
            group_out.append(entry)

            # Save weights
            if save_dir is not None:
                try:
                    import torch
                    fname = f"{save_prefix}_{group_name}_sub{i}_T{T}_N{N}_S{steps}.pt"
                    torch.save(model.state_dict(), os.path.join(save_dir, fname))
                except Exception as e:
                    print(f"   ⚠️ Could not save weights for {group_name}[{i}]: {e}")

        results[group_name] = group_out

    return results

def compute_FC_EC_for_all(
    groups,
    models_by_group,
    ec_method: str = "perturb",   # "perturb" | "jacobian"
    pert_strength: float = .1,   # only used when ec_method="perturb"
    dtype=np.float32
):
    """
    Compute simulated FC and EC for every participant's trained ANN.

    Args:
        groups: dict[str, list[np.ndarray]]
            Raw time series per group. Example:
              groups["CNT"][i] -> Z_i with shape (T, N)

        models_by_group: dict[str, list[dict]]
            Output of your training wrapper. Each entry contains:
              - "model": trained torch model
              - "subject_idx": index in groups[group]
              - "shapes": {"T": T, "N": N, "W": W, "steps": S}
              - "history", "train_time_s", etc.

        ec_method: "perturb" or "jacobian"
            Which EC estimator to use:
              - "perturb"  -> NPI.model_EC (requires X, Y, pert_strength)
              - "jacobian" -> NPI.model_Jacobian (requires X, steps)

        pert_strength: float
            Magnitude for the perturbation method.

        dtype: np.dtype
            Cast FC/EC to this dtype to keep memory predictable.

    Returns:
        conn: dict[group] -> list of dict per subject with:
            {
              "subject_idx": int,
              "FC": (N, N) array,
              "EC": (N, N) array,
              "meta": {
                  "steps": int,
                  "T": int,
                  "N": int,
                  "method": "perturb" | "jacobian"
              }
            }
    """
    assert ec_method in ("perturb", "jacobian"), "ec_method must be 'perturb' or 'jacobian'"

    results = {}
    for group_name, model_entries in models_by_group.items():
        out_list = []
        print(f"\n=== Connectivity: {group_name} ({len(model_entries)} models) ===")

        for entry in model_entries:
            subj_idx = entry["subject_idx"]
            model    = entry["model"]
            S        = entry["shapes"]["steps"]

            # Pull the matching raw series Z (T, N)
            Z = groups[group_name][subj_idx]
            T, N = Z.shape

            # Prepare windowed data for EC methods that need it
            X, Y = multi2one(Z, steps=S)  # X: (W, S*N), Y: (W, N)

            # --- FC from simulated activity via the surrogate ---
            FC_sim = model_FC(model, node_num=N, steps=S).astype(dtype)

            # --- EC via chosen method ---
            if ec_method == "perturb":
                EC_est = model_EC(model, input_X=X, target_Y=Y, pert_strength=pert_strength).astype(dtype)
            else:  # "jacobian"
                EC_est = model_Jacobian(model, input_X=X, steps=S).astype(dtype)

            EABC_est = model_EABC(model, input_X=X, target_Y=Y, pert_strength=pert_strength)#.astype(dtype)
            EABC_red = model_EC_reduced(model, input_X=X, target_Y=Y, pert_strength=pert_strength)#.astype(dtype)

            out = {
                "subject_idx": subj_idx,
                "FC": FC_sim,
                "EC": EC_est,
                "EABC_Cosine": EABC_est[0].astype(dtype),
                "EABC_L2": EABC_est[1].astype(dtype),
                "EC_min": EABC_red[0].astype(dtype),
                "EC_max": EABC_red[1].astype(dtype),
                "EC_rand": EABC_red[2].astype(dtype),
                "meta": {"steps": S, "T": T, "N": N, "method": ec_method}
            }
            out_list.append(out)

            print(f"[OK] {group_name}[{subj_idx}] -> FC, EC computed "
                  f"(T={T}, N={N}, steps={S}, method={ec_method})")

        results[group_name] = out_list

    return results

# def flatten_upper(mat: np.ndarray) -> np.ndarray:
#     """Return vector of upper-triangle (i<j) entries of a square matrix."""
#     n = mat.shape[0]
#     iu = np.triu_indices(n, k=1)
#     return mat[iu]

# def empirical_vs_simulated_corr(groups, conn):
#     """
#     Compute correlation between empirical and simulated FC for each subject.

#     Args:
#         groups: dict[group] -> list of np.ndarray (raw time series Z: (T, N))
#         conn:   dict[group] -> list of dict with keys "subject_idx", "FC", "EC", ...

#     Returns:
#         results: dict[group] -> list of dict with subject_idx and corr
#     """
#     out = {}
#     for group_name, subjects in groups.items():
#         res_group = []
#         for entry in conn[group_name]:
#             subj_idx = entry["subject_idx"]
#             Z = subjects[subj_idx]
#             # Empirical FC
#             FCemp = np.corrcoef(Z, rowvar=False)
#             # Simulated FC
#             FCsim = entry["FC"]

#             # Vectorize upper triangles
#             v_emp = flatten_upper(FCemp)
#             v_sim = flatten_upper(FCsim)

#             # Correlation
#             r, _ = pearsonr(v_emp, v_sim)

#             res_group.append({"subject_idx": subj_idx, "corr": r})
#             print(f"{group_name}[{subj_idx}] corr(emp,sim) = {r:.3f}")

#         out[group_name] = res_group
#     return out


