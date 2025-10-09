import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy, ttest_ind
from scipy.linalg import eigh
from itertools import combinations
from scipy import stats

# --------------------
# Helper functions
# --------------------

def compute_fc(timeseries):
    """Compute Pearson correlation FC (N x N) from time x regions data."""
    return np.corrcoef(timeseries, rowvar=False)

# def eigen_entropy(fc):
#     """Shannon entropy of normalized eigenvalues of FC."""
#     eigvals = eigh(fc, eigvals_only=True)
#     eigvals = np.abs(eigvals)  # ensure non-negativity
#     eigvals /= eigvals.sum()
#     return entropy(eigvals)

# def lz_complexity(ts):
#     """Simple Lempel–Ziv complexity on binarized signal (all regions concatenated)."""
#     # Flatten across regions
#     x = ts.flatten()
#     # Binarize around median
#     x_bin = (x > np.median(x)).astype(int)
#     # LZ parsing
#     s = ''.join(map(str, x_bin))
#     i, c, k, l = 0, 1, 1, 1
#     n = len(s)
#     while True:
#         if s[i+k-1] == s[l+k-1]:
#             k += 1
#             if l+k > n:
#                 c += 1
#                 break
#         else:
#             if k > l:
#                 l = k
#             i += 1
#             if i == l:
#                 c += 1
#                 l += 1
#                 if l > n:
#                     break
#                 i = 0
#             k = 1
#     return c / (len(s) / np.log2(len(s)))  # normalized LZ

# def compute_metrics(groups):
#     """Compute FC entropy (+ LZ?) for all subjects across groups."""
#     metrics = {"group": [], "entropy": []}#, "lz": []}
#     for group_name, subjects in groups.items():
#         for subj in subjects:
#             fc = compute_fc(subj)
#             metrics["group"].append(group_name)
#             metrics["entropy"].append(eigen_entropy(fc))
#             #metrics["lz"].append(lz_complexity(subj))
#     return metrics

# --------------------
# Plotting
# --------------------

# def violin_with_stats(df, metric, colors):
#     plt.figure(figsize=(6,4))
#     sns.violinplot(x="group", y=metric, data=df, palette=colors, inner="box")
    
#     # Pairwise t-tests
#     pairs = list(combinations(df["group"].unique(), 2))
#     y_max = df[metric].max()
#     for i, (g1, g2) in enumerate(pairs):
#         vals1 = df[df["group"]==g1][metric]
#         vals2 = df[df["group"]==g2][metric]
#         t, p = ttest_ind(vals1, vals2, equal_var=False)
#         x1, x2 = list(df["group"].unique()).index(g1), list(df["group"].unique()).index(g2)
#         y = y_max + (i+1)*0.05*y_max
#         plt.plot([x1, x1, x2, x2], [y, y+0.01*y_max, y+0.01*y_max, y], lw=1.5, c="k")
#         plt.text((x1+x2)/2, y+0.02*y_max, f"p={p:.3f}", ha="center", va="bottom")
    
#     plt.title(f"{metric} across groups")
#     plt.tight_layout()
#     plt.show()


def go_edge(tseries):
    nregions=tseries.shape[1]
    Blen=tseries.shape[0]
    nedges=int(nregions**2/2-nregions/2)
    iTriup= np.triu_indices(nregions,k=1) 
    gz=stats.zscore(tseries)
    Eseries = gz[:,iTriup[0]]*gz[:,iTriup[1]]
    return Eseries

def dFC(tseries):
    return np.corrcoef(go_edge(tseries))