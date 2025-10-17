import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Hashable, List, Tuple, Optional, Union
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

Nested = Dict[Hashable, Dict[Hashable, np.ndarray]]

def classification(
    matrix: Nested,
    *,
    time: bool,                                # True if temporal (M,N,N) and should be averaged
    groups: Optional[List[Hashable]] = None,   # e.g. ["CNT","MCS","UWS"] or ["MCS","UWS"];
    drop_diagonal: bool = False,                # zero diagonal before flattening
    subjects_per_group: Optional[Dict[Hashable, List[Hashable]]] = None,
    # SVM / CV options
    kernel: str = "rbf",                       # "linear" or "rbf"
    scoring: str = "balanced_accuracy",        # good default for unbalanced classes
    cv_splits: int = 5,
    class_weight: Union[str, dict, None] = "balanced",
    random_state: int = 42,
    refit: bool = True,
    n_jobs: Optional[int] = None,
    # Plot options
    plot_cm: bool = True,
    use_cv_for_cm: bool = True,                
    normalize_cm: Optional[str] = "true",      
    title: str = "SVM"
) -> Dict[str, object]:
    """
    matrix: {group: {subject: array}}
        Each array is either (N,N) or temporal (M,N,N). If `time=True`, temporal arrays are averaged over axis 0.
    groups: choose 2 or 3 group labels (order defines label mapping); default = all groups in dict order.
    Returns:
      {
        'gs': GridSearchCV (fitted),
        'X': features (np.ndarray),
        'y': labels (np.ndarray),
        'label_map': {int_label: group_name},
        'best_score_': float,
        'best_params_': dict,
        'confusion_matrix_': np.ndarray (if plotted),
      }
    """

    # Average temporal if requested 
    def average_time(
        data: Nested,
        do_time: bool,
    ) -> Nested:
        """
        If the data has information over time, average to get an (N x N) matrix per subject.
        If it's already averaged, don't change anything.

        Parameters
        ----------
        matrix : {group: {subject: array}}
            If temporal, expects arrays shaped (M, N, N); otherwise (N, N).
        time : bool
            True if the data are temporal and should be averaged across the first axis.
        """
        out: Nested = {}
        for g, subjects in data.items():
            out[g] = {}
            for isu, arr in subjects.items():
                arr = np.asarray(arr)
                if do_time and arr.ndim == 3:
                    out[g][isu] = np.mean(arr, axis=0)  # average over M
                elif arr.ndim == 2:
                    out[g][isu] = arr
                else:
                    # if time=False but got 3D, still average to be safe
                    out[g][isu] = arr if arr.ndim == 2 else np.mean(arr, axis=0)
        return out

    avg = average_time(matrix, time)

    # Pick groups & build X and Y 
    if groups is None:
        groups = list(avg.keys())
    if len(groups) not in (2, 3):
        raise ValueError(f"'groups' must have length 2 or 3; got {len(groups)}")

    X_list, y_list = [], []
    for gi, g in enumerate(groups):
        subs = subjects_per_group.get(g) if subjects_per_group else list(avg[g].keys())
        for sid in subs:
            M = np.asarray(avg[g][sid])
            if drop_diagonal and M.shape[0] == M.shape[1]:
                i = np.arange(M.shape[0])
                M = M.copy()
                M[i, i] = 0.0
            X_list.append(M.ravel())
            y_list.append(gi)
    X = np.vstack(X_list)
    y = np.asarray(y_list, dtype=int)
    label_map = {i: g for i, g in enumerate(groups)}

    # Train SVM with CV + grid search 
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("svc", SVC(kernel=kernel, class_weight=class_weight, probability=True,
                    decision_function_shape="ovr")),
    ])
    if kernel == "linear":
        param_grid = {"svc__C": np.logspace(-3, 3, 7)}
    elif kernel == "rbf":
        param_grid = {
            "svc__C": np.logspace(-3, 3, 7),
            "svc__gamma": np.logspace(-4, 1, 6),
        }
    else:
        raise ValueError("kernel must be 'linear' or 'rbf'")

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    gs = GridSearchCV(
        pipe, param_grid,
        scoring=scoring, cv=cv, refit=refit, n_jobs=n_jobs, return_train_score=False
    )
    gs.fit(X, y)

    # Optional confusion matrix
    cm = None
    if plot_cm:
        estimator = getattr(gs, "best_estimator_", gs)
        y_pred = cross_val_predict(clone(estimator), X, y, cv=cv)
        subtitle = f"{title} (CV {cv_splits}-fold)"
        cm = confusion_matrix(y, y_pred, normalize=normalize_cm)
        labels = [label_map[i] for i in sorted(set(y))]
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(values_format=".2f" if normalize_cm else "d")
        if normalize_cm:
            disp.im_.set_clim(0, 1)
        plt.title(subtitle)
        plt.tight_layout()
        plt.show()

    return {
        "gs": gs,
        "X": X,
        "y": y,
        "label_map": label_map,
        "best_score_": float(gs.best_score_),
        "best_params_": gs.best_params_,
        "confusion_matrix_": cm,
    }

