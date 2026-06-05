"""胜平负三分类预测的评估指标。

类别顺序固定为 [H, D, A](主胜 / 平 / 客胜),RPS 依赖该有序性。
"""
from __future__ import annotations

import numpy as np

CLASSES = ["H", "D", "A"]
_IDX = {"H": 0, "D": 1, "A": 2}


def _onehot(y_true) -> np.ndarray:
    Y = np.zeros((len(y_true), 3))
    for i, v in enumerate(y_true):
        Y[i, _IDX[v]] = 1.0
    return Y


def accuracy(y_true, proba) -> float:
    pred = np.asarray(CLASSES)[np.asarray(proba).argmax(1)]
    return float((pred == np.asarray(y_true)).mean())


def log_loss(y_true, proba, eps: float = 1e-15) -> float:
    Y = _onehot(y_true)
    p = np.clip(np.asarray(proba), eps, 1 - eps)
    return float(-(Y * np.log(p)).sum(1).mean())


def brier(y_true, proba) -> float:
    Y = _onehot(y_true)
    return float(((np.asarray(proba) - Y) ** 2).sum(1).mean())


def rps(y_true, proba) -> float:
    """Ranked Probability Score(足球预测的标准指标,越低越好)。

    RPS = 1/(r-1) * Σ_{i=1}^{r-1} (CDF_pred_i - CDF_obs_i)^2
    对有序类别敏感:把主胜预测成平,比预测成客胜罚得轻。
    """
    Y = _onehot(y_true)
    p = np.asarray(proba)
    cp = np.cumsum(p, axis=1)
    cy = np.cumsum(Y, axis=1)
    return float((((cp - cy) ** 2)[:, :-1].sum(1) / (p.shape[1] - 1)).mean())


def evaluate(y_true, proba) -> dict:
    return {
        "n": len(y_true),
        "accuracy": accuracy(y_true, proba),
        "log_loss": log_loss(y_true, proba),
        "brier": brier(y_true, proba),
        "rps": rps(y_true, proba),
    }
