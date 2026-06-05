"""梯度提升模型(胜平负三分类)。

用 sklearn 的 HistGradientBoostingClassifier:
  - 原生支持缺失值(休息天数等首场为 NaN,无需填充)
  - 无需额外依赖,适合开源分发
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .features import FEATURE_COLS
from .metrics import CLASSES


class GBModel:
    def __init__(self, **kwargs):
        params = dict(
            learning_rate=0.05, max_iter=500, max_leaf_nodes=31,
            l2_regularization=1.0, min_samples_leaf=80,
            early_stopping=True, validation_fraction=0.1, random_state=42,
        )
        params.update(kwargs)
        self.clf = HistGradientBoostingClassifier(**params)
        self.classes_: list[str] = []

    def fit(self, X, y):
        self.clf.fit(np.asarray(X[FEATURE_COLS], dtype=float), np.asarray(y))
        self.classes_ = list(self.clf.classes_)
        return self

    def predict_proba(self, X) -> np.ndarray:
        p = self.clf.predict_proba(np.asarray(X[FEATURE_COLS], dtype=float))
        order = [self.classes_.index(c) for c in CLASSES]
        return p[:, order]
