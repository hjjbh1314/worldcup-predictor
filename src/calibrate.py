"""概率校准(多分类 Platt scaling)。

在模型输出的对数概率上再拟合一个多项逻辑回归,用近期、留出的数据学习一个
修正层。可纠正系统性偏差——例如近年(空场/中立场增多)主场优势下降导致的
"主胜被高估"。校准集必须是概率头未训练过的数据,避免乐观偏差。
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from .metrics import CLASSES


class PlattCalibrator:
    def __init__(self):
        self.clf = LogisticRegression(max_iter=3000)
        self.classes_: list[str] = []

    @staticmethod
    def _f(proba) -> np.ndarray:
        return np.log(np.clip(np.asarray(proba, dtype=float), 1e-6, 1.0))

    def fit(self, proba, y):
        self.clf.fit(self._f(proba), np.asarray(y))
        self.classes_ = list(self.clf.classes_)
        return self

    def predict_proba(self, proba) -> np.ndarray:
        p = self.clf.predict_proba(self._f(proba))
        order = [self.classes_.index(c) for c in CLASSES]
        return p[:, order]
