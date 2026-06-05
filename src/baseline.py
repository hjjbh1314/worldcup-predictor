"""Elo 概率头:把赛前 Elo 评分差映射成胜/平/负三概率。

Elo 本身只给"期望得分"(把平局混在里面),无法直接得到三分类概率。
这里用一个很薄的多项逻辑回归作为概率头:
  特征 = [elo_diff, |elo_diff|]
  加入 |elo_diff| 是为了让"平局概率在实力接近时达到峰值"(非单调),
  这是单一线性特征做不到的。

它只在训练集上拟合 3 个方向的系数,不接触测试集,因此无泄漏。
仍属于"Elo 基线"——只是给 Elo 配了一个概率化的头。
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .metrics import CLASSES


class EloProbHead:
    def __init__(self):
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(max_iter=2000)
        self.classes_: list[str] = []

    @staticmethod
    def _feat(elo_diff) -> np.ndarray:
        ed = np.asarray(elo_diff, dtype=float).reshape(-1, 1)
        return np.hstack([ed, np.abs(ed)])

    def fit(self, elo_diff, y):
        X = self.scaler.fit_transform(self._feat(elo_diff))
        self.clf.fit(X, np.asarray(y))
        self.classes_ = list(self.clf.classes_)
        return self

    def predict_proba(self, elo_diff) -> np.ndarray:
        X = self.scaler.transform(self._feat(elo_diff))
        p = self.clf.predict_proba(X)
        order = [self.classes_.index(c) for c in CLASSES]   # 统一成 [H,D,A]
        return p[:, order]


class BaseRateModel:
    """朴素基线:无论谁打谁,都预测训练集里的胜平负总体频率。"""

    def __init__(self):
        self.rates = np.array([1 / 3, 1 / 3, 1 / 3])

    def fit(self, y):
        y = np.asarray(y)
        self.rates = np.array([(y == c).mean() for c in CLASSES])
        return self

    def predict_proba(self, n: int) -> np.ndarray:
        return np.tile(self.rates, (n, 1))
