"""回测:按时间切分,训练概率头,在"未来"赛段上评估。

关键原则:严格按时间切分,绝不用未来信息训练。
  1. Elo 评分在线计算,产出每场赛前的 elo_diff(因果,无泄漏)。
  2. 概率头只在 train 段拟合,在 test 段评估。
"""
from __future__ import annotations

import pandas as pd

from .baseline import BaseRateModel, EloProbHead
from .elo import EloModel
from .metrics import evaluate


def build_features(results: pd.DataFrame, **elo_kwargs) -> pd.DataFrame:
    """跑一遍 Elo,得到每场已踢比赛的赛前特征表。"""
    return EloModel(**elo_kwargs).run(results)


def time_split_backtest(results: pd.DataFrame, split: str = "2018-01-01",
                        **elo_kwargs) -> dict:
    """split 之前训练,split 当天及之后测试。"""
    feat = build_features(results, **elo_kwargs)
    train = feat[feat.date < split]
    test = feat[feat.date >= split]
    if len(train) == 0 or len(test) == 0:
        raise ValueError(f"切分点 {split} 导致训练或测试集为空")

    elo_head = EloProbHead().fit(train.elo_diff.values, train.result.values)
    elo_proba = elo_head.predict_proba(test.elo_diff.values)

    naive = BaseRateModel().fit(train.result.values)
    naive_proba = naive.predict_proba(len(test))

    return {
        "split": split,
        "n_train": len(train),
        "n_test": len(test),
        "elo": evaluate(test.result.values, elo_proba),
        "naive": evaluate(test.result.values, naive_proba),
        "test": test,
        "elo_proba": elo_proba,
        "head": elo_head,
        "features": feat,
    }
