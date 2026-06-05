"""共享预测器:洲际修正版 Elo,供各预测脚本复用。

回测显示在跨洲比赛上优于纯 Elo(见 scripts/run_confed_backtest.py),
世界杯每场都是跨洲,故默认采用此版本。
"""
from __future__ import annotations

import pandas as pd

from .baseline import MultiLogitHead
from .confed import conf_diff_column, confederation_ratings
from .elo import EloModel

HEAD_COLS = ["elo_diff", "abs_elo_diff", "conf_diff"]


def fit_predictor(results: pd.DataFrame):
    """用全部历史拟合:返回 (elo 引擎, 洲际强度, 概率头)。"""
    elo = EloModel()
    feat = elo.run(results)
    feat["abs_elo_diff"] = feat["elo_diff"].abs()
    cr = confederation_ratings(results)
    feat["conf_diff"] = conf_diff_column(feat, cr)
    head = MultiLogitHead(HEAD_COLS).fit(feat, feat.result.values)
    return elo, cr, head


def predict_fixtures(fixtures: pd.DataFrame, elo, cr, head) -> pd.DataFrame:
    """给未踢赛程补上洲际修正特征并输出 p_home/p_draw/p_away。"""
    fx = fixtures.copy()
    fx["elo_diff"] = [elo.elo_diff(r.home_team, r.away_team, r.neutral)
                      for r in fx.itertuples(index=False)]
    fx["abs_elo_diff"] = fx["elo_diff"].abs()
    fx["conf_diff"] = conf_diff_column(fx, cr)
    p = head.predict_proba(fx)
    fx["p_home"], fx["p_draw"], fx["p_away"] = p[:, 0], p[:, 1], p[:, 2]
    return fx


def worldcup_fixtures(results: pd.DataFrame) -> pd.DataFrame:
    return results[(~results.played)
                   & (results.tournament == "FIFA World Cup")
                   & (results.date >= "2026-01-01")].sort_values("date").copy()
