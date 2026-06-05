"""共享预测器:洲际修正版 Elo,供各预测脚本复用。

回测显示在跨洲比赛上优于纯 Elo(见 scripts/run_confed_backtest.py),
世界杯每场都是跨洲,故默认采用此版本。
"""
from __future__ import annotations

import pandas as pd

from .baseline import MultiLogitHead
from .calibrate import PlattCalibrator
from .confed import conf_diff_column, confederation_ratings
from .elo import EloModel

HEAD_COLS = ["elo_diff", "abs_elo_diff", "conf_diff"]
CAL_YEARS = 3        # 用最近多少年的留出数据学校准层


def fit_predictor(results: pd.DataFrame, calibrate: bool = True):
    """用全部历史拟合:返回 (elo 引擎, 洲际强度, 概率头, 校准器或 None)。

    校准:概率头只在 cutoff 之前训练,校准器在最近 CAL_YEARS 年(留出)上拟合,
    纠正近期主场优势下降等系统性偏差。Elo 评分始终用全量历史(因果),不受影响。
    """
    elo = EloModel()
    feat = elo.run(results)
    feat["abs_elo_diff"] = feat["elo_diff"].abs()
    cr = confederation_ratings(results)
    feat["conf_diff"] = conf_diff_column(feat, cr)

    if calibrate:
        cutoff = feat["date"].max() - pd.DateOffset(years=CAL_YEARS)
        tr, cal = feat[feat.date < cutoff], feat[feat.date >= cutoff]
        head = MultiLogitHead(HEAD_COLS).fit(tr, tr.result.values)
        calib = PlattCalibrator().fit(head.predict_proba(cal), cal.result.values)
    else:
        head = MultiLogitHead(HEAD_COLS).fit(feat, feat.result.values)
        calib = None
    return elo, cr, head, calib


def predict_fixtures(fixtures: pd.DataFrame, elo, cr, head, calib=None) -> pd.DataFrame:
    """给未踢赛程补上洲际修正特征并输出 p_home/p_draw/p_away(含校准)。"""
    fx = fixtures.copy()
    fx["elo_diff"] = [elo.elo_diff(r.home_team, r.away_team, r.neutral)
                      for r in fx.itertuples(index=False)]
    fx["abs_elo_diff"] = fx["elo_diff"].abs()
    fx["conf_diff"] = conf_diff_column(fx, cr)
    p = head.predict_proba(fx)
    if calib is not None:
        p = calib.predict_proba(p)
    fx["p_home"], fx["p_draw"], fx["p_away"] = p[:, 0], p[:, 1], p[:, 2]
    return fx


def worldcup_fixtures(results: pd.DataFrame) -> pd.DataFrame:
    return results[(~results.played)
                   & (results.tournament == "FIFA World Cup")
                   & (results.date >= "2026-01-01")].sort_values("date").copy()
