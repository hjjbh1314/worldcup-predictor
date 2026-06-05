"""特征工程:一次时间顺序遍历,产出每场比赛的赛前特征。

所有特征都是"赛前可知"的(只用该场之前的信息),严格无泄漏。
涵盖用户关心的维度:
  - 球队实力      : Elo 评分 / 评分差
  - 球队状态      : 近 N 场积分、进失球(滚动)
  - 场地适应      : 是否中立场(主场优势已在 Elo 内体现)
  - 赛程激烈程度  : 距上场休息天数、近期场次密度(疲劳)
  - 赛事重要性    : 赛事权重 K
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

from .elo import EloModel, tournament_k

FORM_WINDOW = 10        # 近期状态窗口(场)
CONGESTION_DAYS = 14    # 赛程密度统计窗口(天)

FEATURE_COLS = [
    "elo_diff", "elo_home", "elo_away",
    "form_diff", "gf_diff", "ga_diff",
    "rest_diff", "congestion_diff",
    "rest_home", "rest_away",
    "neutral", "tournament_k",
]


def _team_state():
    return {
        "form": deque(maxlen=FORM_WINDOW),   # 每场积分 3/1/0
        "gf": deque(maxlen=FORM_WINDOW),     # 进球
        "ga": deque(maxlen=FORM_WINDOW),     # 失球
        "last_date": None,                   # 上场比赛日期
        "recent": deque(),                   # 近期比赛日期(算密度)
    }


def _mean(dq):
    return float(np.mean(dq)) if len(dq) else np.nan


def build_feature_table(results: pd.DataFrame, **elo_kwargs) -> pd.DataFrame:
    elo = EloModel(**elo_kwargs)
    st: dict[str, dict] = defaultdict(_team_state)
    rows = []

    for r in results.itertuples(index=False):
        if not r.played:
            continue
        h, a, date = r.home_team, r.away_team, r.date
        sh, sa = st[h], st[a]

        # —— 休息天数 / 赛程密度(赛前)——
        rest_h = (date - sh["last_date"]).days if sh["last_date"] else np.nan
        rest_a = (date - sa["last_date"]).days if sa["last_date"] else np.nan
        for s in (sh, sa):
            while s["recent"] and (date - s["recent"][0]).days > CONGESTION_DAYS:
                s["recent"].popleft()
        cong_h, cong_a = len(sh["recent"]), len(sa["recent"])

        rows.append({
            "date": date, "home_team": h, "away_team": a,
            "elo_home": elo.ratings[h], "elo_away": elo.ratings[a],
            "elo_diff": elo.elo_diff(h, a, r.neutral),
            "form_diff": _mean(sh["form"]) - _mean(sa["form"]),
            "gf_diff": _mean(sh["gf"]) - _mean(sa["gf"]),
            "ga_diff": _mean(sh["ga"]) - _mean(sa["ga"]),
            "rest_home": rest_h, "rest_away": rest_a,
            "rest_diff": (rest_h - rest_a) if not (np.isnan(rest_h) or np.isnan(rest_a)) else np.nan,
            "congestion_diff": cong_h - cong_a,
            "neutral": int(bool(r.neutral)),
            "tournament_k": tournament_k(r.tournament),
            "result": r.result,
        })

        # —— 更新状态(赛后)——
        elo._update_one(h, a, r.home_score, r.away_score, r.tournament, r.neutral)
        hp = 3 if r.result == "H" else (1 if r.result == "D" else 0)
        ap = 3 if r.result == "A" else (1 if r.result == "D" else 0)
        sh["form"].append(hp); sa["form"].append(ap)
        sh["gf"].append(r.home_score); sh["ga"].append(r.away_score)
        sa["gf"].append(r.away_score); sa["ga"].append(r.home_score)
        sh["last_date"] = date; sa["last_date"] = date
        sh["recent"].append(date); sa["recent"].append(date)

    return pd.DataFrame(rows)
