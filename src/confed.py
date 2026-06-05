"""洲际强度评分:只用跨洲比赛,学习各大洲的相对强弱。

思路:把每场跨洲比赛看作 confed_home vs confed_away,跑一个大洲级别的 Elo。
同洲比赛对大洲间强弱没有信息(零和内部转移),跳过。
收敛后即得到 UEFA / CONMEBOL / ... 的相对强度,用于修正纯队伍 Elo 的结构性偏差。
"""
from __future__ import annotations

from collections import defaultdict

import pandas as pd

from .confederations import get_confederation


def confederation_ratings(results: pd.DataFrame, before: str | None = None,
                          init: float = 1500.0, k: float = 15.0) -> dict[str, float]:
    """跑大洲级 Elo,返回去均值后的各大洲强度(仅用 before 之前的比赛,防泄漏)。"""
    cr: dict[str, float] = defaultdict(lambda: init)
    cutoff = pd.Timestamp(before) if before else None

    for r in results.itertuples(index=False):
        if not r.played:
            continue
        if cutoff is not None and r.date >= cutoff:
            break                      # results 已按日期排序
        ch, ca = get_confederation(r.home_team), get_confederation(r.away_team)
        if not ch or not ca or ch == ca:
            continue                   # 跳过未知 / 同洲
        we = 1.0 / (1.0 + 10 ** (-(cr[ch] - cr[ca]) / 400.0))
        w = 1.0 if r.home_score > r.away_score else (0.0 if r.home_score < r.away_score else 0.5)
        delta = k * (w - we)
        cr[ch] += delta
        cr[ca] -= delta

    mean = sum(cr.values()) / len(cr) if cr else init
    return {c: v - mean for c, v in cr.items()}   # 去均值,便于解释


def conf_diff_column(df: pd.DataFrame, cr: dict[str, float]) -> pd.Series:
    """每场比赛的洲际强度差(主队大洲 − 客队大洲);同洲或未知为 0。"""
    def diff(h, a):
        ch, ca = get_confederation(h), get_confederation(a)
        if not ch or not ca:
            return 0.0
        return cr.get(ch, 0.0) - cr.get(ca, 0.0)
    return pd.Series([diff(h, a) for h, a in zip(df.home_team, df.away_team)],
                     index=df.index)
