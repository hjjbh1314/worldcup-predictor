"""世界足球 Elo 评分模型(参考 eloratings.net 方法)。

评分更新:R' = R + K * G * (W - We)
  We : 期望得分 = 1 / (1 + 10^(-dr/400)),dr 为评分差(含主场优势)
  W  : 实际得分(胜 1 / 平 0.5 / 负 0)
  K  : 赛事重要性权重
  G  : 净胜球修正系数

Elo 是在线/因果模型:t 时刻的评分只用到 t 之前的比赛,
因此用它产出的赛前评分差做特征不存在数据泄漏。
"""
from __future__ import annotations

from collections import defaultdict

import pandas as pd


def tournament_k(tournament: str) -> float:
    """赛事重要性权重 K。"""
    t = str(tournament).lower()
    if "qualif" in t:                 # 各类预选赛
        return 40.0
    if "friendly" in t:               # 友谊赛
        return 20.0
    if "world cup" in t:              # 世界杯决赛圈
        return 60.0
    majors = ("uefa euro", "copa am", "african cup of nations", "afc asian cup",
              "gold cup", "concacaf", "confederations cup", "nations league")
    if any(m in t for m in majors):   # 各大洲杯赛决赛圈 + 洲际赛
        return 50.0
    return 30.0                       # 其他赛事


def goal_multiplier(goal_diff: int) -> float:
    """净胜球修正系数 G(净胜越多,评分调整越大)。"""
    gd = abs(int(goal_diff))
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11 + gd) / 8.0            # gd>=3:3->1.75, 4->1.875, ...


class EloModel:
    def __init__(self, init_rating: float = 1500.0, home_advantage: float = 100.0,
                 use_goal_diff: bool = True):
        self.init_rating = init_rating
        self.home_advantage = home_advantage
        self.use_goal_diff = use_goal_diff
        self.ratings: dict[str, float] = defaultdict(lambda: init_rating)

    def elo_diff(self, home: str, away: str, neutral: bool = False) -> float:
        ha = 0.0 if neutral else self.home_advantage
        return (self.ratings[home] + ha) - self.ratings[away]

    def expected(self, home: str, away: str, neutral: bool = False) -> float:
        return 1.0 / (1.0 + 10 ** (-self.elo_diff(home, away, neutral) / 400.0))

    def _update_one(self, home, away, hs, as_, tournament, neutral):
        we = self.expected(home, away, neutral)
        w = 1.0 if hs > as_ else (0.0 if hs < as_ else 0.5)
        k = tournament_k(tournament)
        g = goal_multiplier(hs - as_) if self.use_goal_diff else 1.0
        delta = k * g * (w - we)
        self.ratings[home] += delta
        self.ratings[away] -= delta

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """按时间顺序处理所有已踢比赛,返回每场的赛前特征(无泄漏)。"""
        rows = []
        for r in df.itertuples(index=False):
            if not r.played:
                continue
            rows.append((
                r.date, r.home_team, r.away_team,
                self.ratings[r.home_team], self.ratings[r.away_team],
                self.elo_diff(r.home_team, r.away_team, r.neutral),
                bool(r.neutral), r.tournament, r.result,
            ))
            self._update_one(r.home_team, r.away_team,
                             r.home_score, r.away_score, r.tournament, r.neutral)
        return pd.DataFrame(rows, columns=[
            "date", "home_team", "away_team", "elo_home", "elo_away",
            "elo_diff", "neutral", "tournament", "result",
        ])
