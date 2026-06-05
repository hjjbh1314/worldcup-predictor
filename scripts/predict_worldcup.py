"""用全部历史训练 Elo,对 2026 世界杯未踢赛程输出胜平负概率。

用法:
    python -m scripts.predict_worldcup
输出:
    outputs/wc2026_predictions.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_results       # noqa: E402
from src.elo import EloModel            # noqa: E402
from src.baseline import EloProbHead    # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "outputs" / "wc2026_predictions.csv"


def main():
    results = load_results()

    # 用全部已踢比赛训练 Elo 与概率头(部署阶段用满数据)
    elo = EloModel()
    feat = elo.run(results)                      # 跑完后 elo.ratings 为最新评分
    head = EloProbHead().fit(feat.elo_diff.values, feat.result.values)

    # 取未踢的 2026 世界杯赛程
    fix = results[(~results.played) & (results.tournament == "FIFA World Cup")].copy()
    fix = fix[fix.date >= "2026-01-01"].sort_values("date")

    ed = [elo.elo_diff(r.home_team, r.away_team, r.neutral)
          for r in fix.itertuples(index=False)]
    proba = head.predict_proba(ed)

    fix["p_home"] = proba[:, 0]
    fix["p_draw"] = proba[:, 1]
    fix["p_away"] = proba[:, 2]
    fix["elo_home"] = [round(elo.ratings[t]) for t in fix.home_team]
    fix["elo_away"] = [round(elo.ratings[t]) for t in fix.away_team]
    fix["pick"] = proba.argmax(1)
    pick_map = {0: "主胜", 1: "平", 2: "客胜"}
    fix["pick"] = fix["pick"].map(pick_map)

    cols = ["date", "home_team", "away_team", "neutral",
            "elo_home", "elo_away", "p_home", "p_draw", "p_away", "pick", "city"]
    out = fix[cols].reset_index(drop=True)
    OUT.parent.mkdir(exist_ok=True)
    out.to_csv(OUT, index=False)

    # —— 当前实力榜(参赛队 Elo)——
    teams = set(fix.home_team) | set(fix.away_team)
    rank = sorted(((round(elo.ratings[t]), t) for t in teams), reverse=True)
    print("=== 2026 世界杯参赛队 Elo 实力榜 Top 15 ===")
    for i, (rt, t) in enumerate(rank[:15], 1):
        print(f"  {i:2d}. {t:<22} {rt}")

    print(f"\n=== 小组赛预测(共 {len(out)} 场,前 12 场示例)===")
    show = out.head(12).copy()
    for c in ("p_home", "p_draw", "p_away"):
        show[c] = (show[c] * 100).round(1).astype(str) + "%"
    print(show[["date", "home_team", "away_team",
                "p_home", "p_draw", "p_away", "pick"]].to_string(index=False))
    print(f"\n完整预测已保存:{OUT}")


if __name__ == "__main__":
    main()
