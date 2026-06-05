"""Elo + 博彩赔率:对 2026 世界杯赛程出预测,并标出模型与市场分歧最大的比赛。

需要环境变量 ODDS_API_KEY(The Odds API 免费 key)。未设置时自动退回纯 Elo。
用法:
    export ODDS_API_KEY=你的key
    python -m scripts.predict_with_odds
输出:
    outputs/wc2026_with_odds.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_results       # noqa: E402
from src.predict import fit_predictor, predict_fixtures, worldcup_fixtures  # noqa: E402
from src import odds as oddslib         # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "outputs" / "wc2026_with_odds.csv"


def elo_predictions():
    results = load_results()
    elo, cr, head, calib = fit_predictor(results)
    fix = predict_fixtures(worldcup_fixtures(results), elo, cr, head, calib)
    fix["elo_h"], fix["elo_d"], fix["elo_a"] = fix.p_home, fix.p_draw, fix.p_away
    return fix


def main():
    fix = elo_predictions()
    key = oddslib.get_api_key()

    if not key:
        print("⚠️  未检测到 ODDS_API_KEY,仅输出纯 Elo 预测。")
        print("    设置后可叠加实时赔率:export ODDS_API_KEY=你的key\n")
        show = fix[["date", "home_team", "away_team", "elo_h", "elo_d", "elo_a"]].head(12).copy()
        show[["elo_h", "elo_d", "elo_a"]] = show[["elo_h", "elo_d", "elo_a"]].round(3)
        print(show.to_string(index=False))
        return

    print("拉取 The Odds API 实时赔率中(消耗 ~1 credit)…")
    try:
        events = oddslib.fetch_odds(key, sport_key="soccer_fifa_world_cup")
    except Exception as e:                                   # noqa: BLE001
        print(f"❌ 拉取失败:{e}\n请用 list_soccer_sports 确认赛事 key 是否已上线。")
        return
    idx = oddslib.build_odds_index(events)
    print(f"匹配到 {len(idx)} 场带赔率的比赛。\n")

    rows = []
    for r in fix.itertuples(index=False):
        teams = frozenset([oddslib.canon(r.home_team), oddslib.canon(r.away_team)])
        mk = idx.get(teams)
        mh = md = ma = np.nan
        if mk:
            mh = mk.get(oddslib.canon(r.home_team), np.nan)
            ma = mk.get(oddslib.canon(r.away_team), np.nan)
            md = mk.get("Draw", np.nan)
        # 模型与市场在"主胜"上的分歧度(研究价值:正=模型比市场更看好主队)
        edge = (r.elo_h - mh) if mk else np.nan
        rows.append({
            "date": r.date, "home_team": r.home_team, "away_team": r.away_team,
            "elo_h": r.elo_h, "elo_d": r.elo_d, "elo_a": r.elo_a,
            "mkt_h": mh, "mkt_d": md, "mkt_a": ma, "edge_home": edge,
        })
    out = pd.DataFrame(rows)
    OUT.parent.mkdir(exist_ok=True)
    out.to_csv(OUT, index=False)

    matched = out.dropna(subset=["mkt_h"])
    if len(matched):
        print("=== 模型 vs 市场:主胜概率分歧最大的 8 场(|edge| 最大)===")
        top = matched.reindex(matched.edge_home.abs().sort_values(ascending=False).index).head(8)
        for r in top.itertuples(index=False):
            print(f"  {r.date.date()} {r.home_team} vs {r.away_team}: "
                  f"Elo主胜 {r.elo_h:.0%} | 市场 {r.mkt_h:.0%} | 差 {r.edge_home:+.0%}")
    print(f"\n完整结果已保存:{OUT}")


if __name__ == "__main__":
    main()
