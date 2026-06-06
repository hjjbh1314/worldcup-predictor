"""拉取世界杯实时赔率,生成 docs/odds.json 供页面"模型 vs 市场"对照。

只输出去水后的共识胜平负概率(不转售原始盘口),并标注来源。
需要环境变量 ODDS_API_KEY;缺失或拉取失败则不改动现有文件(优雅降级)。
消耗:每次约 1 credit(h2h + 单地区)。
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_results            # noqa: E402
from src.predict import worldcup_fixtures    # noqa: E402
from src import odds as oddslib              # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "docs" / "odds.json"


def main():
    key = oddslib.get_api_key()
    if not key:
        print("⚠️ 未设置 ODDS_API_KEY,跳过(保留现有 odds.json)。")
        return 0
    try:
        events = oddslib.fetch_odds(key, sport_key="soccer_fifa_world_cup")
    except Exception as e:                                   # noqa: BLE001
        print(f"⚠️ 拉取赔率失败:{e}(保留现有 odds.json)。")
        return 0

    idx = oddslib.build_odds_index(events)
    results = load_results()
    fix = worldcup_fixtures(results)

    matches = []
    for r in fix.itertuples(index=False):
        mk = idx.get(frozenset([oddslib.canon(r.home_team), oddslib.canon(r.away_team)]))
        if not mk:
            continue
        ph = mk.get(oddslib.canon(r.home_team))
        pa = mk.get(oddslib.canon(r.away_team))
        pd_ = mk.get("Draw")
        if None in (ph, pd_, pa):
            continue
        matches.append({"home": r.home_team, "away": r.away_team,
                        "ph": round(ph, 4), "pd": round(pd_, 4), "pa": round(pa, 4)})

    payload = {"generated": str(date.today()), "source": "the-odds-api.com",
               "note": "Vig-removed consensus implied probabilities.",
               "matches": matches}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ 写出 {OUT.name}:{len(matches)} 场带市场概率(来源 the-odds-api.com)。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
