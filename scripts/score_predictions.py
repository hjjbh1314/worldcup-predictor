"""赛后对账:用真实结果给"锁定的赛前预测"打分。

只评 docs/pretournament.json(开赛前冻结、不再改动)里的预测,杜绝事后调参。
输出 docs/scoreboard.json 供页面展示战绩。开赛前(无已踢比赛)输出空战绩。
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_results       # noqa: E402
from src.metrics import evaluate, CLASSES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FROZEN = ROOT / "docs" / "pretournament.json"
OUT = ROOT / "docs" / "scoreboard.json"
PICK_NAME = {"H": "home", "D": "draw", "A": "away"}


def main():
    if not FROZEN.exists():
        print("⚠️ 无 pretournament.json,跳过。")
        return 0
    frozen = json.loads(FROZEN.read_text())["matches"]

    results = load_results()
    played = results[results.played]
    actual = {(str(r.date.date()), r.home_team, r.away_team): r.result
              for r in played.itertuples(index=False)}

    rows, y, proba = [], [], []
    for m in frozen:
        res = actual.get((m["date"], m["home"], m["away"]))
        if res is None:
            continue
        p = [m["ph"], m["pd"], m["pa"]]
        pick = CLASSES[int(np.argmax(p))]
        y.append(res); proba.append(p)
        rows.append({"date": m["date"], "home": m["home"], "away": m["away"],
                     "pick": PICK_NAME[pick], "actual": PICK_NAME[res],
                     "correct": bool(pick == res),
                     "p_pick": round(max(p), 3)})

    board = {"generated": str(date.today()), "n_total": len(frozen),
             "n_scored": len(y), "results": rows[-30:]}
    if y:
        m = evaluate(y, np.array(proba))
        board.update(accuracy=round(m["accuracy"], 4), rps=round(m["rps"], 4),
                     brier=round(m["brier"], 4), log_loss=round(m["log_loss"], 4))
    OUT.write_text(json.dumps(board, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ 战绩:已评 {len(y)}/{len(frozen)} 场"
          + (f",命中率 {board['accuracy']*100:.1f}%  RPS {board['rps']}" if y else "(尚无已踢比赛)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
