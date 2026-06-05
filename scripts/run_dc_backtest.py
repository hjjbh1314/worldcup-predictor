"""Dixon-Coles vs 校准版 Elo:同一批比赛上的公平胜平负对比。

DC 采用滚动逐年重拟合(每年初用此前数据拟合,预测该年比赛),贴近真实部署。
Elo 用与部署一致的"洲际修正 + 校准"管线。两者在相同比赛子集上评估。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_results            # noqa: E402
from src.backtest import build_features      # noqa: E402
from src.baseline import MultiLogitHead      # noqa: E402
from src.calibrate import PlattCalibrator    # noqa: E402
from src.confed import confederation_ratings, conf_diff_column  # noqa: E402
from src.dixon_coles import DixonColes       # noqa: E402
from src.metrics import evaluate, CLASSES    # noqa: E402

COLS = ["elo_diff", "abs_elo_diff", "conf_diff"]
YEARS = range(2018, 2026)


def main():
    results = load_results()

    # —— Elo(洲际修正 + 校准),与部署一致 ——
    feat = build_features(results); feat["abs_elo_diff"] = feat["elo_diff"].abs()
    cr = confederation_ratings(results, before="2018-01-01")
    feat["conf_diff"] = conf_diff_column(feat, cr)
    tr = feat[feat.date < "2016-01-01"]
    cal = feat[(feat.date >= "2016-01-01") & (feat.date < "2018-01-01")]
    head = MultiLogitHead(COLS).fit(tr, tr.result.values)
    calib = PlattCalibrator().fit(head.predict_proba(cal), cal.result.values)
    feat = feat.set_index(["date", "home_team", "away_team"])

    # —— DC 滚动逐年重拟合 ——
    models = {}
    for y in YEARS:
        print(f"  拟合 DC @ {y} …")
        models[y] = DixonColes().fit(results, ref_date=f"{y}-01-01")

    played = results[(results.played) & (results.date >= "2018-01-01")
                     & (results.date < "2026-01-01")]
    y_true, elo_p, dc_p = [], [], []
    for r in played.itertuples(index=False):
        dc = models[r.date.year]
        wdl = dc.predict_wdl(r.home_team, r.away_team, r.neutral)
        if wdl is None:
            continue
        try:
            frow = feat.loc[(r.date, r.home_team, r.away_team)]
        except KeyError:
            continue
        if hasattr(frow, "iloc") and frow.ndim > 1:
            frow = frow.iloc[0]
        import pandas as pd
        ep = calib.predict_proba(head.predict_proba(pd.DataFrame([frow])))[0]
        y_true.append(r.result); elo_p.append(ep); dc_p.append(wdl)

    elo_p, dc_p = np.array(elo_p), np.array(dc_p)
    blend = 0.5 * elo_p + 0.5 * dc_p
    print(f"\n对比子集 {len(y_true)} 场(2018–2025):")
    for name, p in [("校准版 Elo", elo_p), ("Dixon-Coles", dc_p), ("两者各半融合", blend)]:
        m = evaluate(y_true, p)
        print(f"  {name:<14} 准确率 {m['accuracy']*100:.2f}%  "
              f"LogLoss {m['log_loss']:.4f}  Brier {m['brier']:.4f}  RPS {m['rps']:.4f}")


if __name__ == "__main__":
    main()
