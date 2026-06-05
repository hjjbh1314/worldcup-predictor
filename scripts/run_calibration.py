"""校准回测:对比"未校准 vs Platt 校准"在留出测试集上的表现。

  概率头   训练 < 2016
  校准器   2016–2018(概率头未见过)
  测试     >= 2018(只评一次)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_results            # noqa: E402
from src.backtest import build_features      # noqa: E402
from src.baseline import MultiLogitHead      # noqa: E402
from src.calibrate import PlattCalibrator    # noqa: E402
from src.confed import confederation_ratings, conf_diff_column  # noqa: E402
from src.metrics import evaluate             # noqa: E402

COLS = ["elo_diff", "abs_elo_diff", "conf_diff"]


def main():
    results = load_results()
    feat = build_features(results)
    feat["abs_elo_diff"] = feat["elo_diff"].abs()
    cr = confederation_ratings(results, before="2018-01-01")
    feat["conf_diff"] = conf_diff_column(feat, cr)

    tr = feat[feat.date < "2016-01-01"]
    cal = feat[(feat.date >= "2016-01-01") & (feat.date < "2018-01-01")]
    te = feat[feat.date >= "2018-01-01"]

    head = MultiLogitHead(COLS).fit(tr, tr.result.values)
    raw = head.predict_proba(te)
    calib = PlattCalibrator().fit(head.predict_proba(cal), cal.result.values)
    cald = calib.predict_proba(raw)

    m_raw, m_cal = evaluate(te.result.values, raw), evaluate(te.result.values, cald)
    print(f"测试集 {len(te)} 场:")
    print(f"  未校准   准确率 {m_raw['accuracy']*100:.2f}%  LogLoss {m_raw['log_loss']:.4f}  "
          f"Brier {m_raw['brier']:.4f}  RPS {m_raw['rps']:.4f}")
    print(f"  已校准   准确率 {m_cal['accuracy']*100:.2f}%  LogLoss {m_cal['log_loss']:.4f}  "
          f"Brier {m_cal['brier']:.4f}  RPS {m_cal['rps']:.4f}")
    yh = (te.result.values == "H").mean()
    print(f"\n  主胜平均预测:未校准 {raw[:,0].mean():.3f} → 校准 {cald[:,0].mean():.3f} "
          f"(实际 {yh:.3f})")


if __name__ == "__main__":
    main()
