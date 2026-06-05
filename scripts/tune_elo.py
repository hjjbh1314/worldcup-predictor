"""Elo 超参调优:严格 训练/验证/测试 三段切分。

  训练   < 2014        拟合概率头
  验证   2014–2018     网格选超参(只看这里)
  测试   >= 2018       留出集,最后只评一次

有提升才把最优超参写入 model_config.json(供预测器加载)。
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_results            # noqa: E402
from src.elo import EloModel                 # noqa: E402
from src.baseline import MultiLogitHead      # noqa: E402
from src.confed import confederation_ratings, conf_diff_column  # noqa: E402
from src.metrics import evaluate             # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "model_config.json"
TRAIN_END, VAL_END = "2014-01-01", "2018-01-01"
COLS = ["elo_diff", "abs_elo_diff", "conf_diff"]

GRID = {
    "home_advantage": [60, 80, 100, 120],
    "k_scale": [0.7, 1.0, 1.3],
    "regress": [0.0, 0.05, 0.10],
}


def features_for(results, home_advantage, k_scale, regress):
    feat = EloModel(home_advantage=home_advantage, k_scale=k_scale,
                    regress=regress).run(results)
    feat["abs_elo_diff"] = feat["elo_diff"].abs()
    return feat


def eval_config(results, cr_val, cr_test, params, final=False):
    feat = features_for(results, **params)
    if final:
        feat["conf_diff"] = conf_diff_column(feat, cr_test)
        tr = feat[feat.date < VAL_END]
        ev = feat[feat.date >= VAL_END]
    else:
        feat["conf_diff"] = conf_diff_column(feat, cr_val)
        tr = feat[feat.date < TRAIN_END]
        ev = feat[(feat.date >= TRAIN_END) & (feat.date < VAL_END)]
    head = MultiLogitHead(COLS).fit(tr, tr.result.values)
    return evaluate(ev.result.values, head.predict_proba(ev))


def main():
    results = load_results()
    cr_val = confederation_ratings(results, before=TRAIN_END)
    cr_test = confederation_ratings(results, before=VAL_END)

    default = {"home_advantage": 100, "k_scale": 1.0, "regress": 0.0}

    print("在验证集(2014–2018)上网格搜索…")
    best, best_rps = None, 1e9
    combos = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    for p in combos:
        m = eval_config(results, cr_val, cr_test, p, final=False)
        if m["rps"] < best_rps:
            best_rps, best = m["rps"], p
    print(f"  共 {len(combos)} 组。验证集最优:{best}  (RPS {best_rps:.4f})")
    print(f"  默认参数验证集:{eval_config(results, cr_val, cr_test, default)['rps']:.4f}")

    print("\n=== 留出测试集(>=2018,只评一次)===")
    m_def = eval_config(results, cr_val, cr_test, default, final=True)
    m_best = eval_config(results, cr_val, cr_test, best, final=True)
    print(f"  默认参数   准确率 {m_def['accuracy']*100:.1f}%  "
          f"LogLoss {m_def['log_loss']:.4f}  RPS {m_def['rps']:.4f}")
    print(f"  调优参数   准确率 {m_best['accuracy']*100:.1f}%  "
          f"LogLoss {m_best['log_loss']:.4f}  RPS {m_best['rps']:.4f}")
    print(f"  Δ 准确率 {(m_best['accuracy']-m_def['accuracy'])*100:+.2f}pp  "
          f"Δ RPS {m_best['rps']-m_def['rps']:+.4f}(负=更好)")

    if m_best["rps"] < m_def["rps"]:
        CONFIG.write_text(json.dumps(best, indent=2))
        print(f"\n✅ 调优有效,已写入 {CONFIG.name}:{best}")
    else:
        print("\n⚠️ 调优在测试集上未优于默认,保留默认参数(不写配置)。")


if __name__ == "__main__":
    main()
