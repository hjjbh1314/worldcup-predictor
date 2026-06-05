"""ML 模型 vs Elo 基线:同一测试集公平对比 + 特征重要性。

用法:
    python -m scripts.run_ml_backtest
    python -m scripts.run_ml_backtest --split 2014-01-01
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.inspection import permutation_importance

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_results          # noqa: E402
from src.features import build_feature_table, FEATURE_COLS  # noqa: E402
from src.baseline import EloProbHead, BaseRateModel        # noqa: E402
from src.ml import GBModel                  # noqa: E402
from src.metrics import evaluate            # noqa: E402


def fmt(name: str, m: dict) -> str:
    return (f"  {name:<16} 准确率 {m['accuracy']*100:5.1f}%  "
            f"LogLoss {m['log_loss']:.4f}  Brier {m['brier']:.4f}  "
            f"RPS {m['rps']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="2018-01-01")
    args = ap.parse_args()

    results = load_results()
    print("构建特征表中(一次时间顺序遍历)…")
    feat = build_feature_table(results)
    train = feat[feat.date < args.split]
    test = feat[feat.date >= args.split]
    print(f"特征:{FEATURE_COLS}")
    print(f"切分点 {args.split}:训练 {len(train)} / 测试 {len(test)}\n")

    # —— 三个模型,同一测试集 ——
    naive = BaseRateModel().fit(train.result.values)
    naive_p = naive.predict_proba(len(test))

    elo_head = EloProbHead().fit(train.elo_diff.values, train.result.values)
    elo_p = elo_head.predict_proba(test.elo_diff.values)

    gb = GBModel().fit(train, train.result.values)
    gb_p = gb.predict_proba(test)

    m_naive = evaluate(test.result.values, naive_p)
    m_elo = evaluate(test.result.values, elo_p)
    m_gb = evaluate(test.result.values, gb_p)

    print("=== 测试集表现(准确率越高越好;LogLoss/Brier/RPS 越低越好) ===")
    print(fmt("朴素基线", m_naive))
    print(fmt("Elo 基线", m_elo))
    print(fmt("GB 多特征模型", m_gb))

    print(f"\nGB vs Elo:准确率 {(m_gb['accuracy']-m_elo['accuracy'])*100:+.2f} 个百分点,"
          f"RPS {m_gb['rps']-m_elo['rps']:+.4f}(负数=更好)")

    # —— 特征重要性(置换法,按 log_loss 评分)——
    print("\n计算特征重要性(置换法)…")
    r = permutation_importance(
        gb.clf, np.asarray(test[FEATURE_COLS], dtype=float), test.result.values,
        scoring="neg_log_loss", n_repeats=5, random_state=42, n_jobs=-1,
    )
    order = r.importances_mean.argsort()[::-1]
    print("=== 特征重要性(越大越关键)===")
    for i in order:
        print(f"  {FEATURE_COLS[i]:<16} {r.importances_mean[i]:+.4f} "
              f"± {r.importances_std[i]:.4f}")


if __name__ == "__main__":
    main()
