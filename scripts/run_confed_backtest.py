"""验证"洲际强度修正"是否真能改进纯 Elo。

对比 Elo 头 vs Elo+洲际修正头,分别在:
  - 全体测试集
  - 跨洲比赛子集(修正本应在这里起作用)
诚实记录结果——有效才集成。

用法:
    python -m scripts.run_confed_backtest --split 2018-01-01
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_results            # noqa: E402
from src.backtest import build_features      # noqa: E402
from src.baseline import MultiLogitHead      # noqa: E402
from src.confed import confederation_ratings, conf_diff_column  # noqa: E402
from src.confederations import get_confederation                # noqa: E402
from src.metrics import evaluate             # noqa: E402


def fmt(name, m):
    return (f"  {name:<22} n={m['n']:<5} 准确率 {m['accuracy']*100:5.1f}%  "
            f"LogLoss {m['log_loss']:.4f}  RPS {m['rps']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="2018-01-01")
    args = ap.parse_args()

    results = load_results()
    feat = build_features(results)
    feat["abs_elo_diff"] = feat["elo_diff"].abs()

    # 洲际强度只用 split 之前的比赛学习(防泄漏)
    cr = confederation_ratings(results, before=args.split)
    feat["conf_diff"] = conf_diff_column(feat, cr)
    feat["cross_conf"] = [
        get_confederation(h) is not None and get_confederation(a) is not None
        and get_confederation(h) != get_confederation(a)
        for h, a in zip(feat.home_team, feat.away_team)
    ]

    print("=== 学到的各大洲强度(去均值,Elo 分) ===")
    for c, v in sorted(cr.items(), key=lambda x: -x[1]):
        print(f"  {c:<10} {v:+6.0f}")

    train = feat[feat.date < args.split]
    test = feat[feat.date >= args.split]
    test_x = test[test.cross_conf]               # 跨洲子集

    v1 = MultiLogitHead(["elo_diff", "abs_elo_diff"]).fit(train, train.result.values)
    v2 = MultiLogitHead(["elo_diff", "abs_elo_diff", "conf_diff"]).fit(train, train.result.values)

    print(f"\n切分点 {args.split} | 训练 {len(train)} / 测试 {len(test)} "
          f"(其中跨洲 {len(test_x)})")
    print("\n=== 全体测试集 ===")
    print(fmt("Elo", evaluate(test.result.values, v1.predict_proba(test))))
    print(fmt("Elo + 洲际修正", evaluate(test.result.values, v2.predict_proba(test))))
    print("\n=== 跨洲比赛子集(修正应在此见效)===")
    m1 = evaluate(test_x.result.values, v1.predict_proba(test_x))
    m2 = evaluate(test_x.result.values, v2.predict_proba(test_x))
    print(fmt("Elo", m1))
    print(fmt("Elo + 洲际修正", m2))
    print(f"\n跨洲子集上:准确率 {(m2['accuracy']-m1['accuracy'])*100:+.2f} 个百分点,"
          f"RPS {m2['rps']-m1['rps']:+.4f}(负=更好)")


if __name__ == "__main__":
    main()
