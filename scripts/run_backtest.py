"""运行 Elo 基线回测,打印对比表格。

用法:
    python -m scripts.run_backtest                 # 默认 2018 切分
    python -m scripts.run_backtest --split 2014-01-01
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_results          # noqa: E402
from src.backtest import time_split_backtest  # noqa: E402


def fmt(name: str, m: dict) -> str:
    return (f"  {name:<18} 准确率 {m['accuracy']*100:5.1f}%  "
            f"LogLoss {m['log_loss']:.4f}  Brier {m['brier']:.4f}  "
            f"RPS {m['rps']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="2018-01-01",
                    help="训练/测试时间切分点 (YYYY-MM-DD)")
    ap.add_argument("--no-goal-diff", action="store_true",
                    help="关闭净胜球修正")
    args = ap.parse_args()

    results = load_results()
    played = results[results.played]
    print(f"数据:{len(results)} 场,其中已踢 {len(played)} 场 "
          f"({played.date.min().date()} → {played.date.max().date()})\n")

    r = time_split_backtest(results, split=args.split,
                            use_goal_diff=not args.no_goal_diff)

    print(f"切分点 {r['split']}:训练 {r['n_train']} 场 / 测试 {r['n_test']} 场")
    print(f"测试集胜平负实际占比 "
          f"{r['test'].result.value_counts(normalize=True).round(3).to_dict()}\n")
    print("=== 测试集表现(指标越低越好,准确率越高越好) ===")
    print(fmt("朴素基线(频率)", r["naive"]))
    print(fmt("Elo 基线", r["elo"]))

    d_acc = (r["elo"]["accuracy"] - r["naive"]["accuracy"]) * 100
    d_rps = r["naive"]["rps"] - r["elo"]["rps"]
    print(f"\nElo 相对朴素基线:准确率 +{d_acc:.1f} 个百分点,RPS 降低 {d_rps:.4f}")


if __name__ == "__main__":
    main()
