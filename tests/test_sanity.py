"""轻量 sanity 测试:指标正确性、Elo 单调性、概率合法性、无泄漏迹象。

直接运行:  python -m tests.test_sanity
或用 pytest: pytest tests/
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import metrics                 # noqa: E402
from src.elo import EloModel            # noqa: E402
from src.data import load_results       # noqa: E402
from src.features import build_feature_table  # noqa: E402


def test_metrics_perfect_and_uniform():
    y = ["H", "D", "A"]
    perfect = np.eye(3)                  # 完美预测
    assert metrics.rps(y, perfect) == 0.0
    assert metrics.brier(y, perfect) == 0.0
    assert metrics.accuracy(y, perfect) == 1.0
    uniform = np.full((3, 3), 1 / 3)
    assert abs(metrics.log_loss(y, uniform) - np.log(3)) < 1e-9


def test_elo_monotonic():
    elo = EloModel()
    elo.ratings["Strong"] = 2000
    elo.ratings["Weak"] = 1500
    assert elo.expected("Strong", "Weak", neutral=True) > 0.5
    assert elo.expected("Weak", "Strong", neutral=True) < 0.5
    # 主场优势应抬高期望
    assert elo.expected("Weak", "Strong", neutral=False) > \
           elo.expected("Weak", "Strong", neutral=True)


def test_proba_valid():
    results = load_results()
    feat = build_feature_table(results)
    from src.baseline import EloProbHead
    head = EloProbHead().fit(feat.elo_diff.values, feat.result.values)
    p = head.predict_proba(feat.elo_diff.values[:100])
    assert np.allclose(p.sum(1), 1.0)
    assert (p >= 0).all()
    assert feat["date"].is_monotonic_increasing
    # 全历史第一场(1872,两队均无历史)状态/休息天数应为 NaN
    r0 = feat.iloc[0]
    assert np.isnan(r0["form_diff"]) and np.isnan(r0["rest_home"])


def test_no_leakage_causality():
    """因果性硬约束:用前 N 场重算特征,结果必须与全量算出的前 N 场逐一相等。

    若特征用到了未来信息,截断数据就会改变早期特征值,此处即会失败。
    """
    results = load_results()
    cols = ["elo_diff", "form_diff", "gf_diff", "ga_diff", "rest_home", "congestion_diff"]
    full = build_feature_table(results)
    sub = build_feature_table(results.iloc[:5000])
    a = full.head(len(sub))[cols].fillna(-9999).to_numpy()
    b = sub[cols].fillna(-9999).to_numpy()
    assert np.allclose(a, b), "检测到数据泄漏:截断后早期特征发生变化"


def _run_all():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  PASS  {name}")
    print("全部通过 ✅")


if __name__ == "__main__":
    _run_all()
