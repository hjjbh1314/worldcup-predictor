# ⚽ worldcup-predictor

A transparent, **leakage-free** match-outcome predictor for international football,
benchmarked on 49k matches (1872–2026) and pointed at the **2026 FIFA World Cup**.

It predicts **Win / Draw / Loss** probabilities for every fixture, with an Elo
engine as the core model and an honest, reproducible backtest comparing it against
both a naive baseline and a multi-feature gradient-boosting model.

> **TL;DR of the backtest (test period 2018→2026, 8,021 matches):**
> a well-tuned Elo model hits **60.0% accuracy / RPS 0.171**, and a gradient-boosting
> model with recent form, fatigue, fixture congestion and neutral-venue features adds
> **essentially nothing** on top of it. We show this result rather than hide it — see
> [Honest findings](#honest-findings).

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m scripts.download_data        # fetch the public dataset (~7 MB)
python -m scripts.run_backtest         # Elo baseline vs naive
python -m scripts.run_ml_backtest      # GB model vs Elo + feature importance
python -m scripts.predict_worldcup     # → outputs/wc2026_predictions.csv
python -m tests.test_sanity            # sanity + no-leakage tests

# optional: overlay live bookmaker odds (free tier, $0) — see docs/ODDS_GUIDE.md
export ODDS_API_KEY=...                 # free key from the-odds-api.com
python -m scripts.predict_with_odds    # Elo vs market, per match
```

### 中文速览
- `run_backtest`:Elo 基线 vs 朴素基线
- `run_ml_backtest`:梯度提升多特征模型 vs Elo,并打印特征重要性
- `predict_worldcup`:对 2026 世界杯赛程输出胜平负概率(存 CSV)
- 所有特征严格"赛前可知",`test_sanity` 含一条**截断重算**的因果/无泄漏硬测试

---

## Method

**Elo engine** (`src/elo.py`), following the [eloratings.net](https://www.eloratings.net/about)
methodology:

- update rule `R' = R + K · G · (W − We)`
- `We = 1 / (1 + 10^(−Δ/400))`, Δ = rating gap + home advantage (100, applied only to true home teams)
- tournament-importance weight `K` (World Cup finals 60, continental finals 50, qualifiers 40, friendlies 20, …)
- goal-difference multiplier `G` (bigger wins move ratings more)

Elo is an **online / causal** model: a rating at time *t* only uses matches before *t*,
so the pre-match rating gap is a leakage-free feature by construction.

**Probability head** (`src/baseline.py`): a thin multinomial logit on `[Δ, |Δ|]`
turns Elo's single expected-score number into proper 3-way W/D/L probabilities
(`|Δ|` lets draw probability peak when teams are evenly matched).

**Gradient-boosting model** (`src/ml.py`, `src/features.py`): `HistGradientBoostingClassifier`
over Elo ratings **plus** recent form (last 10), goals for/against, rest days,
fixture congestion (matches in last 14 days), neutral venue, and tournament weight.

**Metrics** (`src/metrics.py`): accuracy, log-loss, Brier, and **RPS** (Ranked
Probability Score — the standard for ordered football outcomes).

---

## Honest findings

| Model | Accuracy | LogLoss | Brier | RPS |
|---|---|---|---|---|
| Naive (class base rates) | 47.7% | 1.0512 | 0.6340 | 0.2283 |
| **Elo baseline** | **60.0%** | **0.8735** | 0.5137 | 0.1707 |
| Gradient boosting (+form/fatigue/congestion/venue) | 60.0% | 0.8732 | 0.5132 | 0.1705 |

Permutation importance is dominated by one feature — the Elo rating gap:

```
elo_diff         +0.3418      ← everything below is rounding error
ga_diff          +0.0097
form_diff        +0.0036
...
rest_diff/congestion/neutral  ≈ 0
```

**Why?** Elo already absorbs team strength, recent results and home advantage, so
re-feeding "recent form" or "fatigue" is largely redundant. The one lever that
genuinely beats Elo for match prediction is **bookmaker odds** (market information),
which this free-data project deliberately does not use. Treat that as the honest
ceiling of a pure history-based model.

---

## 2026 World Cup

The dataset already ships the 72 group-stage fixtures. Current Elo top of the field:

```
1. Spain 2224   2. Argentina 2187   3. France 2128
4. England 2088 5. Brazil 2066      6. Colombia 2059 ...
```

Run `python -m scripts.predict_worldcup` for the full per-match table
(`outputs/wc2026_predictions.csv`). Predictions are versioned so you can score them
against reality once matches are played — that's the point.

---

## Project layout

```
src/        data · elo · features · baseline · ml · metrics · backtest
scripts/    download_data · run_backtest · run_ml_backtest · predict_worldcup
tests/      test_sanity  (metrics + Elo + causality/no-leakage)
outputs/    wc2026_predictions.csv
```

## Data & credits

National-team results from [martj42/international_results](https://github.com/martj42/international_results)
(public domain). This project is for education and entertainment — not betting advice.

## License

MIT
