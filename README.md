# ⚽ worldcup-predictor

A transparent, **leakage-free** match-outcome predictor for international football,
benchmarked on 49k matches (1872–2026) and pointed at the **2026 FIFA World Cup**.

It predicts **Win / Draw / Loss** probabilities for every fixture, with an Elo
engine as the core model and an honest, reproducible backtest comparing it against
both a naive baseline and a multi-feature gradient-boosting model.

### 📰 **[→ Live forecast (updated daily)](https://hjjbh1314.github.io/worldcup-predictor/)**

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
python -m scripts.run_confed_backtest  # confederation-adjusted Elo (cross-confed lift)
python -m scripts.tune_elo             # hyperparameter search (train/val/test)
python -m scripts.run_calibration      # probability calibration backtest
python -m scripts.run_dc_backtest      # Dixon-Coles vs calibrated Elo bake-off
python -m scripts.predict_worldcup     # → outputs/wc2026_predictions.csv + PREDICTIONS.md
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
genuinely beats Elo for match prediction is **bookmaker odds** (market information) —
see `scripts/predict_with_odds.py` and [docs/ODDS_GUIDE.md](docs/ODDS_GUIDE.md) for a
**free** way to overlay it.

### The one history-only feature that *does* help: confederation strength

Pure Elo treats each confederation as a closed points pool, so it over-rates strong
teams from weak regions and under-rates European/South-American sides — they only
truly meet at the World Cup. Learning each confederation's strength from
**inter-confederation matches** and feeding the gap into the probability head
(`src/confed.py`) gives a real, measured lift **where it matters** — and every World
Cup match is cross-confederation:

```
Learned strength (Elo pts):  UEFA +117  CONMEBOL +104  AFC +18  CONCACAF −27  CAF −40  OFC −171
```

| Test subset | Elo | + confederation |
|---|---|---|
| All matches (8,021) | 60.0% / RPS 0.1707 | 60.2% / RPS 0.1703 |
| **Cross-confederation (1,038)** | 56.8% / RPS 0.1867 | **58.3% / RPS 0.1837** |

### What moved the needle, and what didn't (honest log)

Every change below was judged on a **held-out** window, not the data it was tuned on:

- **Hyperparameter search** — home advantage, K-scale, yearly mean-reversion, grid-searched
  on a 2014–2018 validation window (`scripts/tune_elo.py`): the eloratings.net **defaults were
  already optimal**. No change made.
- **Probability calibration** — Platt scaling fit on recent held-out data
  (`scripts/run_calibration.py`): small but real, and it fixes a genuine bias. The raw model
  **over-predicted home wins** (mean home prob 0.510 vs actual 0.477 — stadiums emptied out in
  2020–21 and never fully recovered); calibration pulls it to 0.477 and improves every metric
  (RPS 0.1704 → 0.1698). **Kept — it's in the deployed model.**
- **Dixon-Coles goal model** — time-decayed Poisson with rolling yearly refits
  (`scripts/run_dc_backtest.py`): **58.4% / RPS 0.177, worse** than the calibrated Elo
  (60.5% / 0.169) on the same 7,741 matches. Internationals are too sparse for per-team
  attack/defence to beat a pooled rating; a 50/50 blend didn't help either. **Not used** for
  W/D/L — kept in `src/dixon_coles.py` for scoreline/xG output and reproducibility.

Final deployed model: **confederation-adjusted, calibrated Elo.** It still won't out-price the
market on team-specific quirks (e.g. New Zealand is stronger than the OFC average) — that's
exactly what odds are for.

---

## 2026 World Cup

The dataset already ships the 72 group-stage fixtures. Current Elo top of the field:

```
1. Spain 2224   2. Argentina 2187   3. France 2128
4. England 2088 5. Brazil 2066      6. Colombia 2059 ...
```

**📋 [Full match-by-match predictions → PREDICTIONS.md](PREDICTIONS.md)** —
regenerated daily during the tournament by a GitHub Action
(`.github/workflows/daily-predictions.yml`). As group matches are played, Elo updates
and the knockout-relevant predictions evolve. Predictions are versioned in git so you
can score them against reality once matches finish — that's the point.

---

## Project layout

```
src/        data · elo · features · baseline · ml · metrics · backtest
            confederations · confed · calibrate · dixon_coles · predict · odds
scripts/    download_data · run_backtest · run_ml_backtest · run_confed_backtest
            tune_elo · run_calibration · run_dc_backtest
            predict_worldcup · predict_with_odds
            build_odds_snapshot · score_predictions
tests/      test_sanity  (metrics + Elo + causality/no-leakage)
docs/       index.html (live site) · predictions.json · odds.json
            pretournament.json (locked) · scoreboard.json · og.png · ODDS_GUIDE.md
outputs/    wc2026_predictions.csv      PREDICTIONS.md (repo root)
```

The [live site](https://hjjbh1314.github.io/worldcup-predictor/) shows **model vs market**
bars (vig-removed consensus via the-odds-api.com), Dixon-Coles **scoreline/xG**, and a
**Reckoning** scoreboard that grades the *locked* pre-tournament forecast against real results
as matches are played. To refresh odds daily in CI, add the free key as a repo secret:

```bash
gh secret set ODDS_API_KEY --body "<your key>" --repo <owner>/worldcup-predictor
```

Without the secret everything still runs — the site simply shows model-only bars.

## Data & credits

National-team results from [martj42/international_results](https://github.com/martj42/international_results)
(public domain). This project is for education and entertainment — not betting advice.

## License

MIT
