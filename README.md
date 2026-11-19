# gta6-trophy-rarity-forecast

A Python 3.13 standard-library-only forecaster for the [gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01) bounty: it forecasts every Grand Theft Auto VI (PS5) trophy's earn rate on Exophase at the case's target, as five quantiles (levels 0.05, 0.25, 0.5, 0.75, 0.95).

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/sw897636-web/gta6-trophy-rarity-forecast-v2/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/sw897636-web/gta6-trophy-rarity-forecast-v2/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

## Method

Per trophy, in `main.py`:

1. **Dedupe** the history: consecutive reads with an identical rate vector collapse to one point, so a stretch where Exophase has not refreshed does not pretend to be many observations.
2. **Weighted least squares** of logit(earn rate) on `ln(1 + hours since release)`, over the distinct reads listing the trophy. Weights halve every 24 hours of age. The slope is used exactly as fitted (unshrunk).
3. Read the fitted line at the target, then **blend the median 15% toward the last-read rate** (logit space). Median capped at 99.0%.
4. **Spread** (logit units) = random-walk component (`0.5 * sqrt(horizon in days)`) + weighted fit-residual SD + a refresh-jump allowance (`0.30 * sqrt(horizon in days + 1)`, for Exophase refreshes that move every rate at once), all times 0.8.
5. With **fewer than 3 distinct reads**, fall back to persistence: median = last-read rate, spread = `0.5 * sqrt(horizon in days)`.
6. A trophy last read at **100%** stays near 100% (median 99.0, spread 0.10). Rates are clamped to 0.01–99.99 before the logit, so 0% rates work too.
7. Quantiles: `q_tau = inv_logit(logit(median) + z_tau * spread)`, `z = (-1.645, -0.674, 0, 0.674, 1.645)`, rounded to 4 decimals (half up).

## Run

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` in the input directory (else `/input/`), writes `<id>.out` per case to the output directory (else `/output/`). Deterministic: two runs on the same case produce byte-identical output. No network, no third-party packages, well under the 8-second limit per case (about 0.05 s for both practice cases).

## Validation

On the kit's practice cases (p00, p01): both outputs are valid JSON with exactly 55 rows (ids 1–55, once each), quantiles non-decreasing within 0–100, 2,644 bytes each (limit 65,536). The two runs are byte-identical. The practice histories contain only one distinct read (launch night, pre-refresh), so the persistence fallback engages and the medians equal the last-read rates — e.g. trophy 4's row is `[0.8974, 1.012, 1.1, 1.1955, 1.3477]`, matching the house baseline's practice row. On a synthetic 6-read, 6,510-minute-horizon case the trend path engages: medians move with the fitted trend and the implied spread (1.47 logit units) is wider than the persistence spread (1.06).
