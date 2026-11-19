"""GTA VI trophy earn-rate forecaster for make.fun competition
gta6-trophy-rarity-forecast-01.

Forecasts, for every trophy of Grand Theft Auto VI (PS5), its earn rate on
Exophase at the case's target, as five quantiles (levels 0.05, 0.25, 0.5,
0.75, 0.95) in percent.

Method (per trophy, deterministic, Python 3.13 standard library only):

1. Dedupe the history: consecutive reads whose whole rate vector is
   identical carry no new information, so only the first read of each
   identical run is kept.
2. Weighted least squares of logit(earn rate) on ln(1 + hours since the
   game's release), over the distinct reads that list the trophy.  The
   weight halves every 24 hours of age, so recent reads matter most.
   The slope is used exactly as fitted (unshrunk).
3. Read the fitted line at the target, then blend the median 15% toward
   the trophy's last-read rate (in logit space) so a flat launch-night
   stretch cannot drag the forecast far from where the page stands now.
   The median is capped at 99.0%.
4. Spread (logit units) = random-walk component (0.5 * sqrt(horizon in
   days)) + weighted fit-residual SD + a refresh-jump allowance
   (0.30 * sqrt(horizon in days + 1), for Exophase refreshes that move
   every rate at once), all times 0.8.
5. With fewer than 3 distinct reads for a trophy, fall back to the
   persistence forecast: median = last-read rate, spread = 0.5 *
   sqrt(horizon in days).
6. A trophy whose last rate is 100% stays near 100%: median 99.0 with a
   small spread (0.10).  Rates of 0% are handled by clamping every rate
   to 0.01..99.99 before taking the logit.
7. Quantiles: q_tau = inv_logit(logit(median) + z_tau * spread) with
   z = (-1.645, -0.674, 0, 0.674, 1.645), rounded to 4 decimals (half up).

I/O: reads every *.in in $MAKEFUN_INPUT_DIR (else /input/), in name
order, and writes /output/<id>.out ($MAKEFUN_OUTPUT_DIR else /output/):
{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}]}, one row per
trophy.  No network, no randomness: two runs on the same case produce
byte-identical output.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# The grader's -P leaves this file's directory off sys.path; put it back
# first so a sibling module would import (none is shipped, but harmless).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard normal quantiles at the scored levels.
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
# Random-walk volatility of a rate's logit per square-root day.
SIGMA_RW = 0.5
# Allowance (logit units) for one unseen Exophase refresh jump, scaled
# mildly with horizon: more days to the target, more refreshes can land.
JUMP_ALLOWANCE = 0.30
# The fitted spread is scaled down: the random walk, the residual and the
# jump allowance overlap, so their plain sum overstates the uncertainty.
SPREAD_SCALE = 0.8
# Fraction the fitted median is pulled toward the last-read rate.
BLEND_LAST = 0.15
# No median is ever forecast above this (percent).
MEDIAN_CAP = 99.0
# Recency half-life of a read's weight, in hours.
WEIGHT_HALF_LIFE_HOURS = 24.0
# Spread bounds (logit units).
SPREAD_MIN = 0.05
SPREAD_MAX = 2.5
# A trophy last read at 100% gets this median and spread.
FULL_MEDIAN = 99.0
FULL_SPREAD = 0.10


def parse_time(value):
    """Parse an ISO 8601 timestamp (or a bare YYYY-MM-DD date) to UTC."""
    text = str(value).strip()
    if len(text) == 10:
        text += "T00:00:00+00:00"
    moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def clamp_rate(pct):
    """Clamp a percent rate into (0, 100) so its logit is finite."""
    return min(99.99, max(0.01, float(pct)))


def logit(pct):
    p = clamp_rate(pct)
    return math.log(p / (100.0 - p))


def inv_logit_pct(y):
    """Inverse logit of y, as a percent in (0, 100)."""
    if y >= 0:
        return 100.0 / (1.0 + math.exp(-y))
    e = math.exp(y)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up (as JavaScript Math.round does)."""
    scaled = float(x) * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def weighted_line(xs, ys, ws):
    """Weighted least squares y = a + b*x. Returns (a, b, resid_sd)."""
    total_w = sum(ws)
    mean_x = sum(w * x for w, x in zip(ws, xs)) / total_w
    mean_y = sum(w * y for w, y in zip(ws, ys)) / total_w
    sxx = sum(w * (x - mean_x) ** 2 for w, x in zip(ws, xs))
    if sxx <= 0.0:
        slope = 0.0
    else:
        sxy = sum(w * (x - mean_x) * (y - mean_y)
                  for w, x, y in zip(ws, xs, ys))
        slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    resid_sq = sum(w * (y - (intercept + slope * x)) ** 2
                   for w, x, y in zip(ws, xs, ys))
    resid_sd = math.sqrt(resid_sq / total_w)
    return intercept, slope, resid_sd


def distinct_reads(history):
    """Keep the first read of each run of consecutive identical reads.

    Two reads are identical when their whole rate vectors match (None
    included), so a stretch where Exophase has not refreshed collapses
    to a single point instead of pretending to be many observations.
    """
    kept = []
    previous = None
    for read in history:
        rates = tuple(read.get("rates") or [])
        if rates != previous:
            kept.append(read)
            previous = rates
    return kept


def last_rate_for(history, index):
    """Most recent non-null rate for trophy `index`, or None."""
    for read in reversed(history):
        rates = read.get("rates") or []
        if index < len(rates) and rates[index] is not None:
            return float(rates[index])
    return None


def forecast_trophy(points, last_rate, horizon_days, x_target):
    """Return (median percent, logit spread) for one trophy.

    points: [(x, logit_rate, weight)] from the distinct reads that list
    this trophy, oldest first.
    """
    if last_rate is None:
        # No observation at all: a valid-shaped, maximally unsure row.
        return 1.0, 1.0
    if last_rate >= 99.995:
        # A trophy everyone tracked has earned stays at (near) 100%.
        return FULL_MEDIAN, FULL_SPREAD
    if len(points) < 3:
        # Too little movement to fit a line: persistence forecast.
        median = min(max(last_rate, 0.01), MEDIAN_CAP)
        spread = SIGMA_RW * math.sqrt(horizon_days)
        return median, spread
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ws = [p[2] for p in points]
    intercept, slope, resid_sd = weighted_line(xs, ys, ws)
    fitted_logit = intercept + slope * x_target
    # Blend the fitted median 15% toward the last-read rate, in logit
    # space, then cap the median at 99.0%.
    median_logit = (1.0 - BLEND_LAST) * fitted_logit + BLEND_LAST * logit(last_rate)
    median = inv_logit_pct(median_logit)
    median = min(max(median, 0.01), MEDIAN_CAP)
    rw = SIGMA_RW * math.sqrt(horizon_days)
    jump = JUMP_ALLOWANCE * math.sqrt(horizon_days + 1.0)
    spread = (rw + resid_sd + jump) * SPREAD_SCALE
    spread = min(max(spread, SPREAD_MIN), SPREAD_MAX)
    return median, spread


def forecast_case(case, name):
    trophies = case["trophies"]
    history = case.get("history") or []
    checkpoint = parse_time(case["at"])
    horizon_days = max(0.0, float(case.get("horizon_minutes", 0)) / 1440.0)
    if case.get("target"):
        target = parse_time(case["target"])
    else:
        target = checkpoint + timedelta(minutes=float(case.get("horizon_minutes", 0)))
    release_raw = (case.get("game") or {}).get("release_date", "2026-11-19")
    release = parse_time(release_raw)
    x_target = math.log1p(max(0.0, (target - release).total_seconds() / 3600.0))

    reads = distinct_reads(history)
    rows = []
    for index, trophy in enumerate(trophies):
        points = []
        for read in reads:
            rates = read.get("rates") or []
            if index >= len(rates) or rates[index] is None:
                continue
            read_at = parse_time(read["at"])
            hours = (read_at - release).total_seconds() / 3600.0
            x = math.log1p(max(0.0, hours))
            age_hours = max(0.0, (checkpoint - read_at).total_seconds() / 3600.0)
            weight = 0.5 ** (age_hours / WEIGHT_HALF_LIFE_HOURS)
            points.append((x, logit(rates[index]), weight))
        last_rate = last_rate_for(history, index)
        median, spread = forecast_trophy(points, last_rate, horizon_days, x_target)
        centre = logit(median)
        quantiles = []
        previous_q = None
        for z in Z:
            q = round4(inv_logit_pct(centre + z * spread))
            q = min(100.0, max(0.0, float(q)))
            if previous_q is not None and q < previous_q:
                q = previous_q
            quantiles.append(q)
            previous_q = q
        rows.append({"id": trophy["id"], "q": quantiles})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        out_path = OUTPUT_DIR / (path.stem + ".out")
        out_path.write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
