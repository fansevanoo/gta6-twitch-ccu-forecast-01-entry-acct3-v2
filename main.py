"""Forecast GTA VI launch-day Twitch viewers, hours 2-24.

Kernel-weighted Gaussian log-ratio forecaster (Python 3.13 stdlib only,
deterministic, no network).

Method, in short: each past big launch (a "donor") gives a day curve as
log-ratios l_h = ln(viewers_h / viewers_1). For the target case we weight
the donors with a kernel (tier x platform-class x log hour-1 size x
pre-release buzz), take the top 16, and per hour use the kernel-weighted
mean and standard deviation of the log-ratios. Two extra pre-release
signals apply to console targets only: the audience waiting in the
category in the last hour before the release day, and the UTC hour at
which that audience peaks over the previous 7 days (a proxy for fanbase
geography, which drives the shape through the rolling-midnight unlock).

The spread is deliberately widened: the previous version of this entry
scored practice coverage_90 of only 0.685 (intervals far too narrow),
so the final standard deviation is multiplied by SPREAD_MULT = 1.25,
chosen so leave-one-out coverage_90 on console-like launches lands near
0.95 while the leave-one-out loss barely moves. The centre (median) is
unchanged by this revision; only the spread is wider.

Reads $MAKEFUN_INPUT_DIR (else /input) *.in files, writes
$MAKEFUN_OUTPUT_DIR (else /output) <id>.out files.
"""

import json
import math
import os
import re
from datetime import datetime, timedelta

# Standard-normal quantiles for the five forecast levels.
Z_SCORES = [-1.6448536269514722, -0.6744897501960817, 0.0,
            0.6744897501960817, 1.6448536269514722]
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# Kernel bandwidths (log units, except the peak-hour one which is hours).
BW_LOG_H1 = 1.2
BW_LOG_BUZZ = 1.3
BW_LAST_HOUR = 1.0
BW_PEAK_HOUR = 5.0

# Donor tier weights, read from each comparable's unlock text.
TIER_WEIGHTS = {
    "comparable": 1.0,
    "early-access-contaminated": 0.7,
    "weak": 0.5,
}
PLATFORM_MISMATCH = 0.2
TOP_K = 16

# Spread: remap the kernel-weighted SD, scale by target class, then widen.
SPREAD_MULT = 1.25


def parse_time(text):
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ")


def gaussian(x, bandwidth):
    return math.exp(-0.5 * (x / bandwidth) ** 2)


def tier_of(unlock_text):
    """Tier weight of a donor, from the 'Tier: ...' note in its unlock text."""
    match = re.search(r"Tier:\s*([a-z\-]+)", unlock_text or "")
    name = match.group(1) if match else "comparable"
    return TIER_WEIGHTS.get(name, 1.0), name


def console_from_platforms(platforms):
    """A launch is console-only when PC is not among its platforms."""
    return "PC" not in (platforms or "")


def console_from_notes(notes):
    """Target console-ness from the case notes' 'released ... on ...'
    clause. Stops at '(' so 'RDR2 (console only; PC 2019-11)' stays
    console."""
    match = re.search(r"released\s+\S+\s+on\s+([^.;(]+)", notes or "")
    if not match:
        return True
    return "PC" not in match.group(1)


def mean_log_by_utc_hour(rows):
    """Mean of log viewers per UTC hour-of-day, over the given rows."""
    by_hour = {}
    for row in rows:
        viewers = row.get("viewers")
        if viewers is None or viewers <= 0:
            continue
        hour = parse_time(row["from"]).hour
        by_hour.setdefault(hour, []).append(math.log(viewers))
    return {h: sum(v) / len(v) for h, v in by_hour.items()}


class Donor:
    """One past launch usable as a donor: its day curve as log-ratios
    plus the pre-release features the kernel compares."""

    def __init__(self, comp):
        self.name = comp.get("name", "")
        self.day_start = parse_time(comp["day_start"])
        self.console = console_from_platforms(comp.get("platforms", ""))
        self.tier_weight, self.tier = tier_of(comp.get("unlock", ""))
        hourly = sorted(comp.get("hourly", []), key=lambda r: r["from"])
        self.hourly = hourly
        day = [r for r in hourly
               if self.day_start <= parse_time(r["from"])
               < self.day_start + timedelta(hours=24)]
        values = [r["viewers"] for r in day]
        self.usable = (len(day) == 24
                       and all(v is not None and v > 0 for v in values))
        if not self.usable:
            return
        self.p1 = values[0]
        self.log_ratios = [math.log(v / self.p1) for v in values]
        pre = [r for r in hourly if parse_time(r["from"]) < self.day_start]
        window24 = [r["viewers"] for r in pre
                    if parse_time(r["from"]) >= self.day_start - timedelta(hours=24)]
        self.buzz24 = (sum(window24) / len(window24)) if window24 else None
        last = [r for r in pre
                if parse_time(r["from"]) == self.day_start - timedelta(hours=1)]
        if last:
            self.last_hour = last[0]["viewers"]
        elif pre and (self.day_start - parse_time(pre[-1]["from"])) <= timedelta(hours=12):
            self.last_hour = pre[-1]["viewers"]
        else:
            self.last_hour = None
        week = [r for r in pre
                if parse_time(r["from"]) >= self.day_start - timedelta(days=7)]
        means = mean_log_by_utc_hour(week)
        self.peak_hour = (max(means, key=lambda h: means[h])
                          if means else None)


class Target:
    """The case being forecast: hour-1 print plus pre-release features
    taken from its own history (never from any other case)."""

    def __init__(self, case):
        self.p1 = case["known"][0]["viewers"]
        self.console = console_from_notes(case.get("notes", ""))
        day_start = parse_time(case["day_start"])
        history = sorted(case.get("history_hourly", []),
                         key=lambda r: r["from"])
        pre = [r for r in history if parse_time(r["from"]) < day_start]
        window24 = [r["viewers"] for r in pre
                    if parse_time(r["from"]) >= day_start - timedelta(hours=24)]
        self.buzz24 = (sum(window24) / len(window24)) if window24 else None
        last = [r for r in pre
                if parse_time(r["from"]) == day_start - timedelta(hours=1)]
        if last:
            self.last_hour = last[0]["viewers"]
        elif pre and (day_start - parse_time(pre[-1]["from"])) <= timedelta(hours=12):
            self.last_hour = pre[-1]["viewers"]
        else:
            self.last_hour = None
        week = [r for r in pre
                if parse_time(r["from"]) >= day_start - timedelta(days=7)]
        means = mean_log_by_utc_hour(week)
        self.peak_hour = (max(means, key=lambda h: means[h])
                          if means else None)


def kernel_weight(donor, target):
    """How much one donor counts for this target."""
    weight = donor.tier_weight
    if donor.console != target.console:
        weight *= PLATFORM_MISMATCH
    weight *= gaussian(math.log(donor.p1) - math.log(target.p1), BW_LOG_H1)
    if donor.buzz24 and target.buzz24:
        weight *= gaussian(
            math.log(donor.buzz24 / donor.p1) - math.log(target.buzz24 / target.p1),
            BW_LOG_BUZZ)
    if target.console:
        if donor.last_hour and target.last_hour:
            weight *= gaussian(
                math.log(donor.last_hour / donor.p1)
                - math.log(target.last_hour / target.p1),
                BW_LAST_HOUR)
        if donor.peak_hour is not None and target.peak_hour is not None:
            gap = abs(donor.peak_hour - target.peak_hour)
            gap = min(gap, 24 - gap)
            weight *= gaussian(gap, BW_PEAK_HOUR)
    return weight


def forecast_hours(target, donors):
    """Forecast rows for hours 2..24: kernel-weighted mean/SD of the
    donors' log-ratios, spread remapped and widened, back in viewer
    counts around the hour-1 print."""
    scored = []
    for donor in donors:
        weight = kernel_weight(donor, target)
        if weight > 0:
            scored.append((weight, donor))
    scored.sort(key=lambda item: -item[0])
    top = scored[:TOP_K]
    if not top:
        # No usable donor: fall back to the house baseline's shape.
        return [{"hour": h,
                 "q": [max(target.p1 * math.exp(z * 0.15 * math.sqrt(h - 1)), 1.0)
                       for z in Z_SCORES]}
                for h in range(2, 25)]
    total_weight = sum(w for w, _ in top)
    rows = []
    for hour in range(2, 25):
        values = [d.log_ratios[hour - 1] for _, d in top]
        weights = [w for w, _ in top]
        mean = sum(w * v for w, v in zip(weights, values)) / total_weight
        var = sum(w * (v - mean) ** 2
                  for w, v in zip(weights, values)) / total_weight
        spread = math.sqrt(max(var, 0.0))
        spread = (0.5 * spread + 0.15) * (0.65 if target.console else 0.9)
        spread *= SPREAD_MULT
        quantiles = []
        previous = 0.0
        for z in Z_SCORES:
            q = target.p1 * math.exp(mean + z * spread)
            q = min(max(q, 1.0), 1e9)
            q = max(q, previous)
            quantiles.append(q)
            previous = q
        rows.append({"hour": hour, "q": quantiles})
    return rows


def forecast_case(case):
    donors = []
    for comp in case.get("comparables", []):
        if "target-history" in comp.get("unlock", ""):
            continue
        try:
            donor = Donor(comp)
        except (KeyError, ValueError, TypeError):
            continue
        if donor.usable:
            donors.append(donor)
    target = Target(case)
    return {"hours": forecast_hours(target, donors)}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(output_dir, exist_ok=True)
    for name in sorted(os.listdir(input_dir)):
        if not name.endswith(".in"):
            continue
        with open(os.path.join(input_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        result = forecast_case(case)
        out_name = name[:-3] + ".out"
        with open(os.path.join(output_dir, out_name), "w", encoding="utf-8") as fh:
            json.dump(result, fh)


if __name__ == "__main__":
    main()
