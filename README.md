# Pre-release-signal kernel forecaster v2

[![make.fun](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![Rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/fansevanoo/gta6-twitch-ccu-forecast-01-entry-acct3-v2/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![Reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/fansevanoo/gta6-twitch-ccu-forecast-01-entry-acct3-v2/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

Entry for the GTA VI launch-day Twitch viewer forecast (hours 2-24, five
quantiles per hour, pinball loss on the log scale).

## Model

A kernel-weighted Gaussian log-ratio forecaster, Python standard library
only. Each past big launch contributes its day curve as log-ratios
against its own hour-1 print. Donors are weighted by tier, platform
class, launch size (log hour-1), and pre-release buzz, then the top 16
give a weighted mean and standard deviation per hour. Two extra
pre-release signals apply to console targets: the audience already
waiting in the category in the last hour before release day, and the UTC
hour at which that audience peaks over the previous 7 days.

## What changed in v2

The v1 entry scored practice loss 0.217 with 90% coverage of only 0.685
— its intervals were far too narrow. v2 keeps the same centre (median)
and widens only the spread: the final standard deviation is multiplied
by 1.25, chosen so leave-one-out coverage_90 on console-like launches
lands near 0.95 (CONSOLE 0.940, practice-like 0.963) while the
leave-one-out loss barely moves (practice-like 0.140 vs 0.137 at 1.0x).

## Run

`python3 main.py` reads `$MAKEFUN_INPUT_DIR` (else `/input`) and writes
`$MAKEFUN_OUTPUT_DIR` (else `/output`). One case takes well under a
second; output is deterministic.
