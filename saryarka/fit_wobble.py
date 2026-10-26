"""Fit the ball-position noise so noisy Metrica sits and wobbles like BAR-ATH (summaries only).

Grid over (drift, rate, smoothing); the score is the mean |log(simulated / BAR-ATH)| over the five
nearest-player distance quantiles and the four free-flight impulse p90s, on live frames, seed 0.

    .venv/bin/python saryarka/fit_wobble.py
"""

import itertools
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import external  # noqa: E402
import noise  # noqa: E402
import noise_stats  # noqa: E402

# Params
DRIFTS = (1.0, 1.25, 1.5, 1.75)
RATES = (3.0, 4.0, 5.0)
SMOOTHS = (0.15, 0.2, 0.3)


def as_vector(summary):
    return np.log(np.array(summary["dist_quantiles"] + summary["impulse_p90"]))


def fit():
    target = as_vector(noise_stats.position_stats())
    stats = noise_stats.load()
    games = [external.metrica_match(g) for g in external.GAMES]
    errors = {}
    for drift, rate, smooth_s in itertools.product(DRIFTS, RATES, SMOOTHS):
        sims = []
        for match, labels, restarts in games:
            noisy = noise.noisy_match(match, labels, restarts, 0, stats, drift, rate, smooth_s)[0]
            live = np.isin(labels, ["H", "A"])
            sims.append(as_vector(noise_stats.position_summary(noisy.ball, noisy.players_xy,
                                                               noisy.team_of_slot, noisy.fps, live)))
        sim = np.mean(sims, axis=0)
        errors[(drift, rate, smooth_s)] = float(np.mean(np.abs(sim - target)))
        print("drift {:<4} rate {:<4} smooth {:<4} error {:.3f} | dist {} | impulse {}".format(
            drift, rate, smooth_s, errors[(drift, rate, smooth_s)], " ".join("{:.2f}".format(x) for x in np.exp(sim[:5])),
            " ".join("{:.2f}".format(x) for x in np.exp(sim[5:]))))
    print("BAR-ATH                                 | dist {} | impulse {}".format(
        " ".join("{:.2f}".format(x) for x in np.exp(target[:5])), " ".join("{:.2f}".format(x) for x in np.exp(target[5:]))))
    best = min(errors, key=errors.get)
    print("best: drift {} rate {} smooth {} (error {:.3f})".format(*best, errors[best]))
    return best


if __name__ == "__main__":
    fit()
