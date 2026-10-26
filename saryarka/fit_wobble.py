"""Fit the ball-position noise so noisy Metrica sits and wobbles like BAR-ATH (summaries only).

Grid over (blur, drift, rate, smoothing); the score is the mean |log(simulated / BAR-ATH)| over the
five nearest-player distance quantiles, the four free-flight impulse p90s and the four near-player
impulse p99s (where kicks show), on live frames, seed 0.

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
DRIFTS = (0.5, 0.75, 1.0)
RATES = (4.0,)
SMOOTHS = (0.2,)
BLURS = (0.2, 0.3, 0.4)  # centred (noise.BLUR_CENTRED); causal fitted worse (0.21 vs 0.14)


def as_vector(summary):
    return np.log(np.array(summary["dist_quantiles"] + summary["impulse_p90"] + summary["near_impulse_p99"]))


def show(vector):
    values = np.exp(vector)
    parts = (values[:5], values[5:9], values[9:])
    return "dist {} | free {} | near {}".format(*(" ".join("{:.2f}".format(x) for x in part) for part in parts))


def fit():
    target = as_vector(noise_stats.position_stats())
    stats = noise_stats.load()
    games = [external.metrica_match(g) for g in external.GAMES]
    errors = {}
    for blur_s, drift, rate, smooth_s in itertools.product(BLURS, DRIFTS, RATES, SMOOTHS):
        sims = []
        for match, labels, restarts in games:
            noisy = noise.noisy_match(match, labels, restarts, 0, stats, drift, rate, smooth_s, blur_s)[0]
            live = np.isin(labels, ["H", "A"])
            sims.append(as_vector(noise_stats.position_summary(noisy.ball, noisy.players_xy,
                                                               noisy.team_of_slot, noisy.fps, live)))
        sim = np.mean(sims, axis=0)
        key = (blur_s, drift, rate, smooth_s)
        errors[key] = float(np.mean(np.abs(sim - target)))
        print("blur {:<4} drift {:<4} rate {:<4} smooth {:<4} error {:.3f} | {}".format(*key, errors[key], show(sim)),
              flush=True)
    print("BAR-ATH{}| {}".format(" " * 47, show(target)))
    best = min(errors, key=errors.get)
    print("best: blur {} drift {} rate {} smooth {} (error {:.3f})".format(*best, errors[best]))
    return best


if __name__ == "__main__":
    fit()
