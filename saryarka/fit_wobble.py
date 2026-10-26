"""Fit the ball-position noise so noisy Metrica sits and wobbles like BAR-ATH (summaries only).

Grid over (blur, drift, rate, smoothing); the score is the mean |log(simulated / BAR-ATH)| over the
five nearest-player distance quantiles, the four free-flight impulse p90s and the four near-player
impulse p99s (where kicks show), on live frames, seed 0.

    .venv/bin/python saryarka/fit_wobble.py                         # Metrica (locked 26 Oct)
    .venv/bin/python saryarka/fit_wobble.py --source skillcorner    # SkillCorner's own profile

SkillCorner's ball already went through a broadcast tracker, so it needs its own blur and drift:
the target is the same BAR-ATH summary, the starting ball is different.
"""

import argparse

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
GRIDS = {"metrica": (BLURS, DRIFTS), "skillcorner": ((0.0, 0.1, 0.2, 0.3), (0.25, 0.5, 0.75, 1.0))}
SKILLCORNER_FIT_MATCHES = 10  # every other SkillCorner match: enough for a summary, half the time


def as_vector(summary):
    return np.log(np.array(summary["dist_quantiles"] + summary["impulse_p90"] + summary["near_impulse_p99"]))


def show(vector):
    values = np.exp(vector)
    parts = (values[:5], values[5:9], values[9:])
    return "dist {} | free {} | near {}".format(*(" ".join("{:.2f}".format(x) for x in part) for part in parts))


def simulate(args):
    """Mean summary vector over the given games for one setting (a worker process)."""
    games, key = args
    blur_s, drift, rate, smooth_s = key
    stats = noise_stats.load()
    sims = []
    for game in games:
        match, labels, restarts = external.load(game)
        noisy = noise.noisy_match(match, labels, restarts, 0, stats, drift, rate, smooth_s, blur_s)[0]
        live = np.isin(labels, ["H", "A"])
        sims.append(as_vector(noise_stats.position_summary(noisy.ball, noisy.players_xy,
                                                           noisy.team_of_slot, noisy.fps, live)))
    return key, np.mean(sims, axis=0)


def fit(source="metrica"):
    import multiprocessing
    target = as_vector(noise_stats.position_stats())
    if source == "metrica":
        games = list(external.GAMES)
    else:
        games = [g for g in external.all_games() if g not in external.GAMES][::2][:SKILLCORNER_FIT_MATCHES]
    blurs, drifts = GRIDS[source]
    keys = list(itertools.product(blurs, drifts, RATES, SMOOTHS))
    errors = {}
    with multiprocessing.get_context("spawn").Pool(max(1, (os.cpu_count() or 2) - 2)) as pool:
        for key, sim in pool.imap(simulate, [(games, key) for key in keys]):
            errors[key] = float(np.mean(np.abs(sim - target)))
            print("blur {:<4} drift {:<4} rate {:<4} smooth {:<4} error {:.3f} | {}".format(*key, errors[key], show(sim)),
                  flush=True)
    print("BAR-ATH{}| {}".format(" " * 47, show(target)))
    best = min(errors, key=errors.get)
    print("best: blur {} drift {} rate {} smooth {} (error {:.3f})".format(*best, errors[best]))
    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", choices=("metrica", "skillcorner"), default="metrica")
    fit(parser.parse_args().source)
