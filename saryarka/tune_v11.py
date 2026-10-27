"""Short possessions, step 2: sharper possession model, judged on held-out probabilities against v10.

v10 misses 125 of 159 short (<5 s) truth possessions per match, and on two thirds of them its P(home)
leans the wrong way (study_short.py). Its signals are mostly smoothed over 1-3 s and know only who
touched the ball last, so a 1 s interruption hardly moves them. This adds fast signals:

  * touch_next / touch_next_s: the team that touches next and how soon (touch_last's mirror image)
  * touch_weak_last / touch_weak_next: the same with a more sensitive touch detector (smaller kick,
    shorter lag) for the quick deflections the main one skips
  * near_gap_past / near_gap_future: the nearest-player gap over the half second before / after
  * close_gap: each team's closest approach to the ball within +/- half a second, away minus home
  * control_gap: a player of the team within CONTROL_M of a slow ball, smoothed over +/- 0.2 s

and trains two models on them: "plain" (as v9's possession model) and "short" (frames inside short
truth possessions count SHORT_WEIGHT times in training). P(dead) is v9's, from study_oof.py's cache
(dead-ball training is unchanged), and both are decoded exactly like v10.

    .venv/bin/python saryarka/tune_v11.py          # builds signals + out-of-fold P(home), ~1 h
    .venv/bin/python saryarka/study_short.py model # judges them against v10
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "2")

import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tune_v8 import CACHE, TEST_SEEDS, TRAIN_SEEDS, source_weights  # noqa: E402

# Params
PAST_S = 0.5              # one-sided window for near_gap_past / near_gap_future
CLOSE_S = 0.5             # +/- window for each team's closest approach
CONTROL_M = 1.5           # player this close to the ball ...
CONTROL_SPEED = 4.0       # ... while the ball moves slower than this (m/s) = control
CONTROL_SMOOTH_S = 0.2
WEAK_IMPULSE = 2.5        # sensitive touch detector: smaller kick ...
WEAK_LAG = 6              # ... with velocities 0.24 s apart instead of 0.48 s
SHORT_S = 5.0             # a truth possession shorter than this is short
SHORT_WEIGHT = 3.0        # how much a frame inside one counts in the "short" model
BUILD_WORKERS = 8
FOLD_WORKERS = 4
MODELS = ("plain", "short")
EXTRA = ("touch_next", "touch_next_s", "touch_weak_last", "touch_weak_next",
         "near_gap_past", "near_gap_future", "close_gap", "control_gap")


def cache_path(game, seed):
    return os.path.join(CACHE, "v11_{}_{}.npz".format(game, seed))


def oof_path(game, seed):
    return os.path.join(CACHE, "oof11_{}_{}.npz".format(game, seed))


def one_sided_mean(x, frames, past):
    """NaN-aware mean of x over the `frames` frames before (past=True) or after each frame."""
    ok = ~np.isnan(x)
    sums = np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0.0))))
    counts = np.concatenate(([0], np.cumsum(ok)))
    index = np.arange(len(x))
    if past:
        low, high = np.maximum(0, index - frames), index
    else:
        low, high = index + 1, np.minimum(len(x), index + 1 + frames)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (sums[high] - sums[low]) / (counts[high] - counts[low])


def rolling_min(x, half):
    padded = np.concatenate((np.full(half, np.inf), np.where(np.isnan(x), np.inf, x), np.full(half, np.inf)))
    out = np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1).min(axis=1)
    return np.where(np.isinf(out), np.nan, out)


def next_touch(touch_list, n, dead_spans):
    """touches.last_touch run backwards: the team that touches next, 0 if none before a dead span."""
    import touches
    flipped = [(n - 1 - f, t) for f, t in reversed(touch_list)]
    spans = [(n - stop, n - start) for start, stop in dead_spans]
    return touches.last_touch(flipped, n, spans)[::-1]


def extra_signals(match, f):
    """The fast signals above, from the home team's point of view."""
    import core
    import features
    import touches
    fps, n = match.fps, len(match.ball)
    spans = core.dead_spans(match, goals=True)
    main = touches.detect(match, core.TOUCH_IMPULSE)
    weak = touches.detect(match, WEAK_IMPULSE, lag=WEAK_LAG)
    out = {"touch_next": next_touch(main, n, spans)}
    touched = np.zeros(n, bool)
    touched[[frame for frame, _ in main]] = True
    out["touch_next_s"] = features.seconds_since(touched[::-1], fps)[::-1]
    out["touch_weak_last"] = touches.last_touch(weak, n, spans)
    out["touch_weak_next"] = next_touch(weak, n, spans)
    k = int(round(PAST_S * fps))
    out["near_gap_past"] = one_sided_mean(f["near_gap"], k, past=True)
    out["near_gap_future"] = one_sided_mean(f["near_gap"], k, past=False)
    half = int(round(CLOSE_S * fps))
    out["close_gap"] = rolling_min(f["near_away"], half) - rolling_min(f["near_home"], half)
    slow = np.nan_to_num(f["ball_speed"], nan=0.0) < CONTROL_SPEED
    control = (slow & (f["near_home"] < CONTROL_M)).astype(float) - (slow & (f["near_away"] < CONTROL_M))
    out["control_gap"] = features.centred_mean(control, int(round(CONTROL_SMOOTH_S * fps)))
    return out


def run_seconds(labels, fps):
    """Per frame, the length in seconds of the run of identical labels it sits in."""
    edges = np.flatnonzero(np.concatenate(([True], labels[1:] != labels[:-1], [True])))
    lengths = np.diff(edges)
    return np.repeat(lengths, lengths) / fps


def build(job):
    warnings.filterwarnings("ignore")
    import external
    import features
    import model
    import noise
    game, seed = job
    path = cache_path(game, seed)
    os.makedirs(CACHE, exist_ok=True)
    if os.path.exists(path):
        return path
    match, labels, restarts = external.load(game)
    variant = noise.noisy_match(match, labels, restarts, seed)[0]
    f = features.match_features(variant)
    f.update(extra_signals(variant, f))
    names = sorted(f)
    if seed in TRAIN_SEEDS:
        X, y = model.training_rows(f, labels, names)
        frames = np.flatnonzero(np.isin(labels, ["H", "A"]))[::model.STRIDE]
        short = run_seconds(labels, match.fps)[frames] < SHORT_S
        np.savez(path, X=X.astype(np.float32), y=y.astype(np.int8), short=np.concatenate([short, short]),
                 names=np.array(names))
    else:
        np.savez(path, F=model.matrix(f, names).astype(np.float32), names=np.array(names))
    print("  cached {} seed {}".format(game, seed), file=sys.stderr, flush=True)
    return path


def fold(held):
    warnings.filterwarnings("ignore")
    import external
    import model
    if all(os.path.exists(oof_path(held, s)) for s in TEST_SEEDS):
        return held
    others = [g for g in external.all_games() if g != held]
    jobs = [(g, s) for g in others for s in TRAIN_SEEDS]
    parts = [np.load(cache_path(g, s)) for g, s in jobs]
    names = list(parts[0]["names"])
    X = np.vstack([p["X"] for p in parts]).astype(float)
    y = np.concatenate([p["y"] for p in parts])
    short = np.concatenate([p["short"] for p in parts])
    sources = np.concatenate([np.full(len(p["y"]), g in external.GAMES) for (g, _), p in zip(jobs, parts)])
    base = source_weights(sources)
    fits = {"plain": model.fit(X, y, "boosting", base),
            "short": model.fit(X, y, "boosting", base * np.where(short, SHORT_WEIGHT, 1.0))}
    for seed in TEST_SEEDS:
        F = np.load(cache_path(held, seed))["F"].astype(float)
        f = {n: F[:, i] for i, n in enumerate(names)}
        np.savez(oof_path(held, seed), **{k: model.home_probability(m, names, f).astype(np.float32)
                                          for k, m in fits.items()})
    print("  oof {}".format(held), file=sys.stderr, flush=True)
    return held


def main():
    import multiprocessing
    import external
    games = external.all_games()
    ctx = multiprocessing.get_context("spawn")
    jobs = [(g, s) for g in games for s in TRAIN_SEEDS + TEST_SEEDS if not os.path.exists(cache_path(g, s))]
    if jobs:
        with ctx.Pool(BUILD_WORKERS) as pool:
            list(pool.imap_unordered(build, jobs))
    with ctx.Pool(FOLD_WORKERS) as pool:
        list(pool.imap_unordered(fold, games))
    print("cached in", CACHE)


if __name__ == "__main__":
    main()
