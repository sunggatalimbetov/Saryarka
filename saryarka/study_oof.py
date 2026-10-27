"""Out-of-fold v9 probabilities on the 23 external matches, cached, so decoder studies are fast.

For each held-out match, v8's possession model and v9's dead-ball model are trained on the other
22 exactly as tune_v9.fold() does, and P(home) and P(dead) for every frame of the held-out match's
TEST_SEEDS noise draws are saved as oof_<game>_<seed>.npz in the signal cache. Any decoder change
(switch cost, X rule, dead spans, priors) can then be scored leave-one-match-out without
retraining. Training-side changes (new data, new noise) still need their own folds.

    .venv/bin/python saryarka/study_oof.py       # ~40 min on 8 cores the first time, then a no-op
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "2")

import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tune_v8 import CACHE, TEST_SEEDS, TRAIN_SEEDS, cache_path  # noqa: E402

BUILD_WORKERS = 8
FOLD_WORKERS = 4


def oof_path(game, seed):
    return os.path.join(CACHE, "oof_{}_{}.npz".format(game, seed))


def fold(held):
    warnings.filterwarnings("ignore")
    import deadball
    import external
    import model
    import tune_v9
    if all(os.path.exists(oof_path(held, s)) for s in TEST_SEEDS):
        return held
    others = [g for g in external.all_games() if g != held]
    home_fit, home_names = tune_v9.train(others, cache_path)
    dead_fit, dead_names = tune_v9.train(others, tune_v9.dead_path)
    for seed in TEST_SEEDS:
        F = np.load(cache_path(held, seed))["F"].astype(float)
        p_home = model.home_probability(home_fit, home_names, {n: F[:, i] for i, n in enumerate(home_names)})
        D = np.load(tune_v9.dead_path(held, seed))["F"].astype(float)
        p_dead = deadball.probability(dead_fit, dead_names, {n: D[:, i] for i, n in enumerate(dead_names)})
        np.savez(oof_path(held, seed), p_home=p_home.astype(np.float32), p_dead=p_dead.astype(np.float32))
    print("  oof {}".format(held), file=sys.stderr, flush=True)
    return held


def load(game, seed):
    z = np.load(oof_path(game, seed))
    return z["p_home"].astype(float), z["p_dead"].astype(float)


def main():
    import multiprocessing
    import external
    import tune_v8
    import tune_v9
    games = external.all_games()
    ctx = multiprocessing.get_context("spawn")
    for builder, path_of in ((tune_v9.build_dead, tune_v9.dead_path), (tune_v8.build, cache_path)):
        jobs = [(g, s) for g in games for s in TRAIN_SEEDS + TEST_SEEDS if not os.path.exists(path_of(g, s))]
        if jobs:
            with ctx.Pool(BUILD_WORKERS) as pool:
                list(pool.imap_unordered(builder, jobs))
    with ctx.Pool(FOLD_WORKERS) as pool:
        list(pool.imap_unordered(fold, games))
    print("cached in", CACHE)


if __name__ == "__main__":
    main()
