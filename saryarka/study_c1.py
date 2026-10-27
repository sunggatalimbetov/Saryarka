"""C1: DFL's 7 matches (study_dfl.py) as extra training data for v9's dead-ball model only.

DFL's BallStatus is a per-frame alive/dead flag from the league's own data, a cleaner dead-ball
answer than Metrica's events or SkillCorner's interruptions. Its BallPossession changes team on
every duel (37% of possessions under 2 s, mean 6.7 s against the organisers' ~14 s), a different
convention from the one we are scored on, so DFL does not train the possession model and is not
scored: the judge stays the 23 harness matches under the agreed rule, against v9's held-out
probabilities (study_oof.py). The dead-ball setting stays v9's (model.V9_DEAD), not re-tuned.

DFL's noise profile: restart miss rates from its own event counts, blur and drift fitted like
SkillCorner's (fit), dead ball blanked (the optical tracker keeps the ball when it is out).

    .venv/bin/python saryarka/study_c1.py fit        # DFL blur and drift against BAR-ATH summaries
    .venv/bin/python saryarka/study_c1.py score      # build DFL dead-ball rows, refit folds, judge
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "2")

import argparse
import itertools
import json
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import external  # noqa: E402
import noise  # noqa: E402
import study_dfl  # noqa: E402
from tune_v8 import CACHE, TEST_SEEDS, TRAIN_SEEDS  # noqa: E402

# Params
DFL_BLUR_S = 0.3       # `fit`, 27 Oct: blur 0.3 drift 0.75, error 0.157 (Metrica's profile too)
DFL_DRIFT = 0.75
FIT_GRID = ((0.0, 0.1, 0.2, 0.3), (0.25, 0.5, 0.75, 1.0))
FIT_MATCHES = 4
DFL_GAMES = tuple("dfl" + g for g in study_dfl.GAMES)
FOLD_WORKERS = 4
BUILD_WORKERS = 7

_load, _trust, _profile = external.load, external.dead_trust, noise.profile


def dfl_events():
    """Mean restarts per match by type over the 7 DFL matches, as noise.PROFILES wants."""
    counts = {}
    for g in study_dfl.GAMES:
        _, _, rs = study_dfl.dfl_match(g)
        for _, kind, _, _ in rs:
            counts[kind] = counts.get(kind, 0) + 1
    return {k: counts.get(k, 0) / len(study_dfl.GAMES) for k in noise.BAR_ATH_TAGS}


def load(game):
    if str(game).startswith("dfl"):
        return study_dfl.dfl_match(str(game)[3:])
    return _load(game)


def dead_trust(game):
    if str(game).startswith("dfl"):
        return np.ones(len(load(game)[1]), bool)
    return _trust(game)


def profile(match):
    if match.name.startswith("dfl"):
        return {"events": PROFILE["events"], "blur_s": DFL_BLUR_S, "drift": DFL_DRIFT, "blank_dead": True}
    return _profile(match)


PROFILE = {}
external.load, external.dead_trust, noise.profile = load, dead_trust, profile


def setup():
    os.makedirs(study_dfl.CACHE, exist_ok=True)
    path = os.path.join(study_dfl.CACHE, "events.json")
    if not os.path.exists(path):
        with open(path, "w") as handle:
            json.dump(dfl_events(), handle)
    with open(path) as handle:
        PROFILE["events"] = json.load(handle)


def simulate(args):
    setup()
    import fit_wobble
    import noise_stats
    games, (blur_s, drift) = args
    stats = noise_stats.load()
    sims = []
    for game in games:
        match, labels, restarts = load(game)
        noisy = noise.noisy_match(match, labels, restarts, 0, stats, drift, noise.WOBBLE_RATE, noise.WOBBLE_SMOOTH_S, blur_s)[0]
        live = np.isin(labels, ["H", "A"])
        sims.append(fit_wobble.as_vector(noise_stats.position_summary(noisy.ball, noisy.players_xy,
                                                                      noisy.team_of_slot, noisy.fps, live)))
    return (blur_s, drift), np.mean(sims, axis=0)


def fit(pool):
    import fit_wobble
    import noise_stats
    target = fit_wobble.as_vector(noise_stats.position_stats())
    games = list(DFL_GAMES[::2][:FIT_MATCHES])
    errors = {}
    for key, sim in pool.imap(simulate, [(games, k) for k in itertools.product(*FIT_GRID)]):
        errors[key] = float(np.mean(np.abs(sim - target)))
        print("blur {:<4} drift {:<4} error {:.3f}".format(*key, errors[key]), flush=True)
    best = min(errors, key=errors.get)
    print("best: blur {} drift {} (error {:.3f})".format(*best, errors[best]))


def build(job):
    setup()
    import tune_v9
    return tune_v9.build_dead(job)


def train_dead(others):
    """tune_v9.train for the dead-ball model, three sources weighted equally."""
    import model
    import tune_v8
    import tune_v9
    jobs = [(g, s) for g in others for s in TRAIN_SEEDS]
    parts = [np.load(tune_v9.dead_path(g, s)) for g, s in jobs]
    kind = lambda g: "dfl" if str(g).startswith("dfl") else ("metrica" if g in external.GAMES else "skillcorner")  # noqa: E731
    sources = np.concatenate([np.full(len(p["y"]), kind(g)) for (g, _), p in zip(jobs, parts)])
    X = np.vstack([p["X"] for p in parts]).astype(float)
    y = np.concatenate([p["y"] for p in parts])
    return model.fit(X, y, "boosting", tune_v8.source_weights(sources)), list(parts[0]["names"])


def c1_path(game, seed):
    return os.path.join(CACHE, "c1dead_{}_{}.npy".format(game, seed))


def fold(held):
    """Held-out P(dead) for one of the 23 from a dead-ball model that also saw DFL. Judged with the
    v9 + A2 decoder (study_decode.v9_labels' defaults) against the same decoder on v9's P(dead)."""
    warnings.filterwarnings("ignore")
    setup()
    import deadball
    import tune_v9
    if all(os.path.exists(c1_path(held, s)) for s in TEST_SEEDS):
        return held
    others = [g for g in external.all_games() if g != held] + list(DFL_GAMES)
    fit_, names = train_dead(others)
    for seed in TEST_SEEDS:
        D = np.load(tune_v9.dead_path(held, seed))["F"].astype(float)
        np.save(c1_path(held, seed), deadball.probability(fit_, names, {n: D[:, i] for i, n in enumerate(names)}).astype(np.float32))
    print("  c1 fold {}".format(held), file=sys.stderr, flush=True)
    return held


def score_case(job):
    import study_decode
    game, seed = job
    c = study_decode.case(game, seed)
    base = study_decode.spell_f1(c["gt"], study_decode.v9_labels(c))
    c["p_dead"] = np.load(c1_path(game, seed)).astype(float)
    return c["match"].name, seed, base, study_decode.spell_f1(c["gt"], study_decode.v9_labels(c))


def score(pool):
    import tune_v9
    from harness import source
    games = external.all_games()
    jobs = [(g, s) for g in DFL_GAMES for s in TRAIN_SEEDS if not os.path.exists(tune_v9.dead_path(g, s))]
    list(pool.imap_unordered(build, jobs))
    list(pool.imap_unordered(fold, games))
    rows = list(pool.imap_unordered(score_case, [(g, s) for g in games for s in TEST_SEEDS]))
    names = sorted({r[0] for r in rows}, key=lambda m: (source(m) != "metrica", m))
    old = {g: np.mean([b for gg, _, b, _ in rows if gg == g]) for g in names}
    new = {g: np.mean([n for gg, _, _, n in rows if gg == g]) for g in names}
    ms = names
    ok, wins, drops = verdict(new, old, ms)
    print("C1 (dead-ball model + DFL): held-out all {:.3f} vs v9 + A2 {:.3f} | wins {}/{} | {} -> {}".format(
        np.mean(list(new.values())), np.mean(list(old.values())), wins, len(ms),
        ", ".join("{} {:+.3f}".format(k, v) for k, v in drops.items()), "BETTER" if ok else "no clear change"))


def verdict(new, old, matches):
    from harness import source
    mean_up = np.mean([new[m] for m in matches]) > np.mean([old[m] for m in matches])
    wins = sum(new[m] > old[m] for m in matches)
    drops = {}
    for name in ("metrica", "skillcorner"):
        ms = [m for m in matches if source(str(m)) == name]
        drops[name] = np.mean([new[m] for m in ms]) - np.mean([old[m] for m in ms])
    return mean_up and wins >= 16 and all(d >= -0.01 for d in drops.values()), wins, drops


def main():
    import multiprocessing
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=("fit", "score"))
    step = parser.parse_args().step
    setup()
    if step == "score" and DFL_BLUR_S is None:
        sys.exit("set DFL_BLUR_S and DFL_DRIFT from `fit` first")
    workers = FOLD_WORKERS if step == "score" else 8
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        fit(pool) if step == "fit" else score(pool)


if __name__ == "__main__":
    main()
