"""v9 (learned dead ball) vs v8 on the 23 external matches, leave-one-match-out.

For each held-out match, both v8's possession model and the dead-ball model (deadball.py) are
trained on the other 22, then the match is scored as v8, as v8 + stage A (tagged spans backdated),
v8 + stage B (new stoppages) and v8 + A + B over a small threshold grid. Settings are picked
leave-one-match-out with lomo().

Decision rule (agreed 27 Oct, before any results): a stage is better when the 23-match mean goes
up AND it wins on at least 16 of 23 matches, and neither source's mean drops by more than 0.01.

    .venv/bin/python saryarka/tune_v9.py      # ~40 min on 8 cores; fold results resume from disk
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
from tune_v8 import CACHE, TEST_SEEDS, TRAIN_SEEDS, cache_path, source_weights  # noqa: E402

# Params
WINDOW = 50
SWITCH_COST = 3.0          # v8's
T_LOWS = (0.3, 0.4, 0.5)
T_HIGHS = (0.6, 0.7, 0.8)
MIN_DEADS = (2.0, 3.0)
MIN_WINS = 16              # of 23
MAX_SOURCE_DROP = 0.01
BUILD_WORKERS = 8
FOLD_WORKERS = 4


def settings():
    """{family: [setting]}; a setting is a tuple (t_low, t_high, min_dead_s)."""
    pairs = [(lo, hi, m) for lo, hi, m in itertools.product(T_LOWS, T_HIGHS, MIN_DEADS) if lo < hi]
    return {"v8": [(0, 0, 0)], "A": [(lo, 0, 0) for lo in T_LOWS], "B": pairs, "AB": pairs}


def key(setting):
    return "{:g}/{:g}/{:g}".format(*setting)


def dead_path(game, seed):
    return os.path.join(CACHE, "dead_{}_{}.npz".format(game, seed))


def build_dead(job):
    """Dead-ball signals for one (game, seed): training rows or the full matrix, cached."""
    warnings.filterwarnings("ignore")
    import deadball
    import external
    import noise
    game, seed = job
    path = dead_path(game, seed)
    os.makedirs(CACHE, exist_ok=True)
    if os.path.exists(path):
        return path
    match, labels, restarts = external.load(game)
    f = deadball.build(noise.noisy_match(match, labels, restarts, seed)[0])
    names = sorted(f)
    if seed in TRAIN_SEEDS:
        X, y = deadball.training_rows(f, labels, external.dead_trust(game), names)
        np.savez(path, X=X.astype(np.float32), y=y, names=np.array(names))
    else:
        np.savez(path, F=np.column_stack([f[n] for n in names]).astype(np.float32), names=np.array(names))
    print("  dead cached {} seed {}".format(game, seed), file=sys.stderr, flush=True)
    return path


def train(others, path_of, weights=True):
    import external
    import model
    jobs = [(g, s) for g in others for s in TRAIN_SEEDS]
    parts = [np.load(path_of(g, s)) for g, s in jobs]
    sources = np.concatenate([np.full(len(p["y"]), g in external.GAMES) for (g, _), p in zip(jobs, parts)])
    X = np.vstack([p["X"] for p in parts]).astype(float)
    y = np.concatenate([p["y"] for p in parts])
    return model.fit(X, y, "boosting", source_weights(sources)), list(parts[0]["names"])


def fold(held):
    """{family: {setting key: [spellF1, D precision, D recall]}} on one held-out match."""
    warnings.filterwarnings("ignore")
    import deadball
    import external
    import model
    import noise
    from core import nearest_team
    from harness import score, to_codes
    others = [g for g in external.all_games() if g != held]
    home_fit, home_names = train(others, cache_path)
    dead_fit, dead_names = train(others, dead_path)

    match, labels, restarts = external.load(held)
    gt = to_codes(labels)
    graded = (labels != "X") & external.dead_trust(held)
    truth = labels == "D"
    out = {family: {key(s): [] for s in options} for family, options in settings().items()}
    for seed in TEST_SEEDS:
        variant = noise.noisy_match(match, labels, restarts, seed)[0]
        codes = nearest_team(variant)
        F = np.load(cache_path(held, seed))["F"].astype(float)
        p_home = model.home_probability(home_fit, home_names, {n: F[:, i] for i, n in enumerate(home_names)})
        D = np.load(dead_path(held, seed))["F"].astype(float)
        p_dead = deadball.probability(dead_fit, dead_names, {n: D[:, i] for i, n in enumerate(dead_names)})
        for family, options in settings().items():
            for s in options:
                lo, hi, m = s
                dead = deadball.spans(variant, p_dead, family in ("A", "AB"), family in ("B", "AB"),
                                      t_low=lo or deadball.T_LOW, t_high=hi or deadball.T_HIGH,
                                      min_dead_s=m or deadball.MIN_DEAD_S)
                predicted = np.zeros(len(labels), bool)
                for a, b in dead:
                    predicted[a:b] = True
                hit = (predicted & truth & graded).sum()
                precision = hit / max(1, (predicted & graded).sum())
                recall = hit / max(1, (truth & graded).sum())
                spell = score(gt, deadball.labels(variant, codes, p_home, dead, WINDOW, SWITCH_COST))[0]
                out[family][key(s)].append([spell, precision, recall])
    print("  scored {}".format(match.name), file=sys.stderr, flush=True)
    return match.name, {fam: {k: np.mean(v, axis=0).tolist() for k, v in d.items()} for fam, d in out.items()}


def saved_fold(held):
    path = os.path.join(CACHE, "fold9_{}.json".format(held))
    if os.path.exists(path):
        with open(path) as handle:
            return tuple(json.load(handle))
    result = fold(held)
    with open(path, "w") as handle:
        json.dump(result, handle)
    return result


def verdict(new, old, matches, source):
    mean_up = np.mean([new[m] for m in matches]) > np.mean([old[m] for m in matches])
    wins = sum(new[m] > old[m] for m in matches)
    drops = {}
    for name in sorted({source(m) for m in matches}):
        ms = [m for m in matches if source(m) == name]
        drops[name] = np.mean([new[m] for m in ms]) - np.mean([old[m] for m in ms])
    ok = mean_up and wins >= MIN_WINS and all(d >= -MAX_SOURCE_DROP for d in drops.values())
    return ok, wins, drops


def main():
    import multiprocessing
    import external
    from harness import lomo, source
    games = external.all_games()
    ctx = multiprocessing.get_context("spawn")
    import tune_v8
    for builder, path_of in ((build_dead, dead_path), (tune_v8.build, cache_path)):
        jobs = [(g, s) for g in games for s in TRAIN_SEEDS + TEST_SEEDS if not os.path.exists(path_of(g, s))]
        if jobs:
            with ctx.Pool(BUILD_WORKERS) as pool:
                list(pool.imap_unordered(builder, jobs))
    with ctx.Pool(FOLD_WORKERS) as pool:
        results = dict(pool.imap_unordered(saved_fold, games))
    matches = sorted(results, key=lambda m: (source(m) != "metrica", m))

    held = {}
    picks = {}
    for family in ("v8", "A", "B", "AB"):
        table = {m: {k: v[0] for k, v in results[m][family].items()} for m in matches}
        picked = lomo(table)
        held[family] = {m: picked[m][1] for m in matches}
        picks[family] = picked

    print("\nheld-out spellF1 (settings picked leave-one-match-out), window {}, cost {}:".format(WINDOW, SWITCH_COST))
    print("{:<20} {:>7} {:>7} {:>7} {:>7}".format("match", "v8", "A", "B", "A+B"))
    for m in matches:
        print("{:<20} {:>7.3f} {:>7.3f} {:>7.3f} {:>7.3f}".format(m, *(held[f][m] for f in ("v8", "A", "B", "AB"))))
    for family in ("v8", "A", "B", "AB"):
        by = {n: np.mean([held[family][m] for m in matches if source(m) == n]) for n in ("metrica", "skillcorner")}
        dprec = np.mean([results[m][family][picks[family][m][0]][1] for m in matches])
        drec = np.mean([results[m][family][picks[family][m][0]][2] for m in matches])
        settings_used = sorted({picks[family][m][0] for m in matches})
        print("MEAN {:<3} all {:.3f} | metrica {:.3f} skillcorner {:.3f} | D precision {:.2f} recall {:.2f} | settings {}".format(
            family, np.mean(list(held[family].values())), by["metrica"], by["skillcorner"], dprec, drec, settings_used))
    print("\nverdicts (mean up AND >= {}/{} wins AND no source drops > {}):".format(MIN_WINS, len(matches), MAX_SOURCE_DROP))
    for family in ("A", "B", "AB"):
        ok, wins, drops = verdict(held[family], held["v8"], matches, source)
        print("  {:<3} vs v8: {} (wins {}/{}, by source {})".format(
            family, "BETTER" if ok else "no clear change / worse", wins, len(matches),
            ", ".join("{} {:+.3f}".format(k, v) for k, v in drops.items())))
    ok, wins, drops = verdict(held["AB"], held["A"], matches, source)
    print("  A+B vs A : {} (wins {}/{}, by source {})".format(
        "BETTER" if ok else "no clear change / worse", wins, len(matches),
        ", ".join("{} {:+.3f}".format(k, v) for k, v in drops.items())))


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    main()
