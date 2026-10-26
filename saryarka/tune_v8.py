"""v8 (gradient boosting) vs v7 (logistic regression), both trained on all external matches.

Leave-one-match-out over the 23 matches (3 Metrica, 20 SkillCorner): each held-out match is
scored by models trained on the other 22, so every number comes from a match the model never saw.
The switch cost is then picked leave-one-match-out with lomo(). v6 is scored on the same runs.

Signals take ~20 s per match and noise draw, so they are built once and cached under
data/external/features-cache (git ignores data/): training rows for TRAIN_SEEDS, full matrices for
TEST_SEEDS.

    .venv/bin/python saryarka/tune_v8.py            # ~30 min on 8 cores, most of it the first time
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "2")  # several workers each running sklearn: keep threads few

import argparse
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

# Params
CACHE = os.path.join(ROOT, "data", "external", "features-cache")
TRAIN_SEEDS = (100, 101)
TEST_SEEDS = (0, 1)
WINDOW = 50
SWITCH_COSTS = (1, 2, 3)
KINDS = ("logistic", "boosting")
# "balanced": boosting with Metrica's matches and SkillCorner's counting the same in total.
# Mixed training let SkillCorner's 20 matches outvote Metrica's 3 (Metrica 0.415 alone, 0.358 mixed).
ALL_KINDS = KINDS + ("balanced",)
BUILD_WORKERS = 8
FOLD_WORKERS = 4


def cache_path(game, seed):
    return os.path.join(CACHE, "{}_{}.npz".format(game, seed))


def build(job):
    """Signals for one (game, seed), cached: training rows or the full matrix."""
    warnings.filterwarnings("ignore")
    import external
    import features
    import model
    import noise
    game, seed = job
    path = cache_path(game, seed)
    if os.path.exists(path):
        return path
    match, labels, restarts = external.load(game)
    f = features.match_features(noise.noisy_match(match, labels, restarts, seed)[0])
    names = sorted(f)
    if seed in TRAIN_SEEDS:
        X, y = model.training_rows(f, labels, names)
        np.savez(path, X=X.astype(np.float32), y=y.astype(np.int8), names=np.array(names))
    else:
        np.savez(path, F=model.matrix(f, names).astype(np.float32), names=np.array(names))
    print("  cached {} seed {}".format(game, seed), file=sys.stderr, flush=True)
    return path


def source_weights(sources):
    """Per-row weights so each source's rows sum to the same total."""
    sources = np.asarray(sources)
    weights = np.ones(len(sources))
    for name in set(sources):
        weights[sources == name] = len(sources) / (len(set(sources)) * np.sum(sources == name))
    return weights


def fold(held, kinds=KINDS):
    """Scores on one held-out game: {kind: {cost: spellF1}}, v6, and per-frame accuracies."""
    warnings.filterwarnings("ignore")
    import external
    import model
    import noise
    from core import nearest_team, predict
    from harness import score, to_codes
    others = [g for g in external.all_games() if g != held]
    parts = [np.load(cache_path(g, s)) for g in others for s in TRAIN_SEEDS]
    names = list(parts[0]["names"])
    X = np.vstack([p["X"] for p in parts]).astype(float)
    y = np.concatenate([p["y"] for p in parts])
    sources = np.concatenate([np.full(len(p["y"]), g in external.GAMES) for g, p in
                              zip([g for g in others for _ in TRAIN_SEEDS], parts)])
    fits = {kind: model.fit(X, y, "boosting", source_weights(sources)) if kind == "balanced"
            else model.fit(X, y, kind) for kind in kinds}

    match, labels, restarts = external.load(held)
    gt = to_codes(labels)
    live = np.isin(labels, ["H", "A"])
    out = {kind: {c: [] for c in SWITCH_COSTS} for kind in kinds}
    out["v6"], out["acc"] = [], {kind: [] for kind in kinds + ("vote",)}
    for seed in TEST_SEEDS:
        variant = noise.noisy_match(match, labels, restarts, seed)[0]
        codes = nearest_team(variant)
        F = np.load(cache_path(held, seed))["F"].astype(float)
        f = {name: F[:, i] for i, name in enumerate(names)}
        out["v6"].append(score(gt, predict(variant, "v6", WINDOW, codes))[0])
        out["acc"]["vote"].append(np.mean((f["vote_share"][live] > 0) == (labels[live] == "H")))
        for kind, fitted in fits.items():
            p = model.home_probability(fitted, names, f)
            out["acc"][kind].append(np.mean((p[live] > 0.5) == (labels[live] == "H")))
            for cost in SWITCH_COSTS:
                out[kind][cost].append(score(gt, model.labels(variant, codes, p, WINDOW, cost))[0])
    print("  scored {}".format(match.name), file=sys.stderr, flush=True)
    return match.name, {
        **{kind: {c: float(np.mean(v)) for c, v in out[kind].items()} for kind in kinds},
        "v6": float(np.mean(out["v6"])),
        "acc": {k: float(np.mean(v)) for k, v in out["acc"].items()},
    }


def saved_fold(job):
    """fold(), with its result kept on disk so a stopped run resumes where it was."""
    import json
    held, kinds, tag = job
    path = os.path.join(CACHE, "fold{}_{}.json".format(tag, held))
    if os.path.exists(path):
        with open(path) as handle:
            name, result = json.load(handle)
        for kind in kinds:  # JSON turned the switch-cost keys into strings
            result[kind] = {float(c) if "." in c else int(c): v for c, v in result[kind].items()}
        return name, result
    result = fold(held, kinds)
    with open(path, "w") as handle:
        json.dump(result, handle)
    return result


def main():
    import multiprocessing
    import external
    from harness import lomo, source
    os.makedirs(CACHE, exist_ok=True)
    games = external.all_games()
    ctx = multiprocessing.get_context("spawn")
    jobs = [(g, s) for g in games for s in TRAIN_SEEDS + TEST_SEEDS if not os.path.exists(cache_path(g, s))]
    if jobs:
        with ctx.Pool(BUILD_WORKERS) as pool:
            list(pool.imap_unordered(build, jobs))
    results = {}
    runs = [(g, KINDS, "") for g in games] + [(g, ("balanced",), "_balanced") for g in games]
    with ctx.Pool(FOLD_WORKERS) as pool:
        for name, result in pool.imap_unordered(saved_fold, runs):
            acc = results.setdefault(name, {"acc": {}})
            acc["acc"].update(result.pop("acc"))
            acc.update(result)

    matches = sorted(results, key=lambda m: (source(m) != "metrica", m))
    print("\nper-frame right team on live frames (mean over matches):")
    for k in ALL_KINDS + ("vote",):
        print("  {:<9} {:.3f}".format(k, np.mean([results[m]["acc"][k] for m in matches])))

    picked = {kind: lomo({m: results[m][kind] for m in matches}) for kind in ALL_KINDS}
    print("\nheld-out spellF1 (switch cost picked leave-one-match-out), window {}:".format(WINDOW))
    print("{:<20} {:>8} {:>9} {:>9} {:>9}".format("match", "v6", *ALL_KINDS))
    for m in matches:
        print("{:<20} {:>8.3f} {:>9.3f} {:>9.3f} {:>9.3f}".format(m, results[m]["v6"], *(picked[k][m][1] for k in ALL_KINDS)))
    groups = {}
    for m in matches:
        groups.setdefault(source(m), []).append(m)
    print("\nmeans ('sources equal' = mean of the two source means, the deciding number):")
    rows = {"v6": {m: results[m]["v6"] for m in matches}}
    rows.update({k: {m: picked[k][m][1] for m in matches} for k in ALL_KINDS})
    for k, by_match in rows.items():
        per = {name: np.mean([by_match[m] for m in ms]) for name, ms in groups.items()}
        print("  {:<9} sources equal {:.3f}  | ".format(k, np.mean(list(per.values())))
              + "  ".join("{} {:.3f}".format(n, v) for n, v in per.items())
              + "  | all {:.3f}".format(np.mean(list(by_match.values()))))
    for a_, b_ in (("balanced", "boosting"), ("balanced", "logistic")):
        wins = sum(rows[a_][m] > rows[b_][m] for m in matches)
        print("  {} beats {} on {}/{} matches".format(a_, b_, wins, len(matches)))
    for kind in ALL_KINDS:
        costs = [picked[kind][m][0] for m in matches]
        print("{} costs picked: {}".format(kind, {c: costs.count(c) for c in sorted(set(costs))}))
        print("  all-match means by cost: " + "  ".join("c{} {:.3f}".format(c, np.mean([results[m][kind][c] for m in matches]))
                                                      for c in SWITCH_COSTS))


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    main()
