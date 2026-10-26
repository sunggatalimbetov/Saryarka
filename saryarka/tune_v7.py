"""v7 vs v6 on the external matches, with nothing tuned on the match being scored.

Each match's probabilities come from a model trained on the other two (training noise seeds
TRAIN_SEEDS); the switch cost is then picked leave-one-match-out with lomo() on those out-of-fold
scores. v6 is scored on the same noisy runs (TEST_SEEDS).

    .venv/bin/python saryarka/tune_v7.py
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from core import nearest_team, predict  # noqa: E402
import external  # noqa: E402
import features  # noqa: E402
from harness import lomo, score, to_codes  # noqa: E402
import model  # noqa: E402
import noise  # noqa: E402
import noise_stats  # noqa: E402

# Params
TRAIN_SEEDS = (100, 101, 102)   # noise draws used for training, distinct from the scored ones
TEST_SEEDS = (0, 1, 2, 3, 4)
WINDOW = 50
SWITCH_COSTS = (1, 2, 3, 5)


def noisy_runs(seeds):
    """{match name: [(variant, labels, features)]} for every external game."""
    stats = noise_stats.load()
    out = {}
    for game in external.GAMES:
        match, labels, restarts = external.metrica_match(game)
        out[match.name] = []
        for seed in seeds:
            variant = noise.noisy_match(match, labels, restarts, seed, stats)[0]
            out[match.name].append((variant, labels, features.match_features(variant)))
        print("  built {} x{}".format(match.name, len(seeds)), file=sys.stderr, flush=True)
    return out


def without(f, drop):
    return {name: values for name, values in f.items() if not any(name.startswith(d) for d in drop)}


def main(drop=(), costs=SWITCH_COSTS):
    global SWITCH_COSTS
    SWITCH_COSTS = tuple(costs)
    train_runs = noisy_runs(TRAIN_SEEDS)
    test_runs = noisy_runs(TEST_SEEDS)
    if drop:
        print("dropping signals starting with:", ", ".join(drop))
        train_runs = {m: [(v, lab, without(f, drop)) for v, lab, f in runs] for m, runs in train_runs.items()}
        test_runs = {m: [(v, lab, without(f, drop)) for v, lab, f in runs] for m, runs in test_runs.items()}
    matches = sorted(test_runs)
    v7, v6, frame_acc = {}, {}, {}
    for held in matches:
        others = [m for m in matches if m != held]
        fit, names = model.train([f for m in others for _, _, f in train_runs[m]],
                                 [lab for m in others for _, lab, _ in train_runs[m]])
        if held == matches[0]:
            print("\nlargest standardised weights (model without {}):".format(held))
            for name, weight in model.weights(fit, names):
                print("  {:<22} {:+.2f}".format(name, weight))
        v7[held] = {cost: [] for cost in SWITCH_COSTS}
        v6[held], frame_acc[held] = [], []
        for variant, labels, f in test_runs[held]:
            gt = to_codes(labels)
            codes = nearest_team(variant)
            p = model.home_probability(fit, names, f)
            live = np.isin(labels, ["H", "A"])
            vote = f["vote_share"]
            frame_acc[held].append((np.mean((p[live] > 0.5) == (labels[live] == "H")),
                                    np.mean((vote[live] > 0) == (labels[live] == "H"))))
            v6[held].append(score(gt, predict(variant, "v6", WINDOW, codes))[0])
            for cost in SWITCH_COSTS:
                v7[held][cost].append(score(gt, model.labels(variant, codes, p, WINDOW, cost))[0])
        v7[held] = {cost: float(np.mean(values)) for cost, values in v7[held].items()}

    print("\nper-frame accuracy on live frames (model p > 0.5 vs vote share > 0):")
    for m in matches:
        a, b = np.mean(frame_acc[m], axis=0)
        print("  {:<9} model {:.3f}  vote {:.3f}".format(m, a, b))
    print("\nv7 spellF1 by switch cost (out-of-fold model), v6 on the same runs:")
    for m in matches:
        print("  {:<9} ".format(m) + "  ".join("c{} {:.3f}".format(c, v7[m][c]) for c in SWITCH_COSTS)
              + "   v6 {:.3f}".format(np.mean(v6[m])))
    held = lomo(v7)
    print("\nleave-one-match-out:")
    for m in matches:
        cost, value = held[m]
        print("  {:<9} cost {:<3} v7 {:.3f}  v6 {:.3f}  {:+.3f}".format(m, cost, value, np.mean(v6[m]),
                                                                       value - np.mean(v6[m])))
    wins = sum(held[m][1] > np.mean(v6[m]) for m in matches)
    print("  held-out mean v7 {:.3f}  v6 {:.3f}, wins {}/{}".format(
        np.mean([held[m][1] for m in matches]), np.mean([np.mean(v6[m]) for m in matches]), wins, len(matches)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--drop", nargs="*", default=[], help="leave out signals starting with these")
    parser.add_argument("--costs", type=float, nargs="+", default=list(SWITCH_COSTS))
    args = parser.parse_args()
    main(args.drop, args.costs)
