"""Pick quick win 6's touch weight, touch threshold and switch cost leave-one-match-out.

Noisy matches are built once per (match, seed) and reused for every setting; v5 is scored on the
same runs for the comparison.

    .venv/bin/python saryarka/tune_v6.py
"""

import argparse
import itertools
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from core import nearest_team, predict  # noqa: E402
import external  # noqa: E402
from harness import lomo, score, to_codes  # noqa: E402
import noise  # noqa: E402
import noise_stats  # noqa: E402

# Params
WINDOW = 50
TOUCH_WEIGHTS = (0.1, 0.15, 0.25, 0.35)
MIN_IMPULSES = (3.0, 4.0)
SWITCH_COSTS = (3.0, 5.0, 8.0)


def grid(seeds):
    stats = noise_stats.load()
    v6, v5 = {}, {}
    for game in external.GAMES:
        match, labels, restarts = external.metrica_match(game)
        gt = to_codes(labels)
        runs6, runs5 = {}, []
        for seed in range(seeds):
            variant = noise.noisy_match(match, labels, restarts, seed, stats)[0]
            codes = nearest_team(variant)
            runs5.append(score(gt, predict(variant, "v5", WINDOW, codes))[0])
            for weight, impulse, cost in itertools.product(TOUCH_WEIGHTS, MIN_IMPULSES, SWITCH_COSTS):
                runs6.setdefault((weight, impulse, cost), []).append(score(gt, predict(
                    variant, "v6", WINDOW, codes, switch_cost=cost, touch_weight=weight, min_impulse=impulse))[0])
        v6[match.name] = {k: float(np.mean(v)) for k, v in runs6.items()}
        v5[match.name] = {"v5": float(np.mean(runs5))}
        print("  scored {}".format(match.name), file=sys.stderr, flush=True)
    return v6, v5


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=3)
    v6, v5 = grid(parser.parse_args().seeds)
    matches = sorted(v6)
    print("\nv6 spellF1 at window {}, mean of matches (v5 {:.3f})".format(
        WINDOW, np.mean([v5[m]["v5"] for m in matches])))
    for key in sorted(v6[matches[0]], key=lambda k: -np.mean([v6[m][k] for m in matches])):
        print("  weight {:<5} impulse {:<4} cost {:<4} {:.3f}   per match {}".format(
            *key, np.mean([v6[m][key] for m in matches]), " ".join("{:.3f}".format(v6[m][key]) for m in matches)))
    print("  v5 per match", " ".join("{:.3f}".format(v5[m]["v5"]) for m in matches))
    held = lomo(v6)
    print("\nleave-one-match-out:")
    for m in matches:
        key, value = held[m]
        print("  {:<9} weight {} impulse {} cost {} -> {:.3f}  v5 {:.3f}  {:+.3f}".format(m, *key, value, v5[m]["v5"],
                                                                                    value - v5[m]["v5"]))
    print("  held-out mean v6 {:.3f}  v5 {:.3f}".format(np.mean([held[m][1] for m in matches]),
                                                      np.mean([v5[m]["v5"] for m in matches])))
