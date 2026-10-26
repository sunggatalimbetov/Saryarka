"""Pick quick win 4's window and switch cost leave-one-match-out on the external matches.

For every (window, switch cost) pair, v4 is scored on each Metrica match (noisy, mean of seeds).
lomo() picks the pair on two matches and scores it on the third, so each held-out number is honest;
the final setting is the pair best on all three. v3 at its own best window is printed alongside.

    .venv/bin/python saryarka/tune_v4.py
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
WINDOWS = (25, 50, 75)
SWITCH_COSTS = (1, 2, 3, 5, 8, 12)
V3_WINDOWS = (25, 50, 125)


def grid(seeds):
    """({match: {(window, cost): spellF1}}, {match: {window: v3 spellF1}}), noisy means over seeds."""
    stats = noise_stats.load()
    v4, v3 = {}, {}
    for game in external.GAMES:
        match, labels, restarts = external.metrica_match(game)
        gt = to_codes(labels)
        runs4, runs3 = {}, {}
        for seed in range(seeds):
            variant = noise.noisy_match(match, labels, restarts, seed, stats)[0]
            codes = nearest_team(variant)
            for window, cost in itertools.product(WINDOWS, SWITCH_COSTS):
                runs4.setdefault((window, cost), []).append(
                    score(gt, predict(variant, "v4", window, codes, switch_cost=cost))[0])
            for window in V3_WINDOWS:
                runs3.setdefault(window, []).append(score(gt, predict(variant, "v3", window, codes))[0])
        v4[match.name] = {k: float(np.mean(v)) for k, v in runs4.items()}
        v3[match.name] = {k: float(np.mean(v)) for k, v in runs3.items()}
        print("  scored {}".format(match.name), file=sys.stderr)
    return v4, v3


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=5)
    v4, v3 = grid(parser.parse_args().seeds)
    matches = sorted(v4)
    print("\nv4 spellF1, mean of matches (rows window, columns switch cost)")
    print("{:>6} ".format("") + " ".join("{:>6}".format(c) for c in SWITCH_COSTS))
    for window in WINDOWS:
        print("{:>6} ".format(window) + " ".join("{:6.3f}".format(np.mean([v4[m][(window, c)] for m in matches]))
                                                for c in SWITCH_COSTS))
    held4, held3 = lomo(v4), lomo(v3)
    print("\nleave-one-match-out (picked on the other two, scored here):")
    for m in matches:
        (w, c), s4 = held4[m]
        w3, s3 = held3[m]
        print("  {:<9} v4 w{} cost {:<4} {:.3f}   v3 w{} {:.3f}   {:+.3f}".format(m, w, c, s4, w3, s3, s4 - s3))
    print("  held-out mean v4 {:.3f}  v3 {:.3f}".format(np.mean([held4[m][1] for m in matches]),
                                                      np.mean([held3[m][1] for m in matches])))
    best = max(v4[matches[0]], key=lambda k: np.mean([v4[m][k] for m in matches]))
    print("final setting (best on all three): window {} switch cost {}".format(*best))
