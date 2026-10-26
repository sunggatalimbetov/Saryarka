"""Score every version of our method on external full matches we hold the answers to.

Each match is scored clean (Metrica as shipped, perfect restart tags) and noisy (BAR-ATH-like
dropout, dead-ball ball and Smart Tagging, see noise.py) over several seeds. Scoring is the
official evaluate() from evaluation/possession_metrics.py. The labels come from Metrica's events,
not from the organisers' convention, so these numbers rank our versions against each other; they
do not predict our score on BAR-ATH.

    .venv/bin/python saryarka/harness.py                               # everything, 5 seeds
    .venv/bin/python saryarka/harness.py --versions baseline v3 --windows 50 --seeds 2

Decision rules (agreed 26 Oct):
  * "better" = higher mean over matches (each match averaged over seeds) AND wins on at least
    two thirds of the matches
  * the noisy baseline must rank the windows like the published full match: 50 > 125 > 25 > raw
  * anything tuned is tuned leave-one-match-out, see lomo()
"""

import argparse
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from core import VERSIONS, nearest_team, predict  # noqa: E402
import external  # noqa: E402
import noise  # noqa: E402
import noise_stats  # noqa: E402

# Params
WINDOWS = (0, 25, 50, 125)
PUBLISHED_ORDER = (50, 125, 25, 0)  # full-match baseline ranking, EVALUATION.md
CODE = {"H": pm.H, "A": pm.A, "D": pm.D, "X": pm.X}


def to_codes(labels):
    out = np.full(len(labels), pm.X, dtype=np.int8)
    for name, code in CODE.items():
        out[labels == name] = code
    return out


def score(gt_codes, labels):
    result = pm.evaluate(gt_codes, to_codes(labels))
    return result["spell_f1_mean"], result["macro_f1"], result["n_pred_spells"]


def run(games, versions, windows, seeds):
    """{(match, condition, version, window): (spellF1, macroF1, n_spells)}; condition 'clean' or seed."""
    stats = noise_stats.load()
    rows = {}
    for game in games:
        match, labels, restarts = external.metrica_match(game)
        gt = to_codes(labels)
        conditions = [("clean", match)] + [(seed, noise.noisy_match(match, labels, restarts, seed, stats)[0])
                                           for seed in range(seeds)]
        for condition, variant in conditions:
            codes = nearest_team(variant)
            for version in versions:
                for window in windows:
                    rows[(match.name, condition, version, window)] = score(gt, predict(variant, version, window, codes))
        print("  scored {} ({} possessions in the answers)".format(match.name, pm.spell_scores(gt, gt, pm.valid_runs(gt))["n_gt"]),
              file=sys.stderr)
    return rows


def noisy_mean(rows, match, version, window, metric=0):
    values = [v[metric] for (m, c, ver, w), v in rows.items()
              if m == match and c != "clean" and ver == version and w == window]
    return float(np.mean(values)) if values else float("nan")


def better(rows, matches, a, b):
    """Is configuration a = (version, window) better than b under the agreed rule?"""
    sa = [noisy_mean(rows, m, *a) for m in matches]
    sb = [noisy_mean(rows, m, *b) for m in matches]
    wins = sum(x > y for x, y in zip(sa, sb))
    return np.mean(sa) > np.mean(sb) and wins >= math.ceil(2 * len(matches) / 3)


def lomo(scores):
    """Leave-one-match-out: scores = {match: {setting: value}}. Returns {held_out: (setting, value)}."""
    out = {}
    for held in scores:
        others = [m for m in scores if m != held]
        settings = scores[held].keys()
        pick = max(settings, key=lambda s: np.mean([scores[m][s] for m in others]))
        out[held] = (pick, scores[held][pick])
    return out


def report(rows, versions, windows):
    matches = sorted({k[0] for k in rows})
    print("\nspellF1 -- noisy (mean of seeds) | clean, per match, then mean of matches")
    header = "{:<10} {:>4}  ".format("version", "win") + "  ".join("{:>15}".format(m) for m in matches) + "  {:>15}".format("MEAN")
    print(header)
    for version in versions:
        for window in windows:
            cells, noisy_all, clean_all = [], [], []
            for m in matches:
                n = noisy_mean(rows, m, version, window)
                c = rows[(m, "clean", version, window)][0]
                noisy_all.append(n)
                clean_all.append(c)
                cells.append("{:.3f} | {:.3f}".format(n, c))
            cells.append("{:.3f} | {:.3f}".format(np.mean(noisy_all), np.mean(clean_all)))
            print("{:<10} {:>4}  ".format(version, window) + "  ".join("{:>15}".format(c) for c in cells))

    if "baseline" in versions and set(PUBLISHED_ORDER) <= set(windows):
        means = {w: np.mean([noisy_mean(rows, m, "baseline", w) for m in matches]) for w in PUBLISHED_ORDER}
        ours = tuple(sorted(means, key=means.get, reverse=True))
        print("\ncalibration: noisy baseline ranks windows {} (published {}) -> {}".format(
            " > ".join(map(str, ours)), " > ".join(map(str, PUBLISHED_ORDER)),
            "PASS" if ours == PUBLISHED_ORDER else "FAIL"))
        print("             noisy baseline_smooth50 mean {:.3f} (real full match 0.243, informative only)".format(means[50]))

    print("\nverdicts (noisy, mean up AND wins on >= 2/3 of matches):")
    for window in windows:
        chain = [v for v in versions]
        for a, b in zip(chain[1:], chain[:-1]):
            print("  w{:<4} {:<8} vs {:<8}: {}".format(window, a, b, "better" if better(rows, matches, (a, window), (b, window)) else "not better"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--games", type=int, nargs="+", default=list(external.CSV_GAMES))
    parser.add_argument("--versions", nargs="+", default=list(VERSIONS))
    parser.add_argument("--windows", type=int, nargs="+", default=list(WINDOWS))
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()
    rows = run(args.games, args.versions, args.windows, args.seeds)
    report(rows, args.versions, args.windows)
