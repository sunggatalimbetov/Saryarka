"""Score every version of our method on external full matches we hold the answers to.

Each match is scored clean (as shipped, perfect restart tags) and noisy (BAR-ATH-like dropout,
dead-ball ball and Smart Tagging, see noise.py) over several seeds. Scoring is the official
evaluate() from evaluation/possession_metrics.py. The labels come from Metrica's events
(external.py) and SkillCorner's per-frame possession (skillcorner.py), not from the organisers'
convention, so these numbers rank our versions against each other; they do not predict our score
on BAR-ATH.

    .venv/bin/python saryarka/harness.py                               # all 23 matches, 5 seeds (~2 min)
    .venv/bin/python saryarka/harness.py --source metrica              # the original 3
    .venv/bin/python saryarka/harness.py --versions baseline v3 --windows 50 --seeds 2

Decision rules (agreed 26 Oct):
  * "better" = higher mean over matches (each match averaged over seeds) AND wins on at least
    two thirds of the matches
  * the noisy baseline must rank the windows like the published full match: 50 > 125 > 25 > raw
  * anything tuned is tuned leave-one-match-out, see lomo()

Known bias (accepted 26 Oct): Metrica's event-based possessions are shorter (10-11 s mean) than the
organisers' (14 s on the public slice), so the harness leans toward shorter windows. Calibration
passes on all three games, but 25 and 125 are a near tie (match 3 favours 125, matches 1-2 favour 25)
where the real match has 125 ahead by 0.025. Choices between nearby windows need a margin.
Noise settings are locked as of this commit. 26 Oct: ball-position wobble added (noise.py, fitted by
fit_wobble.py to BAR-ATH summaries); calibration still passes and 125 now leads 25 by 0.018 (real
0.025), baseline_smooth50 0.232 (real 0.243). Then a centred 0.3 s ball blur, so kicks near players
look like BAR-ATH's: 125 leads 25 by 0.031, baseline_smooth50 0.220. Not modelled: BAR-ATH's ball is far from every player
more often (75-90th percentile distance), and drifts more over a full second.

SkillCorner (26 Oct, 20 A-League matches): ball blur and drift fitted separately (noise.PROFILES).
Calibration FAILS there: the noisy baseline ranks 125 > 50 (0.203 vs 0.193, 16 of 20 matches),
while clean SkillCorner ranks 50 first and v6 prefers 50 on 19 of 20. The ball dropout is the
cause (without it 50 leads on every match tried), not the fitted blur or wobble. So SkillCorner
compares versions at a fixed window, but window choices lean on Metrica and on v6+.
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
WORKERS = max(1, (os.cpu_count() or 2) - 2)  # games run in parallel processes
PER_MATCH_COLUMNS = 4  # more matches than this: the table shows one column per source instead
CODE = {"H": pm.H, "A": pm.A, "D": pm.D, "X": pm.X}


def to_codes(labels):
    out = np.full(len(labels), pm.X, dtype=np.int8)
    for name, code in CODE.items():
        out[labels == name] = code
    return out


def score(gt_codes, labels):
    result = pm.evaluate(gt_codes, to_codes(labels))
    return result["spell_f1_mean"], result["macro_f1"], result["n_pred_spells"]


def run_game(args):
    """Rows for one game (a worker process: games are independent)."""
    game, versions, windows, seeds = args
    stats = noise_stats.load()
    match, labels, restarts = external.load(game)
    gt = to_codes(labels)
    conditions = [("clean", match)] + [(seed, noise.noisy_match(match, labels, restarts, seed, stats)[0])
                                       for seed in range(seeds)]
    rows = {}
    for condition, variant in conditions:
        codes = nearest_team(variant)
        for version in versions:
            for window in windows:
                rows[(match.name, condition, version, window)] = score(gt, predict(variant, version, window, codes))
    print("  scored {} ({} possessions in the answers)".format(match.name, pm.spell_scores(gt, gt, pm.valid_runs(gt))["n_gt"]),
          file=sys.stderr, flush=True)
    return rows


def run(games, versions, windows, seeds, workers=WORKERS):
    """{(match, condition, version, window): (spellF1, macroF1, n_spells)}; condition 'clean' or seed."""
    jobs = [(game, versions, windows, seeds) for game in games]
    rows = {}
    if workers > 1 and len(jobs) > 1:
        import multiprocessing
        with multiprocessing.get_context("spawn").Pool(min(workers, len(jobs))) as pool:
            for part in pool.imap_unordered(run_game, jobs):
                rows.update(part)
    else:
        for job in jobs:
            rows.update(run_game(job))
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


def source(match):
    return "skillcorner" if match.startswith("skillcorner") else "metrica"


def report(rows, versions, windows):
    matches = sorted({k[0] for k in rows}, key=lambda m: (source(m) != "metrica", m))
    groups = {}
    for m in matches:
        groups.setdefault(source(m), []).append(m)
    if len(matches) <= PER_MATCH_COLUMNS:
        columns = [(m, [m]) for m in matches]
    else:
        columns = [("{} ({})".format(name, len(ms)), ms) for name, ms in groups.items()]
    columns.append(("MEAN ({})".format(len(matches)), matches))
    print("\nspellF1 -- noisy (mean of seeds) | clean; per match or source, mean of matches")
    print("{:<10} {:>4}  ".format("version", "win") + "  ".join("{:>17}".format(c) for c, _ in columns))
    for version in versions:
        for window in windows:
            cells = []
            for _, ms in columns:
                n = np.mean([noisy_mean(rows, m, version, window) for m in ms])
                c = np.mean([rows[(m, "clean", version, window)][0] for m in ms])
                cells.append("{:.3f} | {:.3f}".format(n, c))
            print("{:<10} {:>4}  ".format(version, window) + "  ".join("{:>17}".format(c) for c in cells))

    if "baseline" in versions and set(PUBLISHED_ORDER) <= set(windows):
        for name, ms in list(groups.items()) + ([("all", matches)] if len(groups) > 1 else []):
            means = {w: np.mean([noisy_mean(rows, m, "baseline", w) for m in ms]) for w in PUBLISHED_ORDER}
            ours = tuple(sorted(means, key=means.get, reverse=True))
            print("\ncalibration ({}): noisy baseline ranks windows {} (published {}) -> {}".format(
                name, " > ".join(map(str, ours)), " > ".join(map(str, PUBLISHED_ORDER)),
                "PASS" if ours == PUBLISHED_ORDER else "FAIL"))
            print("             noisy baseline_smooth50 mean {:.3f} (real full match 0.243, informative only); "
                  "125 - 25 = {:+.3f}".format(means[50], means[125] - means[25]))

    print("\nverdicts (noisy, mean up AND wins on >= 2/3 of matches), each version vs the one before it:")
    for window in windows:
        for a, b in zip(versions[1:], versions[:-1]):
            same = all(noisy_mean(rows, m, a, window) == noisy_mean(rows, m, b, window) for m in matches)
            if same:
                verdict = "untested (identical output here)"
            else:
                parts = ["{} {}".format(name, "better" if better(rows, ms, (a, window), (b, window)) else "not better")
                         for name, ms in ([("all", matches)] + (list(groups.items()) if len(groups) > 1 else []))]
                verdict = ", ".join(parts)
            print("  w{:<4} {:<8} vs {:<8}: {}".format(window, a, b, verdict))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--games", type=int, nargs="+", default=None,
                        help="Metrica 1-3 and/or SkillCorner match ids (default: every downloaded game)")
    parser.add_argument("--source", choices=("all", "metrica", "skillcorner"), default="all")
    parser.add_argument("--versions", nargs="+", default=list(VERSIONS))
    parser.add_argument("--windows", type=int, nargs="+", default=list(WINDOWS))
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be at least 1: the verdicts are on the noisy runs")
    games = args.games or [g for g in external.all_games()
                           if args.source == "all" or (args.source == "metrica") == (g in external.GAMES)]
    rows = run(games, args.versions, args.windows, args.seeds)
    report(rows, args.versions, args.windows)
