"""BAR-ATH summary checks for decoder studies: counts of our own output only, never answers.

Trains v9's two models on all 23 external matches once (as model.py does) and keeps P(home) and
P(dead) in data/ (git ignores it). Then prints, for a decoder setting, the BAR-ATH guard agreed on
27 Oct (fixed before results): dead share of the graded halves 25-45%, at most 60 untagged
stoppages added (stage B) inside the graded halves; plus possessions per half.

    .venv/bin/python saryarka/study_bar_ath.py            # v9 as shipped: expect 42.6% dead, 60 new
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

# Params
HALVES = ((11185, 90652), (110808, 190198))   # graded halves, published in possession_metrics.valid_runs
DEAD_SHARE = (0.25, 0.45)
MAX_NEW_STOPPAGES = 60
P_HOME = os.path.join(ROOT, "data", "bar_ath_p_home.npy")
P_DEAD = os.path.join(ROOT, "data", "bar_ath_p_dead.npy")


def probabilities(rebuild=False):
    """(match, codes, p_home, p_dead) for BAR-ATH from models trained on the external matches."""
    warnings.filterwarnings("ignore")
    import core
    import deadball
    import features
    import model
    match = core.bar_ath_match()
    codes = core.nearest_team(match)
    if rebuild or not (os.path.exists(P_HOME) and os.path.exists(P_DEAD)):
        fit_, names = model.train_v8()
        np.save(P_HOME, model.home_probability(fit_, names, features.match_features(match, codes)))
        dead_fit, dead_names = model.train_dead()
        np.save(P_DEAD, deadball.probability(dead_fit, dead_names, deadball.build(match)))
    return match, codes, np.load(P_HOME), np.load(P_DEAD)


def in_halves(frame):
    return any(a <= frame < b for a, b in HALVES)


def guard(match, dead, labels):
    """(passes, summary dict) for dead spans `dead` and the labels decoded with them."""
    import core
    base = core.dead_spans(match, goals=True)
    covered = np.zeros(len(labels), bool)
    for a, b in base:
        covered[a:b] = True
    new = [(a, b) for a, b in dead if in_halves((a + b) // 2) and not covered[max(a - 1, 0):b + 1].any()]
    graded = np.zeros(len(labels), bool)
    for a, b in HALVES:
        graded[a:b] = True
    share = np.mean(labels[graded] == "D")
    spells = []
    for a, b in HALVES:
        piece = labels[a:b]
        known = piece[piece != "X"]
        spells.append(int(np.sum((known[1:] != known[:-1]) & np.isin(known[1:], ["H", "A"])) + (known[:1] != "D").sum()))
    ok = DEAD_SHARE[0] <= share <= DEAD_SHARE[1] and len(new) <= MAX_NEW_STOPPAGES
    return ok, {"dead_share": share, "new_stoppages": len(new), "possessions_per_half": spells}


def report(name, match, dead, labels):
    ok, s = guard(match, dead, labels)
    print("BAR-ATH {}: dead {:.1%}, {} new stoppages, possessions per half {} -> guard {}".format(
        name, s["dead_share"], s["new_stoppages"], s["possessions_per_half"], "PASS" if ok else "FAIL"))
    return ok


def tag_team_agreement(match, labels):
    """Share of BAR-ATH restart tags whose Team is the team v9 has on the ball first after the
    restart (a lower bound on the tags' own accuracy, since v9 errs too), and the tag count."""
    sys.path.insert(0, os.path.join(ROOT, "starter"))
    import load_smart_tagging
    import phases
    from set_pieces import RESTART_TYPES, restart_frame
    agree, n = 0, 0
    for event in load_smart_tagging.load(match.name):
        kind = (event["tags"].get("Type") or "").strip().lower()
        team = phases.TEAM_CODE.get((event["tags"].get("Team") or "").strip().lower())
        if kind not in RESTART_TYPES or team is None:
            continue
        r = restart_frame(match.ball, event["start_frame"], match.fps, match.pitch)
        if r is None:
            continue
        after = labels[r:r + 10 * match.fps]
        alive = after[np.isin(after, ["H", "A"])]
        if len(alive):
            n += 1
            agree += (alive[0] == "H") == (team == 0)
    return agree / max(1, n), n


if __name__ == "__main__":
    import deadball
    import model
    match, codes, p_home, p_dead = probabilities("--rebuild" in sys.argv)
    dead = deadball.spans(match, p_dead, **model.V9_DEAD)
    out = deadball.labels(match, codes, p_home, dead, model.WINDOW, model.V8_SWITCH_COST)
    report("v9", match, dead, out)
    if "--tag-teams" in sys.argv:
        share, n = tag_team_agreement(match, out)
        print("BAR-ATH restart tags: Team agrees with v9's first team after the restart on {:.0%} of {}".format(share, n))
