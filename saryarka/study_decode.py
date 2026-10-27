"""Decoder studies on the cached out-of-fold v9 probabilities (study_oof.py): where spellF1 is lost,
and leave-one-match-out comparisons of decoder settings against v9 under the agreed rule.

    .venv/bin/python saryarka/study_decode.py breakdown     # step 0: v9's errors by kind
    .venv/bin/python saryarka/study_decode.py a1            # switch cost x X window, v9 dead spans
    .venv/bin/python saryarka/study_decode.py a2            # no forced X in ball dropouts

Every number is held-out: the probabilities come from models that never saw the match, and any
setting is picked leave-one-match-out (harness.lomo). Each match is the mean of TEST_SEEDS.

Decision rule (agreed 27 Oct, fixed before any result): a change is better than v9 when the 23-match
mean goes up AND it wins on at least 16 of 23 AND neither source's mean drops by more than 0.01;
anything that changes dead spans must also pass the BAR-ATH guard (study_bar_ath.py).
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from tune_v8 import TEST_SEEDS  # noqa: E402

# Params
WORKERS = 8
WINDOW = 50                  # v9
SWITCH_COST = 3.0            # v9
SNAP_S = 5.0                 # timing oracle: boundaries within this of the truth's are moved onto it
SHORT_S = 5.0                # short possession / short phantom
MATCH_IOU = 0.3              # lowest IoU the metric credits
BUCKETS_S = (2.0, 5.0, 10.0)
NO_X_WINDOW = 10 ** 6


def case(game, seed):
    """Everything a decoder needs for one held-out (game, seed)."""
    warnings.filterwarnings("ignore")
    import external
    import noise
    import study_oof
    from core import nearest_team
    from harness import to_codes
    match, labels, restarts = external.load(game)
    variant = noise.noisy_match(match, labels, restarts, seed)[0]
    p_home, p_dead = study_oof.load(game, seed)
    return {"game": game, "seed": seed, "match": variant, "labels": labels, "gt": to_codes(labels), "restarts": restarts,
            "codes": nearest_team(variant), "p_home": p_home, "p_dead": p_dead}


def v9_dead(c, **overrides):
    import deadball
    import model
    return deadball.spans(c["match"], c["p_dead"], **{**model.V9_DEAD, **overrides})


def v9_labels(c, dead=None, window=WINDOW, switch_cost=SWITCH_COST):
    import deadball
    return deadball.labels(c["match"], c["codes"], c["p_home"], v9_dead(c) if dead is None else dead,
                           window, switch_cost)


def spell_f1(gt, labels):
    from harness import to_codes
    runs = pm.valid_runs(gt)
    return pm.spell_scores(gt, pm.fill_abstentions(to_codes(labels), runs), runs)["mean_f1"]


def runs_of(mask):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


# -- step 0: where v9 loses spellF1 ----------------------------------------------------------------

def best_overlap(spell, others):
    """(IoU, index) of the same-team spell in `others` overlapping `spell` most, (0, -1) if none."""
    a, b, k = spell
    best = (0.0, -1)
    for j, (c, d, kk) in enumerate(others):
        if kk != k or d <= a or b <= c:
            continue
        iou = (min(b, d) - max(a, c)) / float(max(b, d) - min(a, c))
        best = max(best, (iou, j))
    return best


def bucket(frames, fps=25):
    s = frames / fps
    for i, edge in enumerate(BUCKETS_S):
        if s < edge:
            return i
    return len(BUCKETS_S)


def breakdown_case(job):
    """Oracle gains and error counts for v9 on one (game, seed)."""
    from harness import to_codes
    game, seed = job
    c = case(game, seed)
    gt, labels, fps = c["gt"], c["labels"], c["match"].fps
    runs = pm.valid_runs(gt)
    out = v9_labels(c)
    base = spell_f1(gt, out)
    filled = pm.fill_abstentions(to_codes(out), runs)
    gt_spells = pm.possession_spells(gt, runs)
    pred_spells = pm.possession_spells(filled, runs)
    letter = {pm.H: "H", pm.A: "A"}
    r = {"v9": base}

    # 1. true dead ball: the decoder cut and painted at the answers' D runs instead of ours
    true_dead = runs_of(labels == "D")
    r["oracle_dead"] = spell_f1(gt, v9_labels(c, dead=true_dead))

    # 2. true switch timing: every predicted boundary within SNAP_S of its matched truth spell's
    #    boundary is moved onto it (spells matched by best same-team IoU)
    fixed = filled.copy()
    snap = SNAP_S * fps
    for spell in pred_spells:
        iou, j = best_overlap(spell, gt_spells)
        if j < 0:
            continue
        a, b, k = spell
        ga, gb, _ = gt_spells[j]
        a2 = ga if abs(ga - a) <= snap else a
        b2 = gb if abs(gb - b) <= snap else b
        fixed[a2:b2] = k
    r["oracle_timing"] = spell_f1(gt, np.array([{pm.H: "H", pm.A: "A", pm.D: "D", pm.X: "X"}[v] for v in fixed]))

    # 3. missed short possessions painted in: truth spells < SHORT_S with no prediction at IoU >= 0.3
    added = filled.copy()
    missed = np.zeros(len(BUCKETS_S) + 1, int)
    for spell in gt_spells:
        iou, _ = best_overlap(spell, pred_spells)
        a, b, k = spell
        if iou < 0.5:
            missed[bucket(b - a, fps)] += 1
        if iou < MATCH_IOU and b - a < SHORT_S * fps:
            added[a:b] = k
    r["oracle_short"] = spell_f1(gt, np.array([{pm.H: "H", pm.A: "A", pm.D: "D", pm.X: "X"}[v] for v in added]))

    # 4. phantom flips removed: predicted spells < SHORT_S matching no truth spell at IoU >= 0.3
    #    become X, which the metric forward-fills with the team before
    cleaned = filled.copy()
    phantom = np.zeros(len(BUCKETS_S) + 1, int)
    for spell in pred_spells:
        iou, _ = best_overlap(spell, gt_spells)
        a, b, k = spell
        if iou < 0.5:
            phantom[bucket(b - a, fps)] += 1
        if iou < MATCH_IOU and b - a < SHORT_S * fps:
            cleaned[a:b] = pm.X
    r["oracle_phantom"] = spell_f1(gt, np.array([{pm.H: "H", pm.A: "A", pm.D: "D", pm.X: "X"}[v] for v in cleaned]))

    graded = labels != "X"
    pd_ = (out == "D") & graded
    td = (labels == "D") & graded
    alive = np.isin(labels, ["H", "A"]) & np.isin(out, ["H", "A"])
    r.update({
        "n_gt": len(gt_spells), "n_pred": len(pred_spells),
        "missed": missed.tolist(), "phantom": phantom.tolist(),
        "dead_true": td.sum() / graded.sum(), "dead_pred": pd_.sum() / graded.sum(),
        "dead_precision": (pd_ & td).sum() / max(1, pd_.sum()), "dead_recall": (pd_ & td).sum() / max(1, td.sum()),
        "team_acc_alive": np.mean(out[alive] == labels[alive]),
        "x_share": np.mean(out[graded] == "X"),
    })
    return c["match"].name, seed, r


def mean_by_match(rows):
    """{game: {key: mean over seeds}} from [(game, seed, dict)]."""
    by = {}
    for game, _, r in rows:
        by.setdefault(game, []).append(r)
    return {g: {k: np.mean([r[k] for r in rs], axis=0) for k in rs[0]} for g, rs in by.items()}


def breakdown(pool, games):
    from harness import source
    rows = list(pool.imap_unordered(breakdown_case, [(g, s) for g in games for s in TEST_SEEDS]))
    by = mean_by_match(rows)
    ms = sorted(by, key=lambda m: (source(m) != "metrica", m))

    def mean(key, sel=None):
        return np.mean([by[m][key] for m in ms if sel is None or source(m) == sel], axis=0)

    print("v9 error breakdown, held-out, {} matches x {} seeds".format(len(ms), len(TEST_SEEDS)))
    print("{:<16} {:>7} {:>7} {:>7}".format("", "all", "metrica", "skillc"))
    for key in ("v9", "oracle_dead", "oracle_timing", "oracle_short", "oracle_phantom"):
        vals = [mean(key), mean(key, "metrica"), mean(key, "skillcorner")]
        gain = "" if key == "v9" else "  gain {:+.3f}".format(vals[0] - mean("v9"))
        print("{:<16} {:>7.3f} {:>7.3f} {:>7.3f}{}".format(key, *vals, gain))
    print("\nspells per match: truth {:.0f}, v9 {:.0f}".format(mean("n_gt"), mean("n_pred")))
    labels_ = ["<{:g}s".format(BUCKETS_S[0])] + ["{:g}-{:g}s".format(a, b) for a, b in zip(BUCKETS_S, BUCKETS_S[1:])] + [">{:g}s".format(BUCKETS_S[-1])]
    print("truth spells not found at IoU 0.5, by length: " + ", ".join("{} {:.0f}".format(l, v) for l, v in zip(labels_, mean("missed"))))
    print("v9 spells matching nothing at 0.5, by length: " + ", ".join("{} {:.0f}".format(l, v) for l, v in zip(labels_, mean("phantom"))))
    print("dead share truth {:.1%} v9 {:.1%} | D precision {:.2f} recall {:.2f} | team right on both-alive {:.1%} | X {:.1%}".format(
        mean("dead_true"), mean("dead_pred"), mean("dead_precision"), mean("dead_recall"), mean("team_acc_alive"), mean("x_share")))


# -- decoder grids, judged leave-one-match-out against v9 -------------------------------------------

def grid_case(job):
    """{setting: spellF1} for one (game, seed) over a decoder grid."""
    game, seed, family = job
    c = case(game, seed)
    out = {}
    for setting, labels in FAMILIES[family](c):
        out[setting] = spell_f1(c["gt"], labels)
    return c["match"].name, seed, out


def family_a1(c):
    dead = v9_dead(c)
    for window in (25, 50, 100):
        for cost in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0):
            yield "w{}/c{:g}".format(window, cost), v9_labels(c, dead, window, cost)


def family_a2(c):
    """No forced X: the decoder's team stands through ball dropouts (a window wider than any half
    always sees the ball, so viterbi.labels' X rule never fires)."""
    dead = v9_dead(c)
    for cost in (2.0, 3.0, 4.0):
        yield "noX/c{:g}".format(cost), v9_labels(c, dead, NO_X_WINDOW, cost)


# A3: the team named on a restart tag as a prior for the possession that follows. On BAR-ATH only
# corner, free and goal kick tags name a team (27 tags; throw-ins and kickoffs are N/A), so only
# those get a prior. The harness's fake tags get the team that really restarts, flipped at
# TAG_TEAM_FLIP (set from BAR-ATH before any harness result, study_bar_ath.py --tag-teams), and a
# random team on false tags.
TAG_TEAM_FLIP = None
TAG_MATCH_S = 15.0
TEAM_TAGGED = ("corner kick", "free kick", "goal kick")
PRIOR_SECONDS = (2.0, 4.0, 6.0)
PRIOR_WEIGHTS = (0.5, 1.0)


def tag_teams(c, flip):
    """[(restart_frame, team 0/1)] for the match's restart tags, teams faked as described above."""
    import external
    from set_pieces import RESTART_TYPES, restart_frame
    m, labels = c["match"], c["labels"]
    import zlib
    rng = np.random.default_rng(zlib.crc32("tag-team {} {}".format(c["game"], c["seed"]).encode()))
    fps = m.fps
    out = []
    for start, _, kind in m.restart_tags:
        if kind not in RESTART_TYPES or kind not in TEAM_TAGGED:
            continue
        r = restart_frame(m.ball, start, fps, m.pitch)
        if r is None:
            continue
        true = [f for f, k, _, _ in c["restarts"] if k == kind and 0 <= f - start <= TAG_MATCH_S * fps]
        team = None
        if true:
            after = labels[true[0]:true[0] + 10 * fps]
            alive = after[np.isin(after, ["H", "A"])]
            if len(alive):
                team = 0 if alive[0] == "H" else 1
        if team is None:
            team = int(rng.integers(2))
        elif rng.random() < flip:
            team = 1 - team
        out.append((r, team))
    return out


def prior_labels(c, dead, teams, seconds, weight, switch_cost=SWITCH_COST):
    import core
    import viterbi
    m = c["match"]
    touch = 2 * c["p_home"] - 1
    for r, team in teams:
        k = int(round(seconds * m.fps))
        touch[r:r + k] = np.clip(touch[r:r + k] + (weight if team == 0 else -weight), -1, 1)
    frames = int(round(WINDOW * m.fps / core.REFERENCE_FPS))
    return viterbi.labels(c["codes"], frames, switch_cost * m.fps / core.REFERENCE_FPS, dead,
                          touch=touch, touch_weight=1.0)


def family_a3(c):
    dead = v9_dead(c)
    teams = tag_teams(c, TAG_TEAM_FLIP)
    for seconds in PRIOR_SECONDS:
        for weight in PRIOR_WEIGHTS:
            yield "prior{:g}s/w{:g}".format(seconds, weight), prior_labels(c, dead, teams, seconds, weight)


# A5: one 3-state (H / A / D) decoder instead of thresholded dead spans plus a 2-state decoder.
# Per-frame costs: H max(0, -s) + beta p_dead, A max(0, s) + beta p_dead, D beta (1 - p_dead), with
# s = 2 P(home) - 1; changing team costs the switch cost, going dead costs DEAD_IN, coming back to
# play is free (a restart is a new possession, as v9's cut pieces are). Tagged spans (v8's) stay
# forced D. X rule as v9.
BETAS = (1.0, 2.0, 4.0)
DEAD_INS = (2.0, 5.0, 10.0)


def decode3(cost_h, cost_a, cost_d, switch, dead_in):
    """Cheapest H/A/D path, 0/1/2 per frame."""
    n = len(cost_h)
    back = np.zeros((n, 3), dtype=np.int8)
    acc = np.zeros(3)
    trans = np.array([[0.0, switch, dead_in], [switch, 0.0, dead_in], [0.0, 0.0, 0.0]])  # from x to
    costs = np.column_stack([cost_h, cost_a, cost_d])
    for i in range(n):
        total = acc[:, None] + trans          # [from, to]
        back[i] = np.argmin(total, axis=0)
        acc = total[back[i], np.arange(3)] + costs[i]
    out = np.zeros(n, dtype=np.int8)
    state = int(np.argmin(acc))
    for i in range(n - 1, -1, -1):
        out[i] = state
        state = back[i, state]
    return out


def three_state_labels(c, beta, dead_in, switch_cost=SWITCH_COST, window=WINDOW):
    import core
    m = c["match"]
    scale = m.fps / core.REFERENCE_FPS
    s = 2 * c["p_home"] - 1
    pd_ = c["p_dead"]
    forced = np.zeros(len(s), bool)
    for a, b in core.dead_spans(m, goals=True):
        forced[a:b] = True
    big = 1e6
    cost_h = np.maximum(0, -s) + beta * pd_ + big * forced
    cost_a = np.maximum(0, s) + beta * pd_ + big * forced
    cost_d = beta * (1 - pd_)
    path = decode3(cost_h, cost_a, cost_d, switch_cost * scale, dead_in * scale)
    out = np.array(["H", "A", "D"])[path]
    codes = c["codes"]
    frames = int(round(window * scale))
    seen = np.concatenate(([0], np.cumsum(codes != -1)))
    index = np.arange(len(codes))
    low, high = np.maximum(0, index - frames), np.minimum(len(codes), index + frames + 1)
    out[(seen[high] == seen[low]) & (out != "D")] = "X"
    return out


def family_a5(c):
    for beta in BETAS:
        for dead_in in DEAD_INS:
            yield "3s/b{:g}/in{:g}".format(beta, dead_in), three_state_labels(c, beta, dead_in)


FAMILIES = {"a1": family_a1, "a2": family_a2, "a3": family_a3, "a5": family_a5}


def verdict(new, old, matches):
    from harness import source
    mean_up = np.mean([new[m] for m in matches]) > np.mean([old[m] for m in matches])
    wins = sum(new[m] > old[m] for m in matches)
    drops = {}
    for name in sorted({source(m) for m in matches}):
        ms = [m for m in matches if source(m) == name]
        drops[name] = np.mean([new[m] for m in ms]) - np.mean([old[m] for m in ms])
    ok = mean_up and wins >= 16 and all(d >= -0.01 for d in drops.values())
    return ok, wins, drops


def grid(pool, games, family):
    from harness import lomo, source
    rows = list(pool.imap_unordered(grid_case, [(g, s, family) for g in games for s in TEST_SEEDS]))
    v9_rows = V9_ROWS(pool, games)
    base = {g: np.mean([v for gg, _, v in v9_rows if gg == g]) for g in {r[0] for r in v9_rows}}
    by = {}
    for game, _, r in rows:
        by.setdefault(game, []).append(r)
    table = {g: {k: np.mean([r[k] for r in rs]) for k in rs[0]} for g, rs in by.items()}
    ms = sorted(table, key=lambda m: (source(m) != "metrica", m))
    picked = lomo(table)
    held = {m: picked[m][1] for m in ms}
    print("{}: settings picked leave-one-match-out; v9 = w{}/c{:g}".format(family, WINDOW, SWITCH_COST))
    print("  mean by setting (all 23, in-sample): " + ", ".join(
        "{} {:.3f}".format(k, np.mean([table[m][k] for m in ms])) for k in table[ms[0]]))
    print("  picks: {}".format(sorted({picked[m][0] for m in ms})))
    ok, wins, drops = verdict(held, base, ms)
    print("  held-out all {:.3f} vs v9 {:.3f} | wins {}/{} | {} -> {}".format(
        np.mean(list(held.values())), np.mean(list(base.values())), wins, len(ms),
        ", ".join("{} {:+.3f}".format(k, v) for k, v in drops.items()),
        "BETTER" if ok else "no clear change"))
    return table, held, base


_V9 = {}


def v9_case(job):
    game, seed = job
    c = case(game, seed)
    return c["match"].name, seed, spell_f1(c["gt"], v9_labels(c))


def V9_ROWS(pool, games):
    if not _V9:
        _V9["rows"] = list(pool.imap_unordered(v9_case, [(g, s) for g in games for s in TEST_SEEDS]))
    return _V9["rows"]


def main():
    import multiprocessing
    import external
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("study", choices=("breakdown",) + tuple(FAMILIES))
    study = parser.parse_args().study
    games = external.all_games()
    with multiprocessing.get_context("spawn").Pool(WORKERS) as pool:
        if study == "breakdown":
            breakdown(pool, games)
        else:
            grid(pool, games, study)


if __name__ == "__main__":
    main()
