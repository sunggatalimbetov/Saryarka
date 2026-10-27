"""Short possessions: why v10 misses them, on the cached out-of-fold probabilities (study_oof.py).

For every truth spell shorter than SHORT_S that v10 matches at no IoU >= 0.3, it asks
  * does the model already lean to the right team inside it (decoder problem), or not (model problem)?
  * does the raw nearest player even belong to that team (is it in the tracking at all)?
  * what is around it: the same team on both sides (a brief interruption), or a hand-over?

    .venv/bin/python saryarka/study_short.py            # the diagnostic, read-only
    .venv/bin/python saryarka/study_short.py evidence   # S1: reshaped evidence x switch cost vs v10

S1 reshapes the evidence the decoder sums, so that confident frames count for more than unsure ones
and a short possession the model is sure of can pay for its two switches, while frames near 0.5
(where phantoms come from) count for less. Shapes: power, e -> sign(e) |e|^gamma, and log-odds,
logit(p) scaled so p = 1 - EPS maps to 1. The restart prior is applied before reshaping, as in
v10. Settings are picked leave-one-match-out and judged against v10 under the agreed rule.
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import study_decode as sd  # noqa: E402
import study_oof  # noqa: E402
from study_decode import pm  # noqa: E402
from tune_v8 import TEST_SEEDS  # noqa: E402

# The out-of-fold v9 probabilities were cached under the v9 code fingerprint; v10 changed only the
# decoder, so they are still the probabilities v10 decodes.
study_oof.CACHE = os.path.join(sd.ROOT, "data", "external", "features-cache", "80c9bd7f2888")

# Params
SHORT_S = 5.0
MATCH_IOU = 0.3
LEAN = (0.0, 0.2)       # mean evidence towards the true team: wrong < 0 <= weak < 0.2 <= clear
V10_PRIOR_S, V10_PRIOR_W, V10_COST = 6.0, 0.5, 4.0


def v10_labels(c):
    return sd.prior_labels(c, sd.v9_dead(c), sd.tag_teams(c, sd.TAG_TEAM_FLIP), V10_PRIOR_S, V10_PRIOR_W, V10_COST)


GAMMAS = (0.5, 1.0, 1.5, 2.0, 3.0)
LOGIT_EPS = (0.02, 0.05)
COSTS = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0)


def reshape(e, kind, value):
    if kind == "pow":
        return np.sign(e) * np.abs(e) ** value
    p = np.clip((e + 1) / 2, value, 1 - value)
    return np.log(p / (1 - p)) / np.log((1 - value) / value)


def shaped_labels(c, dead, teams, kind, value, cost):
    import core
    import restart_prior
    import viterbi
    m = c["match"]
    e = restart_prior.tilt(2 * c["p_home"] - 1, teams, m.fps, V10_PRIOR_S, V10_PRIOR_W)
    frames = int(round(sd.BASE_WINDOW * m.fps / core.REFERENCE_FPS))
    return viterbi.labels(c["codes"], frames, cost * m.fps / core.REFERENCE_FPS, dead,
                          touch=reshape(e, kind, value), touch_weight=1.0)


def short_found(c, out):
    """(short truth spells found at MATCH_IOU, short predicted spells matching nothing)."""
    from harness import to_codes
    gt, fps = c["gt"], c["match"].fps
    runs = pm.valid_runs(gt)
    gs = pm.possession_spells(gt, runs)
    ps = pm.possession_spells(pm.fill_abstentions(to_codes(out), runs), runs)
    found = sum(sd.best_overlap(s, ps)[0] >= MATCH_IOU for s in gs if s[1] - s[0] < SHORT_S * fps)
    phantom = sum(sd.best_overlap(s, gs)[0] < MATCH_IOU for s in ps if s[1] - s[0] < SHORT_S * fps)
    return found, phantom


def evidence_case(job):
    import warnings
    warnings.filterwarnings("ignore")
    game, seed = job
    c = sd.case(game, seed)
    dead, teams = sd.v9_dead(c), sd.tag_teams(c, sd.TAG_TEAM_FLIP)
    base = v10_labels(c)
    out = {"v10": (sd.spell_f1(c["gt"], base),) + short_found(c, base)}
    settings = [("pow", g) for g in GAMMAS] + [("logit", e) for e in LOGIT_EPS]
    for kind, value in settings:
        for cost in COSTS:
            labels = shaped_labels(c, dead, teams, kind, value, cost)
            out["{}{:g}/c{:g}".format(kind, value, cost)] = (sd.spell_f1(c["gt"], labels),) + short_found(c, labels)
    return c["match"].name, seed, out


def evidence_study(pool, games):
    from harness import lomo, source
    rows = list(pool.imap_unordered(evidence_case, [(g, s) for g in games for s in TEST_SEEDS]))
    by = {}
    for game, _, r in rows:
        by.setdefault(game, []).append(r)
    full = {g: {k: np.mean([r[k] for r in rs], axis=0) for k in rs[0]} for g, rs in by.items()}
    ms = sorted(full, key=lambda m: (source(m) != "metrica", m))
    table = {m: {k: v[0] for k, v in full[m].items() if k != "v10"} for m in ms}
    base = {m: full[m]["v10"][0] for m in ms}
    assert abs(table[ms[0]]["pow1/c4"] - base[ms[0]]) < 1e-9, "pow1/c4 must reproduce v10"
    means = {k: np.mean([table[m][k] for m in ms]) for k in table[ms[0]]}
    print("S1 evidence shape x switch cost, top 8 by in-sample all-23 mean (v10 = pow1/c4 {:.3f}):".format(means["pow1/c4"]))
    for k in sorted(means, key=means.get, reverse=True)[:8]:
        f = np.mean([full[m][k][1] for m in ms]); ph = np.mean([full[m][k][2] for m in ms])
        print("  {:<14} {:.3f}  short found {:.0f}/match, short phantoms {:.0f}/match".format(k, means[k], f, ph))
    f0 = np.mean([full[m]["v10"][1] for m in ms]); p0 = np.mean([full[m]["v10"][2] for m in ms])
    print("  {:<14} {:.3f}  short found {:.0f}/match, short phantoms {:.0f}/match".format("v10", np.mean(list(base.values())), f0, p0))
    picked = lomo(table)
    held = {m: picked[m][1] for m in ms}
    print("picks: {}".format(sorted({picked[m][0] for m in ms})))
    ok, wins, drops = sd.verdict(held, base, ms)
    print("held-out all {:.3f} vs v10 {:.3f} | wins {}/{} | {} -> {}".format(
        np.mean(list(held.values())), np.mean(list(base.values())), wins, len(ms),
        ", ".join("{} {:+.3f}".format(k, v) for k, v in drops.items()), "BETTER" if ok else "no clear change"))


def short_case(job):
    import warnings
    warnings.filterwarnings("ignore")
    from harness import to_codes
    game, seed = job
    c = sd.case(game, seed)
    gt, fps = c["gt"], c["match"].fps
    runs = pm.valid_runs(gt)
    out = v10_labels(c)
    filled = pm.fill_abstentions(to_codes(out), runs)
    gt_spells = pm.possession_spells(gt, runs)
    pred_spells = pm.possession_spells(filled, runs)
    evidence = 2 * c["p_home"] - 1
    labels = c["labels"]
    rows = []
    for i, spell in enumerate(gt_spells):
        a, b, k = spell
        if b - a >= SHORT_S * fps:
            continue
        iou, _ = sd.best_overlap(spell, pred_spells)
        sign = 1.0 if k == pm.H else -1.0
        lean = float(np.mean(evidence[a:b] * sign))
        team_code = 0 if k == pm.H else 1
        seen = c["codes"][a:b] >= 0
        nearest_right = float(np.mean(c["codes"][a:b][seen] == team_code)) if seen.any() else np.nan
        before, after = labels[max(0, a - 1)], labels[min(len(labels) - 1, b)]
        other = "A" if k == pm.H else "H"
        if before == other and after == other:
            context = "interruption"   # other team, this team briefly, other team again
        elif after == "D":
            context = "before dead"    # e.g. the touch that puts it out
        elif before == "D":
            context = "after dead"     # e.g. a restart lost at once
        else:
            context = "hand-over"
        rows.append({"len_s": (b - a) / fps, "found": iou >= MATCH_IOU, "lean": lean,
                     "nearest_right": nearest_right, "context": context})
    return c["match"].name, seed, rows, len(gt_spells)


def main():
    import multiprocessing
    import external
    from harness import source
    games = external.all_games()
    if sys.argv[1:] == ["evidence"]:
        with multiprocessing.get_context("spawn").Pool(sd.WORKERS) as pool:
            evidence_study(pool, games)
        return
    with multiprocessing.get_context("spawn").Pool(sd.WORKERS) as pool:
        results = list(pool.imap_unordered(short_case, [(g, s) for g in games for s in TEST_SEEDS]))
    for name in ("all", "metrica", "skillcorner"):
        sel = [r for r in results if name == "all" or source(r[0]) == name]
        rows = [row for r in sel for row in r[2]]
        per = len(sel)  # (game, seed) cases, so counts below are per match
        short = len(rows)
        missed = [r for r in rows if not r["found"]]
        print("\n== {} ({} matches) ==".format(name, len({r[0] for r in sel})))
        print("truth spells per match {:.0f}; short (<{:g}s) {:.0f}; short missed at IoU {:g}: {:.0f} ({:.0%})".format(
            sum(r[3] for r in sel) / per, SHORT_S, short / per, MATCH_IOU, len(missed) / per,
            len(missed) / max(1, short)))
        lean = np.array([r["lean"] for r in missed])
        near = np.array([r["nearest_right"] for r in missed])
        lens = np.array([r["len_s"] for r in missed])
        print("missed short, model's mean lean towards the true team: wrong {:.0%}, weak {:.0%}, clear {:.0%}".format(
            np.mean(lean < LEAN[0]), np.mean((lean >= LEAN[0]) & (lean < LEAN[1])), np.mean(lean >= LEAN[1])))
        print("missed short, nearest player on the true team: median {:.0%} of frames, >50% in {:.0%} of spells".format(
            np.nanmedian(near), np.nanmean(near > 0.5)))
        print("missed short, length: <1s {:.0%}, 1-2s {:.0%}, 2-5s {:.0%}".format(
            np.mean(lens < 1), np.mean((lens >= 1) & (lens < 2)), np.mean(lens >= 2)))
        for ctx in ("interruption", "hand-over", "before dead", "after dead"):
            all_c = [r for r in rows if r["context"] == ctx]
            miss_c = [r for r in missed if r["context"] == ctx]
            ml = np.array([r["lean"] for r in miss_c]) if miss_c else np.zeros(1)
            print("  {:<13} short {:>4.0f}/match, missed {:>4.0f} ({:.0%}), of missed: lean wrong {:.0%} clear {:.0%}".format(
                ctx, len(all_c) / per, len(miss_c) / per, len(miss_c) / max(1, len(all_c)),
                np.mean(ml < LEAN[0]), np.mean(ml >= LEAN[1])))
        found = [r for r in rows if r["found"]]
        if found:
            print("found short, mean lean {:.2f}; missed short, mean lean {:.2f}".format(
                np.mean([r["lean"] for r in found]), np.mean(lean)))


if __name__ == "__main__":
    main()
