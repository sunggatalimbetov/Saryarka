"""Team Saryarka, v9: a learned "is the ball dead?" probability, used to fix and add dead spans.

Smart Tagging's restart tags cover only 15-20% of the true dead time on the harness: a tagged
span starts ~5 s before the restart while the ball went out ~15 s earlier, and most fouls and
offsides carry no tag at all. Giving v8 the true dead spans adds ~+0.15 spellF1 on the harness.

Signals here are team-free and come from tracking alone -- where the ball was last seen relative
to the lines and set-piece spots, how long it has been missing, whether it is still, and whether
the players have slowed to a walk or gathered. No Smart Tagging signal is used: the harness's fake
restart tags start a fixed 5.4 s before the true restart, so their timing would leak the answer.

The probability is used in two stages (agreed 27 Oct, each judged on its own):
  * A, backdating: each tagged dead span's start is walked back while P(dead) > T_LOW, at most
    BACK_MAX_S -- anchored on a tag, so low risk
  * B, new stoppages: P(dead) > T_HIGH for at least MIN_DEAD_S, outside every tagged span,
    widened to the T_LOW edges (hysteresis)
Both feed viterbi.labels' existing cut-and-paint-D logic.

Trained like v8: gradient boosting on the 23 external matches, Metrica and SkillCorner weighted
equally, X frames and SkillCorner's rule-only D frames (external.dead_trust) left out.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from features import centred_mean, forward_fill, seconds_since, velocity  # noqa: E402

# Params
STRIDE = 5               # training rows: every 5th trusted, graded frame
WALK_MPS = 2.0           # a player slower than this is walking
STILL_MPS = 0.5          # a ball slower than this is still
GATHER_M = 6.0           # players within this of the last ball spot count as gathered
DROP_S = 3.0             # player-speed change is measured over this
SMOOTH_S = (1.0, 3.0)    # centred means of the main signals over +/- these
SMOOTHED = ("players_median_speed", "players_walking", "ball_speed", "ball_seen", "gathered")
BACK_MAX_S = 25.0
T_LOW = 0.4
T_HIGH = 0.7
MIN_DEAD_S = 2.0


def spot_distances(x, y, length, width):
    """{name: distance (m)} from (x, y) to the set-piece spots that matter for a restart."""
    corners = np.min([np.hypot(x - cx, y - cy) for cx in (0, length) for cy in (0, width)], axis=0)
    penalty = np.min([np.hypot(x - px, y - width / 2) for px in (11.0, length - 11.0)], axis=0)
    goal_area = np.min([np.hypot(np.maximum(0, np.abs(x - cx) - 2.75), np.maximum(0, np.abs(y - width / 2) - 9.16))
                        for cx in (2.75, length - 2.75)], axis=0)  # the 5.5 m x 18.32 m box
    centre = np.hypot(x - length / 2, y - width / 2)
    return {"to_corner": corners, "to_penalty_spot": penalty, "to_goal_area": goal_area, "to_centre": centre}


def build(match):
    """{name: (n,) float array} of team-free dead-ball signals for every frame of `match`."""
    fps, n = match.fps, len(match.ball)
    length, width = match.pitch
    seen = ~np.isnan(match.ball[:, 0])
    ball = forward_fill(match.ball[:, :2])
    x, y = ball[:, 0], ball[:, 1]
    f = {"ball_seen": seen.astype(float), "ball_age_s": seconds_since(seen, fps)}

    # Where the ball was last seen: lines, outside, set-piece spots.
    to_touch = np.minimum(y, width - y)
    to_goal_line = np.minimum(x, length - x)
    f["to_touchline"] = to_touch           # negative = outside
    f["to_goal_line"] = to_goal_line
    f["outside"] = ((to_touch < 0) | (to_goal_line < 0)).astype(float)
    f.update(spot_distances(x, y, length, width))

    # Ball speed while seen, and as last seen (forward-filled) once it vanishes.
    v = velocity(np.where(seen[:, None], match.ball[:, :2], np.nan), fps)
    speed = np.hypot(v[:, 0], v[:, 1])
    f["ball_speed"] = speed
    f["ball_still"] = np.where(np.isnan(speed), np.nan, (speed < STILL_MPS).astype(float))
    last_v = forward_fill(v)
    f["last_speed"] = np.hypot(last_v[:, 0], last_v[:, 1])
    # heading towards the nearest line: + when moving out of the pitch
    outward_y = np.where(y < width / 2, -last_v[:, 1], last_v[:, 1])
    outward_x = np.where(x < length / 2, -last_v[:, 0], last_v[:, 0])
    f["last_outward_speed"] = np.where(to_touch < to_goal_line, outward_y, outward_x)

    # Players: slowing to a walk, gathering near the ball's last spot, spreading.
    players = match.players_xy
    pv = velocity(players, fps)
    pspeed = np.hypot(pv[:, :, 0], pv[:, :, 1])
    with np.errstate(all="ignore"):
        f["players_median_speed"] = np.nanmedian(pspeed, axis=1)
        f["players_fast"] = np.nanpercentile(pspeed, 90, axis=1)
        tracked = np.sum(~np.isnan(pspeed), axis=1)
        f["players_walking"] = np.where(tracked > 0, np.sum(pspeed < WALK_MPS, axis=1) / np.maximum(1, tracked), np.nan)
        dist = np.hypot(players[:, :, 0] - x[:, None], players[:, :, 1] - y[:, None])
        f["gathered"] = np.sum(dist < GATHER_M, axis=1).astype(float)
        f["nearest_player"] = np.nanmin(dist, axis=1)
        f["players_spread_x"] = np.nanstd(players[:, :, 0], axis=1)
        f["players_spread_y"] = np.nanstd(players[:, :, 1], axis=1)
    lag = int(round(DROP_S * fps))
    med = f["players_median_speed"]
    f["speed_drop"] = np.concatenate((np.full(lag, np.nan), med[:-lag] - med[lag:]))    # before minus now
    f["speed_rise"] = np.concatenate((med[lag:] - med[:-lag], np.full(lag, np.nan)))    # later minus now

    for seconds in SMOOTH_S:
        half = int(round(seconds * fps))
        for name in SMOOTHED:
            f["{}_m{:g}s".format(name, seconds)] = centred_mean(f[name], half)
    return f


def training_rows(f, labels, trust, names, stride=STRIDE):
    """(X, y) on every stride-th graded, trusted frame; y = 1 where the ball is dead."""
    frames = np.flatnonzero((labels != "X") & trust)[::stride]
    return np.column_stack([f[name][frames] for name in names]), (labels[frames] == "D").astype(np.int8)


def probability(fitted, names, f):
    return fitted.predict_proba(np.column_stack([f[name] for name in names]))[:, 1]


def runs(mask):
    """[(start, stop)] of consecutive True runs."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def spans(match, p_dead, stage_a=True, stage_b=True, t_low=T_LOW, t_high=T_HIGH, min_dead_s=MIN_DEAD_S,
          back_max_s=BACK_MAX_S):
    """Dead spans [(start, stop)]: v8's tagged ones, backdated (A), plus new stoppages (B)."""
    import core
    fps = match.fps
    base = core.dead_spans(match, goals=True)
    low = p_dead > t_low
    out = []
    for start, stop in base:
        if stage_a and (match.kickoff_span is None or (start, stop) != tuple(match.kickoff_span)):
            limit = max(0, start - int(round(back_max_s * fps)))
            while start > limit and low[start - 1]:
                start -= 1
        out.append((start, stop))
    if stage_b:
        covered = np.zeros(len(p_dead), bool)
        for start, stop in out:
            covered[start:stop] = True
        low_runs = runs(low)
        high = p_dead > t_high
        for start, stop in runs(high & ~covered):
            if stop - start < min_dead_s * fps:
                continue
            # widen to the surrounding T_LOW run, without reaching into a tagged span
            run = next((r for r in low_runs if r[0] <= start and r[1] >= stop), (start, stop))
            a, b = run
            while a < start and covered[a]:
                a += 1
            while b > stop and covered[b - 1]:
                b -= 1
            out.append((a, b))
    return sorted(out)


def labels(match, codes, p_home, dead, window, switch_cost):
    """v9 labels: the v8 decoder over 2 p_home - 1, cut at the given dead spans."""
    import core
    import viterbi
    frames = int(round(window * match.fps / core.REFERENCE_FPS))
    return viterbi.labels(codes, frames, switch_cost * match.fps / core.REFERENCE_FPS, dead,
                          touch=2 * p_home - 1, touch_weight=1.0)
