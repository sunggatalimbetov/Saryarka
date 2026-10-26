"""Summary statistics of how BAR-ATH's tracking loses the ball -- the noise model's only input.

Only aggregates leave this file: rates, a hazard table and run-length quantiles. They are cached
under data/ (git ignores it) rather than committed, so the repository carries nothing derived from
the provided match beyond the code that derives it.

"Live" here means graded frames outside the dead spans we can identify without any ground truth
(the 2nd-half pre-kickoff span and the restart spans from quick win 3). It still contains some
unidentified dead ball, so it slightly overstates live-play dropout -- a known, small bias.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
from games import GAME, half_video_blocks  # noqa: E402

# Params
CROWD_RADIUS = 3.0                # m around the ball
CROWD_BINS = (0, 1, 2, 3)         # players within the radius: 0, 1, 2, 3+
SPEED_EDGES = (2.0, 8.0, 15.0)    # m/s: still/dribbled, passed, driven, struck
SPEED_LAG = 5                     # frames
N_QUANTILES = 101


def run_lengths(mask):
    """Lengths of consecutive True runs."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)


def ball_features(ball, players):
    """Per frame: players within CROWD_RADIUS of the ball (capped) and ball step, metres per frame.

    Without a ball both read as zero, i.e. the uncrowded, slow cell.
    """
    dx = players[:, :, 0] - ball[:, None, 0]
    dy = players[:, :, 1] - ball[:, None, 1]
    with np.errstate(invalid="ignore"):
        crowd = np.minimum(np.nansum((dx * dx + dy * dy) < CROWD_RADIUS ** 2, axis=1), CROWD_BINS[-1])
    step = np.full(len(ball), np.nan)
    step[SPEED_LAG:] = np.hypot(*(ball[SPEED_LAG:, :2] - ball[:-SPEED_LAG, :2]).T) / SPEED_LAG
    return crowd.astype(int), step  # step is metres per frame; caller scales by fps


def speed_bin(step, fps):
    speed = step * fps
    out = np.digitize(np.nan_to_num(speed, nan=0.0), SPEED_EDGES)
    return out


def hazard_table(present, crowd, sbin, live):
    """P(ball missing next frame | present now, crowd bin, speed bin) on live frames."""
    table = np.zeros((len(CROWD_BINS), len(SPEED_EDGES) + 1))
    counts = np.zeros_like(table)
    now, nxt = np.flatnonzero(live[:-1] & live[1:] & present[:-1]), None
    nxt = ~present[now + 1]
    for c in range(table.shape[0]):
        for s in range(table.shape[1]):
            pick = (crowd[now] == c) & (sbin[now] == s)
            counts[c, s] = pick.sum()
            table[c, s] = nxt[pick].mean() if pick.any() else 0.0
    return table, counts


def compute(game=GAME):
    import tracking
    from set_pieces import dead_spans
    from kickoff_d import second_half_dead_span

    ball, players = tracking.ball(game), tracking.players(game)
    fps = 25
    graded = np.zeros(len(ball), bool)
    for _, first, last in half_video_blocks(game):
        graded[first:last + 1] = True
    known_dead = np.zeros(len(ball), bool)
    spans, _ = dead_spans(game)
    for start, stop in spans + [second_half_dead_span(game)]:
        known_dead[start:stop] = True
    live = graded & ~known_dead
    present = ~np.isnan(ball[:, 0])

    crowd, step = ball_features(ball, players)
    table, counts = hazard_table(present, crowd, speed_bin(step, fps), live)
    missing_runs = run_lengths(~present & live)
    present_runs = run_lengths(present & live)
    dead_restart = np.zeros(len(ball), bool)
    for start, stop in spans:
        dead_restart[start:stop] = True
    return {
        "live_missing_share": float(np.mean(~present[live])),
        "dead_present_share": float(np.mean(present[dead_restart])),
        "hazard": table.tolist(),
        "hazard_counts": counts.astype(int).tolist(),
        "missing_run_quantiles": np.quantile(missing_runs, np.linspace(0, 1, N_QUANTILES)).tolist(),
        "missing_run_mean": float(missing_runs.mean()),
        "present_run_mean": float(present_runs.mean()),
        "dead_missing_run_mean": float(run_lengths(~present & dead_restart).mean()),
        "dead_present_run_mean": float(run_lengths(present & dead_restart).mean()),
    }


def load(game=GAME):
    path = os.path.join(ROOT, "data", "{}_noise_stats.json".format(game))
    if not os.path.exists(path):
        with open(path, "w") as handle:
            json.dump(compute(game), handle, indent=1)
    with open(path) as handle:
        return json.load(handle)


if __name__ == "__main__":
    stats = load()
    print("live missing share {:.3f}, dead (restart spans) present share {:.3f}".format(
        stats["live_missing_share"], stats["dead_present_share"]))
    print("live runs: missing mean {:.1f} f, present mean {:.1f} f; dead runs: missing {:.1f} f, present {:.1f} f".format(
        stats["missing_run_mean"], stats["present_run_mean"], stats["dead_missing_run_mean"],
        stats["dead_present_run_mean"]))
    print("missing-run quantiles 10/50/90/99:", [round(stats["missing_run_quantiles"][q]) for q in (10, 50, 90, 99)])
    print("hazard per 1000 frames (rows crowd 0,1,2,3+ ; cols speed <2, 2-8, 8-15, 15+ m/s):")
    for row, count in zip(stats["hazard"], stats["hazard_counts"]):
        print("   ", " ".join("{:6.1f}".format(1000 * h) for h in row), "   n =", count)
