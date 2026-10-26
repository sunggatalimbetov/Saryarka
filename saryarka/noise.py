"""Make a clean external match as rough as BAR-ATH, from BAR-ATH's summary statistics alone.

Metrica's sample tracking loses the ball almost only when it is dead (92-100% of dead frames, 9-15%
of live ones), which quietly hands any model the answer. BAR-ATH is the other way round: the ball
goes missing during play, in long stretches, mostly when it is slow and crowded, and is still seen
on a quarter of dead-ball frames. So both sides are rebuilt:

  * live play: dropouts start with BAR-ATH's hazard per (crowd, speed) bin, scaled by one factor
    solved so the live missing share equals BAR-ATH's, and last a length drawn from BAR-ATH's
    missing-run distribution
  * dead ball: a ball reappears at the spot the restart is taken from -- where a tracker sees it
    while it is retrieved and placed -- switching on and off with BAR-ATH's dead-ball run lengths
  * Smart Tagging: each restart type is missed at the rate BAR-ATH's tag counts imply against
    Metrica's event counts, and ~10% false tags are added at random live moments
  * GOALS tags: never missed (4 for 4 kickoffs after a goal on BAR-ATH), each start moved by up
    to GOAL_TAG_JITTER_S either way -- an assumption, BAR-ATH has no answers to measure it on

Everything is seeded by (match, seed), so a run repeats exactly.
"""

import os
import sys
import zlib
from dataclasses import replace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import noise_stats  # noqa: E402

# Params
# Restart tags per type on BAR-ATH (Smart Tagging) and the mean logged per Metrica CSV game.
BAR_ATH_TAGS = {"throw-in": 26, "corner kick": 12, "goal kick": 10, "free kick": 5, "kick off (after goal)": 4}
METRICA_EVENTS = {"throw-in": 30.0, "corner kick": 9.0, "goal kick": 18.5, "free kick": 31.0, "kick off (after goal)": 4.5}
FALSE_TAG_SHARE = 6 / 57  # BAR-ATH restart tags that showed no restart at all
TAG_LENGTH_S = 35.0
GOAL_TAG_JITTER_S = 10.0  # GOALS tags start within about this of the goal on BAR-ATH (player speeds)


def miss_rates():
    return {kind: float(np.clip(1 - BAR_ATH_TAGS[kind] / METRICA_EVENTS[kind], 0, 1)) for kind in BAR_ATH_TAGS}


def rng_for(name, seed):
    return np.random.default_rng([zlib.crc32(name.encode()), seed])


def draw_run(rng, quantiles):
    """One missing-run length (frames) from the quantile table, by inverse-CDF interpolation."""
    q = np.asarray(quantiles)
    return max(1, int(round(np.interp(rng.random(), np.linspace(0, 1, len(q)), q))))


def live_dropout(rng, live, crowd, sbin, hazard, quantiles, scale):
    """Boolean mask of frames the simulated tracker loses, on live frames only."""
    hz = np.asarray(hazard)[np.clip(crowd, 0, len(hazard) - 1), sbin] * scale
    draws = rng.random(len(live))
    mask = np.zeros(len(live), bool)
    frame = 0
    n = len(live)
    while frame < n:
        if live[frame] and draws[frame] < hz[frame]:
            length = draw_run(rng, quantiles)
            mask[frame:frame + length] = True
            frame += length
        else:
            frame += 1
    return mask & live


def onoff(rng, n, mean_on, mean_off, start_on):
    """Alternating on/off mask with geometric run lengths of the given means."""
    out = np.zeros(n, bool)
    frame, on = 0, start_on
    while frame < n:
        length = int(rng.geometric(1.0 / (mean_on if on else mean_off)))
        out[frame:frame + length] = on
        frame += length
        on = not on
    return out


def noisy_tags(rng, restarts, labels, fps):
    """Smart-Tagging-like tags with per-type misses and false alarms."""
    from external import perfect_tags
    rates = miss_rates()
    kept = [r for r in restarts if rng.random() >= rates.get(r[1], 0.0)]
    tags = perfect_tags(kept, fps)
    live_frames = np.flatnonzero(np.isin(labels, ["H", "A"]))
    kinds = sorted(BAR_ATH_TAGS)
    for _ in range(int(round(FALSE_TAG_SHARE * len(tags)))):
        start = int(rng.choice(live_frames))
        tags.append((start, start + int(TAG_LENGTH_S * fps), kinds[int(rng.integers(len(kinds)))]))
    return sorted(tags)


def noisy_match(match, labels, restarts, seed, stats=None):
    """A copy of `match` with BAR-ATH-like ball dropout, dead-ball ball and Smart Tagging."""
    stats = stats or noise_stats.load()
    rng = rng_for(match.name, seed)
    ball = match.ball.copy()
    live = np.isin(labels, ["H", "A"])
    dead = labels == "D"

    crowd, step = noise_stats.ball_features(ball, match.players_xy)
    sbin = noise_stats.speed_bin(step, match.fps)
    target = stats["live_missing_share"]
    already = np.isnan(ball[:, 0])

    # Solve the hazard scale so the live missing share lands on BAR-ATH's, one rng stream per try.
    # The share is only roughly monotone in the scale (run lengths share the rng), so the try
    # closest to the target is kept rather than the last midpoint.
    low, high, best = 0.0, 50.0, None
    for _ in range(25):
        scale = (low + high) / 2
        trial = live_dropout(rng_for(match.name + "/scale", seed), live, crowd, sbin,
                             stats["hazard"], stats["missing_run_quantiles"], scale)
        share = np.mean((trial | already)[live])
        if best is None or abs(share - target) < abs(best[1] - target):
            best = (scale, share, trial)
        low, high = (scale, high) if share < target else (low, scale)
    scale, share, drop = best
    if abs(share - target) > 0.02:
        raise ValueError("{} seed {}: live missing share {:.3f} cannot reach {:.3f}".format(
            match.name, seed, share, target))
    ball[drop] = np.nan

    # Dead ball: the ball sits at the restart spot, seen on and off. BAR-ATH's dead runs are cut
    # short by the span edges, so their raw means overstate visibility; the "on" mean is set from
    # the measured share instead, keeping the measured "off" mean.
    share = stats["dead_present_share"]
    dead_on = share / (1 - share) * stats["dead_missing_run_mean"]
    restart_at = {frame: (x, y) for frame, _, x, y in restarts}
    edges = np.diff(np.concatenate(([0], dead.astype(np.int8), [0])))
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        spot = restart_at.get(int(stop))
        if spot is None or np.isnan(spot[0]):
            continue
        seen = onoff(rng, stop - start, dead_on, stats["dead_missing_run_mean"],
                     rng.random() < stats["dead_present_share"])
        frames = np.arange(start, stop)[seen]
        ball[frames, 0], ball[frames, 1], ball[frames, 2] = spot[0], spot[1], np.nan

    tags = noisy_tags(rng, restarts, labels, match.fps)
    goal_tags = [(min(end - 1, max(0, start + int(round(rng.uniform(-1, 1) * GOAL_TAG_JITTER_S * match.fps)))), end)
                 for start, end in match.goal_tags]  # drawn last, so the locked noise above is unchanged
    return replace(match, ball=ball, restart_tags=tags, goal_tags=goal_tags), scale
