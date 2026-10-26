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
  * ball blur: BAR-ATH's ball reaches real pass speeds but never shows a sharp kick, as if an
    offline tracker filter smoothed it (centred, no delay), so the true ball is blurred that way first
  * ball position: a smooth wandering error (a critically damped spring kicked by random noise)
    so the ball sits as far from the players and wobbles as much in free flight as BAR-ATH's,
    where Metrica's is glued to the player on the ball
  * phases of play: Smart-Tagging-like phase tags placed by the true possession, with spill and
    wrong-team rates calibrated in phases.py
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
BLUR_S = 0.3             # tracker-like blur of the true ball (s), centred; fitted by fit_wobble.py
WOBBLE_DRIFT_M = 0.75    # ball position error, std per axis (m); fitted by fit_wobble.py
WOBBLE_RATE = 4.0        # how fast the error wanders (rad/s, critically damped spring)
WOBBLE_SMOOTH_S = 0.2    # time constant of two passes of exponential smoothing on the error
PHASE_SPILL = {"build": 0.104, "progression": 0.531, "chances": 1.0, "transition": 1.0}  # phases.py
PHASE_FLIP = {"build": 0.0, "progression": 0.0, "chances": 0.265, "transition": 0.077}
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


def smooth_twice(x, fps, seconds):
    """Two forward passes of an exponential moving average: strips the quick shake, keeps drift."""
    if seconds <= 0:
        return x
    alpha = 1 - np.exp(-1.0 / (seconds * fps))
    for _ in range(2):
        out = np.empty_like(x)
        acc = x[0]
        for i in range(len(x)):
            acc = acc + alpha * (x[i] - acc)
            out[i] = acc
        x = out
    return x


BLUR_CENTRED = True      # True: forward then backward pass (offline smoothing, no delay)


def blur(ball_xy, fps, seconds, centred=None):
    """Two-pass exponential smoothing that restarts after every gap (NaN) in the track.

    Causal (two forward passes, delays the ball) or centred (forward then backward, no delay).
    """
    if seconds <= 0:
        return ball_xy
    centred = BLUR_CENTRED if centred is None else centred
    alpha = 1 - np.exp(-1.0 / (seconds * fps))
    x = ball_xy.copy()
    for direction in ((1, -1) if centred else (1, 1)):
        acc = None
        for i in (range(len(x)) if direction == 1 else range(len(x) - 1, -1, -1)):
            if np.isnan(x[i, 0]):
                acc = None
                continue
            acc = x[i].copy() if acc is None else acc + alpha * (x[i] - acc)
            x[i] = acc
    return x


def wobble(rng, n, fps, drift=WOBBLE_DRIFT_M, rate=WOBBLE_RATE, smooth_s=WOBBLE_SMOOTH_S):
    """(n, 2) smooth position error: o'' = -rate^2 o - 2 rate o' + noise, stationary std `drift`,
    then smoothed over `smooth_s` (which shrinks the std a little; the fit absorbs that)."""
    dt = 1.0 / fps
    kick = drift * np.sqrt(4 * rate ** 3 * dt) * rng.standard_normal((n, 2))
    out = np.empty((n, 2))
    o, v = drift * rng.standard_normal(2), drift * rate * rng.standard_normal(2)
    for i in range(n):
        v = v + dt * (-rate * rate * o - 2 * rate * v) + kick[i]
        o = o + dt * v
        out[i] = o
    return smooth_twice(out, fps, smooth_s)


def noisy_match(match, labels, restarts, seed, stats=None, drift=WOBBLE_DRIFT_M, rate=WOBBLE_RATE,
                smooth_s=WOBBLE_SMOOTH_S, blur_s=BLUR_S):
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
    ball[:, :2] = blur(ball[:, :2], match.fps, blur_s)  # after the ball is placed, before it wobbles
    if drift > 0:  # its own stream, so every other draw above is unchanged by it
        ball[:, :2] += wobble(rng_for(match.name + "/wobble", seed), len(ball), match.fps,
                                 drift, rate, smooth_s)
    import phases  # own stream, so nothing above changes
    phase_tags = phases.fake(match, labels, rng_for(match.name + "/phases", seed), PHASE_FLIP, PHASE_SPILL)
    return replace(match, ball=ball, restart_tags=tags, goal_tags=goal_tags, phase_tags=phase_tags), scale
