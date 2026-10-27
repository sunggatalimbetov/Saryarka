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
PHASE_SPILL = {"build": 0.065, "progression": 0.413, "chances": 1.0, "transition": 0.816}  # phases.py, 27 Oct
PHASE_FLIP = {"build": 0.0, "progression": 0.0, "chances": 0.016, "transition": 0.0}
# Restart tags (27 Oct): on BAR-ATH the ball restarts 0-12 s after a set-piece tag starts
# (percentile, seconds; summary of set_pieces.restart_frame over the 48 tags it finds). A fixed lead
# let a model read the true restart off the fake tag.
RESTART_LEAD_QUANTILES = ((0, 0.0), (10, 1.1), (25, 3.8), (50, 5.8), (75, 7.2), (90, 9.5), (100, 11.7))
MAX_REDRAWS = 20        # fresh dropout streams tried when the live missing share cannot be hit
GOAL_TAG_JITTER_S = 10.0  # GOALS tags start within about this of the goal on BAR-ATH (player speeds)
# Per source: restart counts per match (miss rates are BAR-ATH's tags against these), the ball
# blur and wobble, and whether the source's own dead-ball ball must be removed first. Metrica
# already loses the ball when it is dead; SkillCorner's broadcast keeps it, which would hand any
# model the answer. SkillCorner's blur and drift are fitted by fit_wobble.py --source skillcorner
# (26 Oct, 10 of the 20 matches: blur 0.2 s, drift 0.75 m, error 0.149): its broadcast ball is
# already a little smooth, so it needs less blur than Metrica's to look like BAR-ATH's.
SKILLCORNER_EVENTS = {"throw-in": 40.8, "corner kick": 8.7, "goal kick": 15.2, "free kick": 22.0,
                      "kick off (after goal)": 3.0}  # mean per match over the 20, skillcorner.py
PROFILES = {
    "metrica": {"events": METRICA_EVENTS, "blur_s": BLUR_S, "drift": WOBBLE_DRIFT_M, "blank_dead": False},
    "skillcorner": {"events": SKILLCORNER_EVENTS, "blur_s": 0.2, "drift": 0.75, "blank_dead": True},
}


def profile(match):
    return PROFILES["skillcorner" if match.name.startswith("skillcorner") else "metrica"]


def miss_rates(events=METRICA_EVENTS):
    return {kind: float(np.clip(1 - BAR_ATH_TAGS[kind] / events[kind], 0, 1)) for kind in BAR_ATH_TAGS}


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


def restart_leads(rng, restarts):
    """Seconds from each fake tag's start to its restart: BAR-ATH's spread for set pieces
    (RESTART_LEAD_QUANTILES), a fixed 3.4 s for kickoffs (BAR-ATH: 3.0-3.3 s)."""
    q = np.array(RESTART_LEAD_QUANTILES)
    draws = rng.random(len(restarts))
    return [3.4 if kind.startswith("kick off") else float(np.interp(u, q[:, 0] / 100, q[:, 1]))
            for (_, kind, _, _), u in zip(restarts, draws)]


def noisy_tags(rng, restarts, labels, fps, events=METRICA_EVENTS, lead_rng=None):
    """Smart-Tagging-like tags with per-type misses and false alarms.

    lead_rng: draw each tag's restart lead from BAR-ATH's spread instead of a fixed 5.4 s."""
    from external import perfect_tags
    rates = miss_rates(events)
    kept = [r for r in restarts if rng.random() >= rates.get(r[1], 0.0)]
    tags = perfect_tags(kept, fps, None if lead_rng is None else restart_leads(lead_rng, kept))
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


def noisy_match(match, labels, restarts, seed, stats=None, drift=None, rate=WOBBLE_RATE,
                smooth_s=WOBBLE_SMOOTH_S, blur_s=None):
    """A copy of `match` with BAR-ATH-like ball dropout, dead-ball ball and Smart Tagging.

    drift and blur_s default to the match's source profile (PROFILES)."""
    stats = stats or noise_stats.load()
    source = profile(match)
    drift = source["drift"] if drift is None else drift
    blur_s = source["blur_s"] if blur_s is None else blur_s
    rng = rng_for(match.name, seed)
    ball = match.ball.copy()
    live = np.isin(labels, ["H", "A"])
    dead = labels == "D"
    if source["blank_dead"]:
        ball[dead] = np.nan

    crowd, step = noise_stats.ball_features(ball, match.players_xy)
    sbin = noise_stats.speed_bin(step, match.fps)
    target = stats["live_missing_share"]
    already = np.isnan(ball[:, 0])

    # Solve the hazard scale so the live missing share lands on BAR-ATH's, one rng stream per try.
    # The share is only roughly monotone in the scale (run lengths share the rng), so the try
    # closest to the target is kept rather than the last midpoint. BAR-ATH's longest missing runs
    # (minutes) can make the share jump past the target; then the dropout is redrawn from a fresh
    # stream. Only reached when the first stream fails, so matches it served keep their noise.
    for attempt in range(MAX_REDRAWS + 1):
        stream = rng_for(match.name + "/scale" + ("/{}".format(attempt) if attempt else ""), seed)
        state = stream.bit_generator.state
        low, high, best = 0.0, 50.0, None
        for _ in range(25):
            scale = (low + high) / 2
            stream.bit_generator.state = state
            trial = live_dropout(stream, live, crowd, sbin, stats["hazard"], stats["missing_run_quantiles"], scale)
            share = np.mean((trial | already)[live])
            if best is None or abs(share - target) < abs(best[1] - target):
                best = (scale, share, trial)
            low, high = (scale, high) if share < target else (low, scale)
        if abs(best[1] - target) <= 0.02:
            break
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

    tags = noisy_tags(rng, restarts, labels, match.fps, source["events"],
                      rng_for(match.name + "/tag-lead", seed))  # own stream: other draws unchanged
    goal_tags = [(min(end - 1, max(0, start + int(round(rng.uniform(-1, 1) * GOAL_TAG_JITTER_S * match.fps)))), end)
                 for start, end in match.goal_tags]  # drawn last, so the locked noise above is unchanged
    ball[:, :2] = blur(ball[:, :2], match.fps, blur_s)  # after the ball is placed, before it wobbles
    if drift > 0:  # its own stream, so every other draw above is unchanged by it
        ball[:, :2] += wobble(rng_for(match.name + "/wobble", seed), len(ball), match.fps,
                                 drift, rate, smooth_s)
    import phases  # own stream, so nothing above changes
    phase_tags = phases.delay_starts(
        phases.fake(match, labels, rng_for(match.name + "/phases", seed), PHASE_FLIP, PHASE_SPILL),
        rng_for(match.name + "/phase-delay", seed), match.fps)
    return replace(match, ball=ball, restart_tags=tags, goal_tags=goal_tags, phase_tags=phase_tags), scale
