"""Team Saryarka: find the moments a player touches the ball, from tracking alone.

A touch (kick, trap, deflection) is a sudden change in the ball's velocity with a player right
next to it. A defender standing close does not change the ball's path; the player who plays it
does, so a touch is stronger evidence of possession than "nearest player".

For every frame t with the ball seen at t - LAG, t and t + LAG, the impulse is
|velocity out - velocity in| in m/s. A touch is a local maximum of the impulse above MIN_IMPULSE
with a player within MAX_DISTANCE; its team is that player's team.

BAR-ATH's tracker smooths the ball (no sharp kick within 0.12 s of a touch), so the velocities are
taken half a second apart. On the harness ball (noise.py) that finds ~48% of logged touches, 87%
of detections are real, 85% with the right team, at a rate (18/min) close to BAR-ATH's (14/min).

    .venv/bin/python saryarka/touches.py      # precision / recall against Metrica's event log
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Params
LAG = 12             # frames each side for the in / out velocity (0.48 s): BAR-ATH's ball is blurred
MIN_IMPULSE = 4.0    # m/s change of ball velocity
MAX_DISTANCE = 3.0   # m from the ball to the touching player
PEAK_WINDOW = 8      # frames: keep only the strongest impulse within +/- this
MATCH_TOLERANCE = 6  # frames: a detection this close to a logged touch counts as found


def impulse(ball, fps, lag=LAG):
    """|v_out - v_in| per frame in m/s, NaN where the ball is not seen at t - lag, t, t + lag."""
    xy = ball[:, :2]
    out = np.full(len(xy), np.nan)
    v_in = (xy[lag:-lag] - xy[:-2 * lag]) * fps / lag
    v_out = (xy[2 * lag:] - xy[lag:-lag]) * fps / lag
    out[lag:-lag] = np.hypot(*(v_out - v_in).T)
    return out


def nearest(ball, players_xy, team_of_slot):
    """(distance in m, team) of the player nearest the ball per frame; inf / -1 without one."""
    d = np.hypot(players_xy[:, :, 0] - ball[:, None, 0], players_xy[:, :, 1] - ball[:, None, 1])
    d = np.where(np.isnan(d), np.inf, d)
    slot = np.argmin(d, axis=1)
    dist = d[np.arange(len(d)), slot]
    return dist, np.where(np.isfinite(dist), team_of_slot[slot], -1)


def detect(match, min_impulse=MIN_IMPULSE, max_distance=MAX_DISTANCE, lag=LAG):
    """[(frame, team)] detected touches, in frame order."""
    imp = impulse(match.ball, match.fps, lag)
    dist, team = nearest(match.ball, match.players_xy, match.team_of_slot)
    strength = np.where((imp >= min_impulse) & (dist <= max_distance), imp, 0.0)
    padded = np.concatenate((np.zeros(PEAK_WINDOW), strength, np.zeros(PEAK_WINDOW)))
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * PEAK_WINDOW + 1)
    peak = (strength > 0) & (strength >= windows.max(axis=1))
    frames = np.flatnonzero(peak)
    keep = np.concatenate(([True], np.diff(frames) > PEAK_WINDOW))  # equal peaks: keep the first
    return [(int(f), int(team[f])) for f in frames[keep]]


def last_touch(touch_list, n, dead_spans):
    """Per frame +1 (home touched last) / -1 (away) / 0 (no touch yet since the last dead span)."""
    out = np.zeros(n)
    marks = np.zeros(n)
    for frame, team in touch_list:
        marks[frame] = 1 if team == 0 else -1
    resets = np.zeros(n, bool)
    for start, stop in dead_spans:
        resets[start:stop] = True
    current = 0.0
    for i in range(n):
        if resets[i]:
            current = 0.0
        elif marks[i]:
            current = marks[i]
        out[i] = current
    return out


def logged_touches(events):
    """[(frame, team)] touches in a Metrica event log: pass kick and reception, shots, recoveries."""
    out = []
    for e in events:
        if e["type"] == "PASS":
            out += [(e["start"], e["team"]), (e["end"], e["team"])]
        elif e["type"] in ("SHOT", "RECOVERY", "BALL LOST"):
            out.append((e["start"], e["team"]))
    return sorted(set(out))


def compare(found, truth, tolerance=MATCH_TOLERANCE):
    """(recall, precision, team accuracy of matched detections)."""
    truth_frames = np.array([f for f, _ in truth])
    truth_team = np.array([t for _, t in truth])
    hit_truth = np.zeros(len(truth), bool)
    hits, right = 0, 0
    for frame, team in found:
        i = np.searchsorted(truth_frames, frame)
        near = [j for j in (i - 1, i) if 0 <= j < len(truth) and abs(truth_frames[j] - frame) <= tolerance]
        if near:
            j = min(near, key=lambda j: abs(truth_frames[j] - frame))
            hits += 1
            hit_truth[j] = True
            right += truth_team[j] == team
    return hit_truth.mean(), hits / max(1, len(found)), right / max(1, hits)


if __name__ == "__main__":
    import external
    for game in external.GAMES:
        match, labels, _ = external.metrica_match(game)
        events = external.read_events(game) if game in external.CSV_GAMES else external.read_events_json(game)
        truth = logged_touches(events)
        found = detect(match)
        recall, precision, team = compare(found, truth)
        print("{}: {} logged, {} found | recall {:.2f} precision {:.2f} team right {:.2f}".format(
            match.name, len(truth), len(found), recall, precision, team))
