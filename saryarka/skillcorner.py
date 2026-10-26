"""SkillCorner open data (A-League 2024/25) as harness inputs: a Match plus per-frame answers.

Broadcast tracking at 10 fps, upsampled here to 25 fps so every frame-based setting of the
pipeline and of the noise model (both fitted on BAR-ATH at 25 fps) means the same thing. Player
and ball positions are interpolated linearly between the 10 fps samples; answers take the nearest
sample.

The answers come from SkillCorner's own per-frame `possession.group`, mechanically, never by hand:

  * outside a period, and on frames with no tracking at all (the broadcast shows a replay or a
    close-up), X -- not scored. Our input has nothing there, and BAR-ATH always has tracking.
  * the group's team where it names one (it stays with the passing team while a pass travels)
  * where it names nobody: D before a period's kickoff, inside a stoppage the dynamic events log
    (a possession with `game_interruption_after`, up to the next possession), and in any unnamed
    stretch of at least STOPPAGE_S (unlogged stoppages); otherwise the team that had it last --
    a short loose ball, as in external.py

Restarts are where a D stretch ends, typed from the logged interruption (throw-in, corner, goal
kick, free kick, kickoff after a goal) or kick off at a period start, placed at the true ball.

Credit: SkillCorner x PySport, github.com/SkillCorner/opendata (please credit SkillCorner).

    .venv/bin/python saryarka/skillcorner.py        # label sanity summary for every match
"""

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from baseline import AWAY, HOME  # noqa: E402
from core import Match  # noqa: E402

# Params
DATA = os.path.join(ROOT, "data", "external", "skillcorner-opendata", "data")
CACHE = os.path.join(ROOT, "data", "external", "skillcorner-cache")
SOURCE_FPS = 10
FPS = 25
STOPPAGE_S = 3.0   # an unnamed stretch this long is a stoppage even if the events log none
RESTART_SLACK_S = 2.0  # the restart hand-over may come this long after the next possession starts
INTERRUPTION_TYPES = {"throw_in": "throw-in", "corner": "corner kick", "goal_kick": "goal kick",
                      "free_kick": "free kick", "goal": "goal"}
GROUPS = {"home team": 0, "away team": 1}


def match_ids():
    with open(os.path.join(DATA, "matches.json")) as handle:
        return sorted(m["id"] for m in json.load(handle))


GAMES = tuple(match_ids()) if os.path.exists(os.path.join(DATA, "matches.json")) else ()


def path(game, suffix):
    return os.path.join(DATA, "matches", str(game), "{}_{}".format(game, suffix))


def read_meta(game):
    with open(path(game, "match.json")) as handle:
        return json.load(handle)


def read_raw(game):
    """10 fps arrays, parsed once from the jsonl and cached: players, ball, group, period, seen."""
    os.makedirs(CACHE, exist_ok=True)
    cached = os.path.join(CACHE, "{}.npz".format(game))
    if os.path.exists(cached):
        return dict(np.load(cached))
    meta = read_meta(game)
    slot_of = {player["id"]: index for index, player in enumerate(meta["players"])}
    with open(path(game, "tracking_extrapolated.jsonl")) as handle:
        rows = [json.loads(line) for line in handle]
    n = max(row["frame"] for row in rows) + 1
    players = np.full((n, len(slot_of), 2), np.nan, dtype=np.float32)
    ball = np.full((n, 3), np.nan, dtype=np.float32)
    group = np.full(n, -1, dtype=np.int8)
    period = np.zeros(n, dtype=np.int8)
    seen = np.zeros(n, dtype=bool)
    for row in rows:
        frame = row["frame"]
        period[frame] = row["period"] or 0
        group[frame] = GROUPS.get(row["possession"]["group"], -1)
        if row["ball_data"]["x"] is not None:
            ball[frame, :2] = row["ball_data"]["x"], row["ball_data"]["y"]
        for player in row["player_data"]:
            players[frame, slot_of[player["player_id"]]] = player["x"], player["y"]
        seen[frame] = bool(row["player_data"])
    raw = {"players": players, "ball": ball, "group": group, "period": period, "seen": seen}
    np.savez_compressed(cached, **raw)
    return raw


def stoppages(game):
    """[(first dead frame, next possession frame, interruption type)] from the dynamic events, 10 fps."""
    events = pd.read_csv(path(game, "dynamic_events.csv"), low_memory=False,
                         usecols=["event_type", "frame_start", "frame_end", "period", "game_interruption_after"])
    possessions = events[events.event_type == "player_possession"].sort_values("frame_start")
    out = []
    rows = list(possessions.itertuples())
    for before, after in zip(rows[:-1], rows[1:]):
        if isinstance(before.game_interruption_after, str) and before.period == after.period:
            kind = before.game_interruption_after.rsplit("_", 1)[0]  # throw_in_for -> throw_in
            out.append((int(before.frame_end) + 1, int(after.frame_start), INTERRUPTION_TYPES.get(kind)))
    return out


def labels_10fps(raw, logged):
    """(per-frame answers, kickoff frame per half) at 10 fps, per the rules in the module docstring."""
    group, period, seen = raw["group"], raw["period"], raw["seen"]
    n = len(group)
    labels = np.full(n, "X", dtype="<U1")
    kickoffs = []
    in_stoppage = np.zeros(n, bool)
    for start, stop, _ in logged:
        in_stoppage[start:stop] = True
    min_run = int(round(STOPPAGE_S * SOURCE_FPS))
    for half in (1, 2):
        frames = np.flatnonzero(period == half)
        if not len(frames):
            continue
        first, stop = int(frames[0]), int(frames[-1]) + 1
        named = np.flatnonzero(group[first:stop] >= 0)
        kickoff = first + (int(named[0]) if len(named) else stop - first)
        unnamed = group[first:stop] < 0
        edges = np.diff(np.concatenate(([0], unnamed.astype(np.int8), [0])))
        long_run = np.zeros(stop - first, bool)
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if b - a >= min_run:
                long_run[a:b] = True
        kickoffs.append(kickoff)
        owner = None
        for i in range(first, stop):
            g = group[i]
            if g >= 0:
                owner = g
                labels[i] = "HA"[g]
            elif i < kickoff or in_stoppage[i] or long_run[i - first] or owner is None:
                labels[i] = "D"
            else:
                labels[i] = "HA"[owner]
    labels[~seen] = "X"
    return labels, kickoffs


def restarts_10fps(labels, logged, kickoffs, ball):
    """[(frame, type, x, y)] restarts at 10 fps, plus a ('goal', frame the ball died) entry before
    each kickoff after a goal.

    A restart is where a D stretch (or a replay, X) hands the ball to a team. Replays split one
    stoppage into several D stretches, so each logged stoppage gets one restart: its last hand-over,
    within RESTART_SLACK_S after the next possession starts. A stoppage the events do not type (an
    injury, a drop ball) gets none: Smart Tagging would not tag it either.
    """
    edges = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    handovers = [int(stop) for start, stop in zip(np.concatenate(([0], edges)), edges)
                 if labels[start] in ("D", "X") and labels[stop] in ("H", "A")]
    slack = int(round(RESTART_SLACK_S * SOURCE_FPS))
    out = []
    for frame in kickoffs:
        out.append((frame, "kick off", 0.0, 0.0))  # the centre spot, SkillCorner's origin
    for start, stop, kind in logged:
        inside = [f for f in handovers if start <= f <= stop + slack]
        if not kind or not inside or inside[-1] in kickoffs:
            continue
        frame = inside[-1]
        if kind == "goal":
            out.append((start, "goal", np.nan, np.nan))
            kind = "kick off (after goal)"
        spot = (0.0, 0.0) if kind.startswith("kick off") else (float(ball[frame, 0]), float(ball[frame, 1]))
        out.append((frame, kind) + spot)
    return sorted(out)


def upsample(raw, labels, restarts, pitch):
    """25 fps copies: positions interpolated (NaN unless both neighbours are seen), labels nearest."""
    n10 = len(labels)
    n = int((n10 - 1) * FPS / SOURCE_FPS) + 1
    t = np.arange(n) * SOURCE_FPS / FPS
    low = np.minimum(np.floor(t).astype(int), n10 - 1)
    high = np.minimum(low + 1, n10 - 1)
    w = (t - low)
    offset = np.array([pitch[0] / 2, pitch[1] / 2])  # SkillCorner's origin is the centre spot

    def interpolate(a):
        a = a.astype(float)
        shape = (-1,) + (1,) * (a.ndim - 1)
        return a[low] * (1 - w).reshape(shape) + a[high] * w.reshape(shape)

    players = interpolate(raw["players"]) + offset
    ball = interpolate(raw["ball"])
    ball[:, :2] += offset
    ball[:, 2] = np.nan  # height is not used; core.Match's third column is speed, unknown here
    nearest = np.minimum(np.round(t).astype(int), n10 - 1)
    # a 10 fps frame f becomes the first 25 fps frame whose nearest sample is f
    first = lambda f: int(np.ceil((f - 0.5) * FPS / SOURCE_FPS))  # noqa: E731
    restarts = [(first(f), kind, x + offset[0], y + offset[1]) for f, kind, x, y in restarts]
    return players, ball, labels[nearest], restarts


def skillcorner_match(game):
    """(Match with perfect tags, answers, true restarts) for a SkillCorner match, at 25 fps."""
    from external import perfect_goal_tags, perfect_tags
    meta = read_meta(game)
    raw = read_raw(game)
    logged = stoppages(game)
    labels, kickoffs = labels_10fps(raw, logged)
    restarts = restarts_10fps(labels, logged, kickoffs, raw["ball"])
    pitch = (float(meta["pitch_length"]), float(meta["pitch_width"]))
    players, ball, labels, events_restarts = upsample(raw, labels, restarts, pitch)
    home = meta["home_team"]["id"]
    team_of_slot = np.array([HOME if p["team_id"] == home else AWAY for p in meta["players"]], dtype=np.int8)
    restarts = [r for r in events_restarts if r[1] != "goal"]
    match = Match("skillcorner{}".format(game), players, team_of_slot, ball, perfect_tags(restarts, FPS),
                  None, FPS, pitch, perfect_goal_tags(events_restarts, FPS))
    return match, labels, restarts


if __name__ == "__main__":
    from external import summary
    for game in GAMES:
        match, labels, restarts = skillcorner_match(game)
        print(summary(match.name, labels, restarts))
        graded = labels != "X"
        print("    pitch {}, ball missing on {:.1f}% of graded frames, {:.1f}% of in-period time has no tracking".format(
            match.pitch, 100 * np.mean(np.isnan(match.ball[graded, 0])),
            100 * np.mean(np.all(np.isnan(match.players_xy[:, :, 0]), axis=1))), flush=True)
