"""DFL open data (Bassek et al. 2025, IDSSE: 7 Bundesliga / 2. Bundesliga matches, CC-BY 4.0) as
harness inputs: a Match plus per-frame answers.

Optical tracking at 25 fps. Every ball frame carries DFL's own BallStatus (1 alive, 0 dead) and
BallPossession (1 home, 2 away), so the answers are mechanical:

  * D where BallStatus is 0
  * otherwise the team BallPossession names

Restarts are where BallStatus goes 0 -> 1, typed from the nearest set-piece event (throw-in,
corner, goal kick, free kick, kick off, penalty) within RESTART_MATCH_S; a kickoff that is not a
half's first is a kickoff after a goal, and the goal sits where that dead stretch began.

Data: huggingface.co/datasets/pysport/idsse-data (downloaded to data/external/dfl, git ignores it).

    .venv/bin/python saryarka/study_dfl.py        # parse, cache and print a label summary per match
"""

import datetime as dt
import os
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from baseline import AWAY, HOME  # noqa: E402
from core import Match  # noqa: E402

# Params
DATA = os.path.join(ROOT, "data", "external", "dfl")
CACHE = os.path.join(DATA, "cache")
FPS = 25
RESTART_MATCH_S = 5.0
GAMES = {"J03WMX": "000001", "J03WN1": "000001", "J03WOH": "000002", "J03WOY": "000002",
         "J03WPY": "000002", "J03WQQ": "000002", "J03WR9": "000002"}
RESTART_EVENTS = {"ThrowIn": "throw-in", "CornerKick": "corner kick", "GoalKick": "goal kick",
                  "FreeKick": "free kick", "Penalty": "free kick", "KickOff": "kick off"}
SECTIONS = ("firstHalf", "secondHalf")

FRAMESET = re.compile(r'<FrameSet GameSection="(\w+)"[^>]*TeamId="([^"]+)" PersonId="([^"]+)"')
FRAME = re.compile(r'<Frame N="(\d+)" T="([^"]+)" X="([-\d.]+)" Y="([-\d.]+)"')
BALL_FLAGS = re.compile(r'BallPossession="(\d)" BallStatus="(\d)"')


def path(game, kind):
    prefix = {"info": "DFL_02_01_matchinformation", "events": "DFL_03_02_events_raw",
              "positions": "DFL_04_03_positions_raw_observed"}[kind]
    return os.path.join(DATA, "{}_DFL-COM-{}_DFL-MAT-{}.xml".format(prefix, GAMES[game], game))


def time(text):
    return dt.datetime.fromisoformat(text).timestamp()


def read_raw(game):
    """Arrays over both halves in play order, parsed once and cached."""
    os.makedirs(CACHE, exist_ok=True)
    cached = os.path.join(CACHE, "{}.npz".format(game))
    if os.path.exists(cached):
        return dict(np.load(cached))
    info = ET.parse(path(game, "info")).getroot()
    home = next(t.get("TeamId") for t in info.iter("Team") if t.get("Role") == "home")
    tracks = {}   # (section, team, person) -> (N array, x, y[, possession, status])
    current, rows = None, []

    def flush():
        if current is not None and rows:
            tracks[current] = np.array(rows, dtype=np.float64)

    with open(path(game, "positions")) as handle:
        for line in handle:
            if line.startswith("<FrameSet"):
                flush()
                m = FRAMESET.search(line)
                current, rows = (m.group(1), m.group(2), m.group(3)), []
            elif line.startswith("<Frame "):
                m = FRAME.search(line)
                row = [int(m.group(1)), time(m.group(2)) if current[1] == "BALL" else 0.0,
                       float(m.group(3)), float(m.group(4))]
                if current[1] == "BALL":
                    flags = BALL_FLAGS.search(line)
                    row += [int(flags.group(1)), int(flags.group(2))]
                rows.append(row)
    flush()
    persons = sorted({(team, person) for _, team, person in tracks if team not in ("BALL", "referee")})
    slot = {p: i for i, p in enumerate(persons)}
    parts = {}
    for section in SECTIONS:
        ball = tracks[(section, "BALL", next(p for s, t, p in tracks if s == section and t == "BALL"))]
        first, n = int(ball[0, 0]), int(ball[-1, 0] - ball[0, 0]) + 1
        b = np.full((n, 6), np.nan)
        b[(ball[:, 0] - first).astype(int)] = ball
        players = np.full((n, len(persons), 2), np.nan)
        for (s, team, person), arr in tracks.items():
            if s != section or (team, person) not in slot:
                continue
            index = (arr[:, 0] - first).astype(int)
            keep = (index >= 0) & (index < n)
            players[index[keep], slot[(team, person)]] = arr[keep][:, 2:4]
        parts[section] = (b, players)
    ball = np.vstack([parts[s][0] for s in SECTIONS])
    players = np.vstack([parts[s][1] for s in SECTIONS])
    half = np.concatenate([np.full(len(parts[s][0]), i + 1, np.int8) for i, s in enumerate(SECTIONS)])
    raw = {"t": ball[:, 1], "ball": ball[:, 2:4].astype(np.float32), "possession": np.nan_to_num(ball[:, 4]).astype(np.int8),
           "status": np.nan_to_num(ball[:, 5]).astype(np.int8), "half": half, "players": players.astype(np.float32),
           "home": np.array([team == home for team, _ in persons])}
    np.savez_compressed(cached, **raw)
    return raw


def events(game):
    """[(unix time, kind)] set-piece events, and [unix time] goals."""
    root = ET.parse(path(game, "events")).getroot()
    out, goals = [], []
    for event in root.iter("Event"):
        when = time(event.get("EventTime"))
        for child in event:
            if child.tag in RESTART_EVENTS:
                out.append((when, child.tag))
            if child.tag == "ShotAtGoal" and any(c.tag == "SuccessfulShot" for c in child):
                goals.append(when)
    return sorted(out), sorted(goals)


def answers(raw):
    labels = np.where(raw["status"] == 1, np.where(raw["possession"] == 1, "H", "A"), "D").astype("<U1")
    labels[(raw["status"] == 1) & ~np.isin(raw["possession"], (1, 2))] = "X"
    return labels


def restarts(game, raw, labels):
    """[(frame, kind, x, y)] at every 0 -> 1 of BallStatus with a matching event, plus goals."""
    pieces, goals = events(game)
    times = np.array([t for t, _ in pieces])
    status, half, t = raw["status"], raw["half"], raw["t"]
    starts = np.flatnonzero((status[1:] == 1) & (status[:-1] == 0)) + 1
    out = []
    half_firsts = {h: int(np.flatnonzero(half == h)[0]) for h in (1, 2)}
    for h, frame in half_firsts.items():
        out.append((frame, "kick off", 52.5, 34.0))
    for frame in starts:
        if not len(times):
            break
        i = int(np.argmin(np.abs(times - t[frame])))
        if abs(times[i] - t[frame]) > RESTART_MATCH_S:
            continue
        kind = RESTART_EVENTS[pieces[i][1]]
        if kind == "kick off":
            dead = frame
            while dead > 0 and status[dead - 1] == 0 and half[dead - 1] == half[frame]:
                dead -= 1
            out.append((dead, "goal", np.nan, np.nan))
            kind = "kick off (after goal)"
        x, y = raw["ball"][frame]
        out.append((int(frame), kind, float(x) + 52.5, float(y) + 34.0))
    return sorted(out)


def dfl_match(game):
    """(Match with perfect tags, answers, true restarts) for a DFL match, at 25 fps."""
    from external import perfect_goal_tags, perfect_tags
    raw = read_raw(game)
    labels = answers(raw)
    events_restarts = restarts(game, raw, labels)
    ball = np.full((len(labels), 3), np.nan)
    ball[:, :2] = raw["ball"] + np.array([52.5, 34.0])
    players = raw["players"].astype(float) + np.array([52.5, 34.0])
    team_of_slot = np.where(raw["home"], HOME, AWAY).astype(np.int8)
    restart_list = [r for r in events_restarts if r[1] != "goal"]
    match = Match("dfl{}".format(game), players, team_of_slot, ball, perfect_tags(restart_list, FPS),
                  None, FPS, (105.0, 68.0), perfect_goal_tags(events_restarts, FPS))
    return match, labels, restart_list


if __name__ == "__main__":
    import external
    for game in GAMES:
        if not os.path.exists(path(game, "positions")):
            print(game, "not downloaded")
            continue
        match, labels, rs = dfl_match(game)
        print(external.summary(match.name, labels, rs))
