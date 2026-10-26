"""Metrica sample matches as harness inputs: a Match for our pipeline plus per-frame answers.

The answers come from Metrica's event log, mechanically, never by hand (RULES.md allows labelled
data from other games; the technical document must say this):

  * outside a period, and after the END HALF event, X -- not scored, like halftime in BAR-ATH
  * D from a ball-killing event to the next restart: BALL OUT, a shot that goes out or in, a goal,
    a foul (FAULT RECEIVED), an offside, and the start of each period before its kickoff
  * otherwise the team of the latest on-ball event (pass, set piece, recovery, shot, ball lost),
    held until the OTHER team's next on-ball event -- a loose ball stays with the last team

These are not the organisers' conventions, which come from corrected tracking. The harness only
compares versions of our method against each other with them, it never predicts our real score.

    .venv/bin/python saryarka/external.py        # label sanity summary for every Metrica game
"""

import csv
import json
import os
import xml.etree.ElementTree as ET
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
METRICA = os.path.join(ROOT, "data", "external", "metrica-sample-data", "data")
FPS = 25
PITCH = (105.0, 68.0)  # metres, per the Metrica sample-data README
CSV_GAMES = (1, 2)
EPTS_GAMES = (3,)
GAMES = CSV_GAMES + EPTS_GAMES

ON_BALL = {"PASS", "CARRY", "SET PIECE", "RECOVERY", "SHOT", "BALL LOST"}  # CARRY: game 3 only
# Restart types in Smart Tagging's vocabulary, so the same quick-win code reads both.
SET_PIECE_TYPES = {"THROW IN": "throw-in", "CORNER KICK": "corner kick", "FREE KICK": "free kick",
                   "KICK OFF": "kick off"}
AFTER_GOAL_KICKOFF = "kick off (after goal)"


def ball_killer(kind, subtype):
    """Frame-choice for an event that ends live play: 'end', 'start', or None if it does not."""
    if kind == "BALL OUT":
        return "end"
    if kind == "SHOT" and ("OUT" in subtype.split("-") or "GOAL" in subtype.split("-")):
        return "end"
    if kind == "FAULT RECEIVED":
        return "start"
    if kind == "BALL LOST" and subtype == "OFFSIDE":
        return "end"
    return None


def restart_kind(kind, subtype):
    """Smart-Tagging-style restart type for an event that restarts play, else None."""
    if kind == "SET PIECE":
        return SET_PIECE_TYPES.get(subtype.split("-")[0])  # game 3: FREE KICK-INDIRECT etc.
    if kind in ("PASS", "BALL LOST") and subtype.startswith("GOAL KICK") and "END HALF" not in subtype.split("-"):
        return "goal kick"
    return None


def read_events(game):
    path = os.path.join(METRICA, "Sample_Game_{0}".format(game), "Sample_Game_{0}_RawEventsData.csv".format(game))
    with open(path) as handle:
        rows = list(csv.DictReader(handle))
    events = [{"team": HOME if row["Team"] == "Home" else AWAY, "type": row["Type"],
             "subtype": row["Subtype"] or "", "period": int(row["Period"]),
             "start": int(row["Start Frame"]) - 1, "end": int(row["End Frame"]) - 1,  # 0-based
             "x": float(row["Start X"]) * PITCH[0], "y": float(row["Start Y"]) * PITCH[1]}
              for row in rows]
    return fill_locations(events)


def fill_locations(events):
    """A SET PIECE row carries no location; the pass that takes it, logged next, does."""
    for index, event in enumerate(events):
        if np.isnan(event["x"]):
            later = next((e for e in events[index + 1:index + 4]
                          if e["start"] >= event["start"] and not np.isnan(e["x"])), None)
            if later:
                event["x"], event["y"] = later["x"], later["y"]
    return events


def read_events_json(game):
    """Game 3's JSON events in the same shape as read_events(); subtypes joined with '-'."""
    folder = os.path.join(METRICA, "Sample_Game_{0}".format(game))
    with open(os.path.join(folder, "Sample_Game_{0}_events.json".format(game))) as handle:
        data = json.load(handle)["data"]
    home = epts_home_team(game)
    events = []
    for row in data:
        subtypes = row.get("subtypes") or []
        if isinstance(subtypes, dict):
            subtypes = [subtypes]
        start = row["start"]
        events.append({"team": HOME if row["team"]["id"] == home else AWAY, "type": row["type"]["name"],
                       "subtype": "-".join(sub["name"] for sub in subtypes), "period": int(row["period"]),
                       "start": int(start["frame"]) - 1, "end": int(row["end"]["frame"]) - 1,
                       "x": (start["x"] if start["x"] is not None else np.nan) * PITCH[0],
                       "y": (start["y"] if start["y"] is not None else np.nan) * PITCH[1]})
    return fill_locations(events)


def epts_metadata(game):
    folder = os.path.join(METRICA, "Sample_Game_{0}".format(game))
    return ET.parse(os.path.join(folder, "Sample_Game_{0}_metadata.xml".format(game))).getroot(), folder


def epts_home_team(game):
    root, _ = epts_metadata(game)
    return next(root.iter("Team")).get("id")  # first team listed = home, the EPTS convention


def read_tracking_epts(game):
    """(period, players_xy, team_of_slot, ball) from an EPTS file whose player list changes per block."""
    root, folder = epts_metadata(game)
    home = epts_home_team(game)
    player_ids = [player.get("id") for player in root.iter("Player")]
    team_of_slot = np.array([HOME if player.get("teamId") == home else AWAY for player in root.iter("Player")],
                            dtype=np.int8)
    slot_of = {pid: index for index, pid in enumerate(player_ids)}
    channel_player = {ch.get("id"): ch.get("playerId") for ch in root.iter("PlayerChannel")}
    blocks = []
    for spec in root.iter("DataFormatSpecification"):
        refs = [ref.get("playerChannelId") for ref in spec.iter("PlayerChannelRef")]
        slots = [slot_of[channel_player[ref]] for ref in refs if ref.endswith("_x")]
        blocks.append((int(spec.get("startFrame")), int(spec.get("endFrame")), np.array(slots)))
    params = {p.findtext("Name"): p.findtext("Value") for p in root.iter("ProviderParameter")}
    n = max(end for _, end, _ in blocks)
    period = np.zeros(n, dtype=int)
    for half, key in ((1, "first_half"), (2, "second_half")):
        period[int(params[key + "_start"]) - 1:int(params[key + "_end"])] = half

    length, width = PITCH
    players = np.full((n, len(player_ids), 2), np.nan)
    ball = np.full((n, 3), np.nan)
    block = 0
    with open(os.path.join(folder, "Sample_Game_{0}_tracking.txt".format(game))) as handle:
        for line in handle:
            frame_text, people, ball_text = line.strip().split(":")
            frame = int(frame_text)
            while frame > blocks[block][1]:
                block += 1
            xy = np.array([g.split(",") for g in people.split(";")], dtype=float)
            players[frame - 1, blocks[block][2]] = xy * (length, width)
            bx, by = (float(v) for v in ball_text.split(",")[:2])
            ball[frame - 1, :2] = (bx * length, by * width)
    return period, players, team_of_slot, ball


def read_tracking(game):
    """(period per frame, players_xy, team_of_slot, ball) from the two Metrica tracking CSVs."""
    folder = os.path.join(METRICA, "Sample_Game_{0}".format(game))
    length, width = PITCH
    frames, slots, teams = None, [], []
    for side, code in (("Home", HOME), ("Away", AWAY)):
        table = pd.read_csv(os.path.join(folder, "Sample_Game_{0}_RawTrackingData_{1}_Team.csv".format(game, side)),
                            skiprows=2)
        values = table.to_numpy(dtype=float)
        if frames is None:
            frames = values[:, :2]  # period, frame
            ball = values[:, -2:]
        n_players = (values.shape[1] - 5) // 2  # period, frame, time ... ball x, ball y
        xy = values[:, 3:3 + 2 * n_players].reshape(len(values), n_players, 2)
        slots.append(xy)
        teams += [code] * n_players
    players = np.concatenate(slots, axis=1) * np.array([length, width])
    ball_xyz = np.column_stack([ball * np.array([length, width]), np.full(len(ball), np.nan)])
    order = frames[:, 1].astype(int) - 1
    if not np.array_equal(order, np.arange(len(order))):
        raise ValueError("game {}: tracking frames are not 1..n in order".format(game))
    return frames[:, 0].astype(int), players, np.array(teams, dtype=np.int8), ball_xyz


def labels_from_events(events, period, n_frames):
    """Per-frame H/A/D/X answers and the true restarts [(frame, type, x, y)], per the rules above."""
    labels = np.full(n_frames, "X", dtype="<U1")
    restarts = []
    by_period = {}
    for event in events:
        by_period.setdefault(event["period"], []).append(event)

    for half, half_events in sorted(by_period.items()):
        frames = np.flatnonzero(period == half)
        first, stop = int(frames[0]), int(frames[-1]) + 1
        end_half = [e["start"] for e in half_events
                    if e["type"] == "BALL LOST" and "END HALF" in e["subtype"].split("-")]  # also FORCED-END HALF
        stop = min([stop] + [f + 1 for f in end_half])

        state, owner, cursor = "D", None, first  # every half starts dead, before its kickoff
        goal_pending = False  # per half, so a goal ending one half never relabels the next kickoff
        for event in sorted(half_events, key=lambda e: e["start"]):
            kind, subtype = event["type"], event["subtype"]
            restart = restart_kind(kind, subtype)
            if restart:
                if restart == "kick off" and goal_pending:
                    restart = AFTER_GOAL_KICKOFF
                goal_pending = False
                restarts.append((event["start"], restart, event["x"], event["y"]))
            # Any on-ball event revives a dead ball, logged restart or not: play has visibly resumed.
            # One that began before the ball died (game 3 logs carries overlapping the ball-out)
            # cannot.
            if event["start"] < cursor:
                pass
            elif kind in ON_BALL and (state == "D" or event["team"] != owner):
                labels[cursor:event["start"]] = "D" if state == "D" else "HA"[owner]
                state, owner, cursor = "alive", event["team"], event["start"]
            killer = ball_killer(kind, subtype)
            if killer and state == "alive" and event[killer] + (killer == "end") >= cursor:
                frame = event[killer] + 1 if killer == "end" else event["start"]
                frame = max(frame, cursor)
                labels[cursor:frame] = "HA"[owner]
                state, cursor = "D", frame
                if kind in ("SHOT", "BALL OUT") and "GOAL" in subtype.split("-"):
                    restarts.append((frame, "goal", np.nan, np.nan))
                    goal_pending = True
        labels[cursor:stop] = "D" if state == "D" or owner is None else "HA"[owner]
        labels[stop:frames[-1] + 1] = "X"
    return labels, [restart for restart in restarts if restart[1] != "goal"]


def perfect_tags(restarts, fps=FPS):
    """Restarts as Smart-Tagging-like tags: 35 s long, starting where BAR-ATH's tags start.

    On BAR-ATH the ball restarts a median 5.4 s after a set-piece tag starts, and 3.3-3.5 s
    after a kickoff tag starts. Those two summary numbers place the fake tags.
    """
    tags = []
    for frame, kind, _, _ in restarts:
        lead, length = (3.4, 5.0) if kind.startswith("kick off") else (5.4, 35.0)
        start = max(0, frame - int(round(lead * fps)))
        tags.append((start, start + int(round(length * fps)), kind))
    return tags


def metrica_match(game):
    """(Match with perfect tags, answers, true restarts) for a Metrica sample game."""
    if game in EPTS_GAMES:
        period, players, team_of_slot, ball = read_tracking_epts(game)
        events = read_events_json(game)
    else:
        period, players, team_of_slot, ball = read_tracking(game)
        events = read_events(game)
    labels, restarts = labels_from_events(events, period, len(period))
    match = Match("metrica{}".format(game), players, team_of_slot, ball, perfect_tags(restarts),
                  None, FPS, PITCH)
    return match, labels, restarts


def summary(name, labels, restarts, fps=FPS):
    graded = labels != "X"
    edges = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    runs = np.split(labels, edges)
    spells = [r for r in runs if r[0] in "HA"]
    kinds = {}
    for _, kind, _, _ in restarts:
        kinds[kind] = kinds.get(kind, 0) + 1
    return ("{}: graded {:.1f} min, D {:.1f}%, H share of alive {:.1f}%, {} possessions, mean {:.1f} s\n"
            "    restarts {}".format(
                name, graded.sum() / fps / 60, 100 * np.mean(labels[graded] == "D"),
                100 * np.mean(labels[graded & (labels != "D")] == "H"), len(spells),
                np.mean([len(s) for s in spells]) / fps, dict(sorted(kinds.items()))))


if __name__ == "__main__":
    for game in GAMES:
        match, labels, restarts = metrica_match(game)
        print(summary(match.name, labels, restarts))
        print("    ball missing on {:.1f}% of graded frames".format(
            100 * np.mean(np.isnan(match.ball[labels != "X", 0]))))
