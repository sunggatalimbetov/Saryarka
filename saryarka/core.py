"""Team Saryarka: the whole pipeline on in-memory arrays, so any match can be fed to it.

BAR-ATH is one Match; the external matches the harness scores on are others. Every version of our
method is a function of a Match alone, which is what lets the harness run them on games we hold
the answers to.

    .venv/bin/python saryarka/core.py --check   # BAR-ATH files must equal the scripts' output
"""

import argparse
import filecmp
import os
import sys
from dataclasses import dataclass, field

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
sys.path.insert(0, HERE)
from baseline import (NOBODY, RESULTS_DIR, smooth, team_of_each_track,  # noqa: E402
                      write_submission)
from games import GAME, GAMES, require  # noqa: E402
from kickoff_d import second_half_dead_span  # noqa: E402
from set_pieces import goal_spans, restart_spans, smart_tagging_goals, smart_tagging_restarts  # noqa: E402
import tracking  # noqa: E402
import touches  # noqa: E402
import viterbi  # noqa: E402

# Params
VERSIONS = ("baseline", "v1", "v2", "v3", "v4", "v5", "v6")
REFERENCE_FPS = 25  # smoothing windows are quoted in frames at 25 fps, as in starter/baseline.py
SWITCH_COST = 3.0   # v4: price of a change of team, in frames of unanimous vote; tune_v4.py
TOUCH_WEIGHT = 0.25    # v6: share of the evidence from the team that touched last; tune_v6.py
TOUCH_IMPULSE = 3.0    # v6: m/s velocity change counted as a touch (touches.py, half-second lag)
V6_SWITCH_COST = 5.0   # v6: touches make the vote twitchier, so a change of team costs more


@dataclass
class Match:
    name: str
    players_xy: np.ndarray      # (n, slots, 2) metres, NaN = not tracked
    team_of_slot: np.ndarray    # (slots,) HOME or AWAY
    ball: np.ndarray            # (n, 3) x, y metres and speed, NaN = missing
    restart_tags: list          # [(start, end, type)] restart tags, type lower case
    kickoff_span: tuple         # (start, stop) graded dead span before a kickoff, or None
    fps: int
    pitch: tuple
    goal_tags: list = field(default_factory=list)  # [(start, end)] GOALS tags: goal to kickoff


def bar_ath_match(game=GAME):
    meta_path, = require(["atd_meta"], game)
    return Match(game, tracking.players(game), team_of_each_track(meta_path), tracking.ball(game),
                 smart_tagging_restarts(game), second_half_dead_span(game), GAMES[game]["fps"],
                 GAMES[game]["pitch"], smart_tagging_goals(game))


def nearest_team(match):
    """Per-frame team of the player nearest the ball, NOBODY without a ball or players.

    The vectorised twin of starter/baseline.nearest_player_labels, ties going to the lower slot.
    """
    dx = match.players_xy[:, :, 0] - match.ball[:, None, 0]
    dy = match.players_xy[:, :, 1] - match.ball[:, None, 1]
    distance = dx * dx + dy * dy
    seen = ~np.isnan(distance)
    usable = seen.any(axis=1)
    nearest = np.argmin(np.where(seen, distance, np.inf), axis=1)
    out = np.full(len(distance), NOBODY, dtype=np.int8)
    out[usable] = match.team_of_slot[nearest[usable]]
    return out


def dead_spans(match, goals=False):
    """[(start, stop)] spans v2-v5 paint D: the kickoff gap, the tagged restarts, and (v5) goals."""
    spans = [match.kickoff_span] if match.kickoff_span else []
    spans += restart_spans(match.ball, match.restart_tags, match.fps, match.pitch)[0]
    if goals:
        spans += goal_spans(match.ball, match.goal_tags, match.fps, match.pitch)
    return spans


def predict(match, version, window, codes=None, switch_cost=None, touch_weight=TOUCH_WEIGHT,
            min_impulse=TOUCH_IMPULSE):
    """Labels for one version: baseline, v1 (+X), v2 (+kickoff D), v3 (+restart D), v4 (+switch cost),
    v5 (+goal D), v6 (+last touch)."""
    if codes is None:
        codes = nearest_team(match)
    if switch_cost is None:
        switch_cost = V6_SWITCH_COST if version == "v6" else SWITCH_COST
    frames = int(round(window * match.fps / REFERENCE_FPS))
    if version in ("v4", "v5", "v6"):
        spans = dead_spans(match, goals=version != "v4")
        touch = None
        if version == "v6":
            touch = touches.last_touch(touches.detect(match, min_impulse), len(codes), spans)
        return viterbi.labels(codes, frames, switch_cost * match.fps / REFERENCE_FPS, spans, touch,
                              touch_weight if version == "v6" else 0.0)
    labels = smooth(codes, frames)
    if version == "baseline":
        return labels
    labels[labels == "D"] = "X"
    if version in ("v2", "v3") and match.kickoff_span:
        start, stop = match.kickoff_span
        labels[start:stop] = "D"
    if version == "v3":
        spans, _ = restart_spans(match.ball, match.restart_tags, match.fps, match.pitch)
        for start, stop in spans:
            labels[start:stop] = "D"
    return labels


def check():
    """Rebuild every committed variant through predict() and compare with the scripts' files."""
    match = bar_ath_match()
    codes = nearest_team(match)
    names = [("baseline", w, "baseline_smooth{}".format(w) if w else "baseline_raw") for w in (0, 25, 50, 125)]
    names += [("v1", w, "saryarka_x_smooth{}".format(w) if w else "saryarka_x_raw") for w in (0, 25, 50, 125)]
    names += [(v, w, "saryarka_{}_smooth{}".format(v, w)) for v in ("v2", "v3") for w in (50, 125)]
    tmp = os.path.join(ROOT, "data", "core_check")
    ok = True
    for version, window, name in names:
        path = write_submission(os.path.join(tmp, name + ".csv"), predict(match, version, window, codes))
        same = filecmp.cmp(path, os.path.join(RESULTS_DIR, name + ".csv"), shallow=False)
        ok &= same
        print("{:<24} {}".format(name, "identical" if same else "DIFFERS"))
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    if parser.parse_args().check:
        sys.exit(0 if check() else 1)
