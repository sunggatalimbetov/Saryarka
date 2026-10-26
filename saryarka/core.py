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
from dataclasses import dataclass

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
from set_pieces import restart_spans, smart_tagging_restarts  # noqa: E402
import tracking  # noqa: E402

# Params
VERSIONS = ("baseline", "v1", "v2", "v3")
REFERENCE_FPS = 25  # smoothing windows are quoted in frames at 25 fps, as in starter/baseline.py


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


def bar_ath_match(game=GAME):
    meta_path, = require(["atd_meta"], game)
    return Match(game, tracking.players(game), team_of_each_track(meta_path), tracking.ball(game),
                 smart_tagging_restarts(game), second_half_dead_span(game), GAMES[game]["fps"],
                 GAMES[game]["pitch"])


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


def predict(match, version, window, codes=None):
    """Labels for one version: baseline, v1 (+X), v2 (+kickoff D), v3 (+restart D)."""
    if codes is None:
        codes = nearest_team(match)
    labels = smooth(codes, int(round(window * match.fps / REFERENCE_FPS)))
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
