"""Team Saryarka, quick win 3: every Smart Tagging restart is preceded by a dead ball.

Restart tags (throw-in, corner, goal kick, free kick, kickoff after a goal) start while the ball is
still dead: averaged over the 53 non-kickoff set pieces the ball is off the pitch or missing around
the tag start and back in play 5-7 s in, and the two real kickoffs sit 3.3-3.5 s into theirs. So
each tag gives a dead span [tag start, restart), where the restart is the first frame the tracked
ball is inside the pitch and moving. Writing D there ends the previous possession and starts the
next one at the right moment, which is what spellF1 rewards.

The tag is evidence, not an answer (DATA.md): when the ball never visibly restarts within
MAX_SEARCH_S the tag is assumed wrong and skipped, rather than painting seconds of live play D.

    .venv/bin/python saryarka/set_pieces.py     # writes evaluation/results/saryarka_v3_smooth*.csv
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
sys.path.insert(0, HERE)
from baseline import RESULTS_DIR, build, write_submission  # noqa: E402
from games import GAME, GAMES  # noqa: E402
from kickoff_d import quick_wins_1_2, second_half_dead_span  # noqa: E402
import load_smart_tagging  # noqa: E402
import tracking  # noqa: E402

# Params
RESTART_TYPES = {"throw-in", "corner kick", "goal kick", "free kick", "kick off (after goal)"}
MAX_SEARCH_S = 12      # restart must show within this long after the tag starts
SPEED_LAG = 5          # frames over which ball speed is measured (0.2 s)
MIN_SPEED = 3.0        # m/s: a struck or thrown ball, not one rolling on the spot
PITCH_MARGIN = 1.0     # m: the ball on the line still counts as in
WINDOWS = (50, 125)


def restart_frame(ball, start, fps, pitch):
    """First frame at or after `start` where the ball is in the pitch and moving, or None."""
    length, width = pitch
    stop = min(start + MAX_SEARCH_S * fps, len(ball) - SPEED_LAG)
    for frame in range(start, stop):
        x, y = ball[frame, 0], ball[frame, 1]
        ahead = ball[frame + SPEED_LAG]
        if np.isnan(x) or np.isnan(ahead[0]):
            continue
        inside = -PITCH_MARGIN <= x <= length + PITCH_MARGIN and -PITCH_MARGIN <= y <= width + PITCH_MARGIN
        speed = np.hypot(ahead[0] - x, ahead[1] - y) * fps / SPEED_LAG
        if inside and speed >= MIN_SPEED:
            return frame
    return None


def restart_spans(ball, tags, fps, pitch):
    """[(start, stop)] dead spans for restart tags [(start, end, type)], and how many were skipped."""
    spans, skipped = [], 0
    for start, _, kind in tags:
        if kind not in RESTART_TYPES:
            continue
        restart = restart_frame(ball, start, fps, pitch)
        if restart is None:
            skipped += 1
            continue
        spans.append((start, restart))
    return spans, skipped


def smart_tagging_restarts(game=GAME):
    """[(start, end, type)] for every tagged restart in the game's Smart Tagging, type lower case."""
    return [(event["start_frame"], event["end_frame"],
             (event["tags"].get("Type") or "").strip().lower())
            for event in load_smart_tagging.load(game)
            if (event["tags"].get("Type") or "").strip().lower() in RESTART_TYPES]


def dead_spans(game=GAME):
    """[(start, stop)] dead-ball spans before each tagged restart, plus a one-line summary."""
    fps = GAMES[game]["fps"]
    spans, skipped = restart_spans(tracking.ball(game), smart_tagging_restarts(game), fps,
                                   GAMES[game]["pitch"])
    lengths = [(stop - start) / fps for start, stop in spans]
    summary = "{} restart spans, {} tags skipped, dead span median {:.1f} s (max {:.1f} s)".format(
        len(spans), skipped, np.median(lengths), max(lengths))
    return spans, summary


def quick_wins_1_2_3(codes, window, kickoff_span, spans):
    labels = quick_wins_1_2(codes, window, kickoff_span)
    for start, stop in spans:
        labels[start:stop] = "D"  # quick win 3
    return labels


if __name__ == "__main__":
    kickoff_span = second_half_dead_span()
    spans, summary = dead_spans()
    print(summary)
    codes = build()
    for window in WINDOWS:
        name = "saryarka_v3_smooth{}".format(window)
        path = write_submission(os.path.join(RESULTS_DIR, name + ".csv"),
                                quick_wins_1_2_3(codes, window, kickoff_span, spans))
        print("{:<22} -> {}".format(name, os.path.relpath(path, ROOT)))
