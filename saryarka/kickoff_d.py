"""Team Saryarka, quick win 2: the scored gap before the second-half kickoff is a dead ball.

The second half is graded from frame 110808 but kicks off at 113742 (FAQ.md, DATA.md). All 2934
frames in between are D in the ground truth: players walking out, lining up, the ball on the spot.
Quick win 1 leaves them as nearest-player H/A, or X that the scorer back-fills with the first
possession of the half -- either way a wrong label and a phantom possession at the restart.

Both numbers are published as half boundaries, not labels ("they are boundaries, not answers"), and
the Smart Tagging 'Kick Off (Start)' tag for the 2nd half is checked to contain the kickoff, so
the boundary is confirmed by the data rather than taken on trust.

The public 5 minutes are all in the first half, so this cannot move the public score. It only
shows up on the full match.

    .venv/bin/python saryarka/kickoff_d.py      # writes evaluation/results/saryarka_v2_smooth*.csv
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from baseline import RESULTS_DIR, build, smooth, write_submission  # noqa: E402
from games import GAME, half_video_blocks  # noqa: E402
import load_smart_tagging  # noqa: E402

WINDOWS = (50, 125)


def second_half_dead_span(game=GAME):
    """[first graded frame of the 2nd half, 2nd-half kickoff) -- graded, and all of it dead ball."""
    report = load_smart_tagging.check_time_base(load_smart_tagging.load(game))
    if not report["agrees"]:
        raise SystemExit("Smart Tagging kickoff tags do not contain the published kickoffs")
    first_graded = dict((half, first) for half, first, _ in half_video_blocks(game))[2]
    return first_graded, load_smart_tagging.KICKOFF_VIDEO_FRAMES["2nd Half"]


def quick_wins_1_2(codes, window, dead_span):
    labels = smooth(codes, window)
    labels[labels == "D"] = "X"  # quick win 1
    start, stop = dead_span
    labels[start:stop] = "D"  # quick win 2
    return labels


if __name__ == "__main__":
    dead_span = second_half_dead_span()
    print("2nd-half dead span {}..{} ({} frames), confirmed by Smart Tagging".format(
        dead_span[0], dead_span[1] - 1, dead_span[1] - dead_span[0]))
    codes = build()
    for window in WINDOWS:
        labels = quick_wins_1_2(codes, window, dead_span)
        name = "saryarka_v2_smooth{}".format(window)
        path = write_submission(os.path.join(RESULTS_DIR, name + ".csv"), labels)
        print("{:<22} -> {}".format(name, os.path.relpath(path, ROOT)))
