"""Team Saryarka, quick win 1: the reference baseline, but abstain (X) where it would say D.

The reference baseline says D whenever no tracked ball falls inside the smoothing window. Most of
those frames are live play with the ball occluded, not a dead ball. Under spellF1 the scorer fills
an X with the last label in that half, so abstaining there means "the last team keeps the ball",
which stops one real possession from being cut into many short ones. macroF1 counts X as a miss,
so expect it to drop; spellF1 is what ranks the board.

    .venv/bin/python saryarka/baseline_x.py            # writes evaluation/results/saryarka_x_*.csv
    .venv/bin/python evaluation/validate_submission.py evaluation/results/saryarka_x_smooth50.csv
    .venv/bin/python evaluation/run_evaluation.py
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "starter"))
from baseline import LABELS, RESULTS_DIR, build, smooth, write_submission  # noqa: E402

# Same windows as the published baseline, so each file compares one-to-one with its baseline twin.
WINDOWS = (0, 25, 50, 125)

if __name__ == "__main__":
    codes = build()
    for window in WINDOWS:
        labels = smooth(codes, window)
        labels[labels == "D"] = "X"  # D only ever means "no ball seen in the window"
        name = "saryarka_x_smooth{}".format(window) if window else "saryarka_x_raw"
        path = write_submission(os.path.join(RESULTS_DIR, name + ".csv"), labels)
        counts = {label: int(np.sum(labels == label)) for label in list(LABELS) + ["X"]}
        print("{:<22} H {:>6}  A {:>6}  X {:>6}  -> {}".format(
            name, counts["H"], counts["A"], counts["X"], os.path.relpath(path, ROOT)))
