"""Ball arrays from the raw ATD, parsed once and cached under data/ (which git ignores)."""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, GAMES, require, source_frames  # noqa: E402


def ball(game=GAME):
    """(x, y, speed) per video frame: x and y in metres from the corner, NaN where the ball is missing."""
    raw_path, = require(["atd_raw"], game)
    cache = os.path.join(os.path.dirname(raw_path), "{}_ball_cache.npy".format(game))
    if os.path.exists(cache):
        return np.load(cache)

    length, width = GAMES[game]["pitch"]
    out = np.full((source_frames("tactical", game), 3), np.nan)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(":")
            if len(parts) != 3 or parts[2][0] == "N":
                continue
            x, y, speed = (float(v) for v in parts[2].strip().split(",")[:3])
            out[int(parts[0]) - 1] = (x * length, y * width, speed)  # frame_id counts from 1
    np.save(cache, out)
    return out
