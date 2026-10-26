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


def players(game=GAME):
    """(n_frames, n_slots, 2) player x, y in metres, NaN where a slot is not tracked.

    float64 and scaled exactly as starter/baseline.py scales them, so nearest-player distances
    computed from this array match the baseline's bit for bit.
    """
    raw_path, = require(["atd_raw"], game)
    cache = os.path.join(os.path.dirname(raw_path), "{}_players_cache.npy".format(game))
    if os.path.exists(cache):
        return np.load(cache)

    length, width = GAMES[game]["pitch"]
    out = None
    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(":")
            if len(parts) != 3:
                continue
            groups = parts[1].split(";")
            if out is None:
                out = np.full((source_frames("tactical", game), len(groups), 2), np.nan)
            frame = int(parts[0]) - 1
            for slot, group in enumerate(groups):
                if group[0] == "N":
                    continue
                x, y = group.split(",")[:2]
                out[frame, slot] = (float(x) * length, float(y) * width)
    np.save(cache, out)
    return out
