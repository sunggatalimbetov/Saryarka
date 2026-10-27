"""Team Saryarka, v10: the team a restart tag names, as a nudge for the possession that follows.

On BAR-ATH, Smart Tagging names a team on corner, free and goal kick tags (27 tags; throw-ins and
kickoffs say N/A). For PRIOR_S seconds from the restart (set_pieces.restart_frame, the same frame
that ends the tag's dead span) the decoder's evidence 2 P(home) - 1 is shifted by PRIOR_WEIGHT
toward that team. It only tilts the evidence, it never overwrites it.

Picked leave-one-match-out on the harness (study_decode.py a3, 27 Oct) over 2/4/6 s x 0.5/1, with
fake tags carrying the true restarting team flipped 12% (BAR-ATH tags agree with v9's first team
after the restart on 88% of 24): held-out 0.451 vs 0.448 for v9 + A2, wins 19/23.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "starter"))

# Params
TEAM_TAGGED = ("corner kick", "free kick", "goal kick")
PRIOR_S = 6.0
PRIOR_WEIGHT = 0.5


def smart_tagging_teams(game):
    """[(tag start frame, type, team 0 home / 1 away)] for BAR-ATH's team-naming restart tags."""
    import load_smart_tagging
    import phases
    out = []
    for event in load_smart_tagging.load(game):
        kind = (event["tags"].get("Type") or "").strip().lower()
        team = phases.TEAM_CODE.get((event["tags"].get("Team") or "").strip().lower())
        if kind in TEAM_TAGGED and team is not None:
            out.append((event["start_frame"], kind, team))
    return out


def restart_teams(match, tags):
    """[(restart frame, team)] for [(tag start, type, team)], where a restart is found."""
    from set_pieces import restart_frame
    out = []
    for start, _, team in tags:
        r = restart_frame(match.ball, start, match.fps, match.pitch)
        if r is not None:
            out.append((r, team))
    return out


def tilt(evidence, teams, fps, seconds=PRIOR_S, weight=PRIOR_WEIGHT):
    """Copy of `evidence` (+1 home ... -1 away per frame) shifted toward each restart's team."""
    out = evidence.copy()
    k = int(round(seconds * fps))
    for r, team in teams:
        out[r:r + k] = np.clip(out[r:r + k] + (weight if team == 0 else -weight), -1, 1)
    return out
