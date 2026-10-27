"""Team Saryarka, quick win 4: change team only on strong evidence.

A majority vote over a fixed window flips whenever its vote dips to the other side for a moment,
and every flip is a phantom possession that spellF1 punishes. Instead we keep the vote but price
every change of team and pick the cheapest H/A sequence, where

  * each frame costs how strongly its window votes for the other team: the vote share
    (H - A) / (H + A) in the same +/- window as the majority vote, so 0 when the window is
    split or empty and 1 when it is unanimous against
  * every change of team costs SWITCH_COST

so the label moves only when the vote has leaned the other way for long enough to pay for the
switch. This is a two-state Viterbi decoder. The dead spans we already trust (2nd-half kickoff,
tagged restarts) cut the match into pieces decoded on their own, so a new possession after a
restart is free. On the plain per-frame nearest team instead of the vote share it fell short of
the majority vote: short real possessions could not pay for their two switches.

Frames with no ball seen within the window stay X as in quick win 1, and the dead spans stay D.
Window 50 and SWITCH_COST 3 were picked leave-one-match-out on the external matches (tune_v4.py).

    .venv/bin/python saryarka/viterbi.py     # writes evaluation/results/saryarka_v{4,5,6}_smooth*.csv

v5 adds the GOALS-tag dead spans (set_pieces.goal_spans) to the cuts. v6 blends in the team that
touched the ball last (touches.py), weight and costs from tune_v6.py.
"""

import numpy as np

HOME, AWAY, NOBODY = 0, 1, -1  # as in starter/baseline.py


def vote_share(codes, window):
    """(H - A) / (H + A) over a centred +/- window, 0 where the window sees nobody."""
    home = np.concatenate(([0], np.cumsum(codes == HOME)))
    away = np.concatenate(([0], np.cumsum(codes == AWAY)))
    index = np.arange(len(codes))
    low, high = np.maximum(0, index - window), np.minimum(len(codes), index + window + 1)
    h, a = home[high] - home[low], away[high] - away[low]
    return (h - a) / np.maximum(1, h + a)


def decode(share, switch_cost):
    """Cheapest path for one piece, 0 = H and 1 = A per frame. Ties keep the team."""
    n = len(share)
    cost_home, cost_away = np.maximum(0, -share), np.maximum(0, share)
    home, away = 0.0, 0.0
    from_away = np.zeros(n, dtype=bool)  # H at frame i was reached from A at i-1
    from_home = np.zeros(n, dtype=bool)
    for i in range(n):
        from_away[i] = away + switch_cost < home
        from_home[i] = home + switch_cost < away
        home, away = (min(home, away + switch_cost) + cost_home[i],
                      min(away, home + switch_cost) + cost_away[i])
    out = np.zeros(n, dtype=np.int8)
    state = AWAY if away < home else HOME
    for i in range(n - 1, -1, -1):
        out[i] = state
        if state == HOME and from_away[i]:
            state = AWAY
        elif state == AWAY and from_home[i]:
            state = HOME
    return out


def labels(codes, window, switch_cost, dead_spans, touch=None, touch_weight=0.0, abstain=True):
    """H/A from the decoder over the +/- window vote, X where the window sees no ball, D on `dead_spans` [(start, stop)].

    v6: `touch` (+1 / -1 / 0 per frame, the team that touched last) is blended into the evidence
    with `touch_weight`. v10: abstain=False keeps the decoder's team where the window sees no ball.
    """
    out = np.full(len(codes), "X", dtype="<U1")
    share = vote_share(codes, window)
    if touch is not None and touch_weight > 0:
        share = (1 - touch_weight) * share + touch_weight * touch
    cuts = sorted(dead_spans)
    pieces, cursor = [], 0
    for start, stop in cuts:
        pieces.append((cursor, max(cursor, start)))
        cursor = max(cursor, stop)
    pieces.append((cursor, len(codes)))
    for start, stop in pieces:
        if stop > start:
            out[start:stop] = np.where(decode(share[start:stop], switch_cost) == HOME, "H", "A")
    seen = (codes != NOBODY).astype(int)
    seen_cumulative = np.concatenate(([0], np.cumsum(seen)))
    index = np.arange(len(codes))
    low, high = np.maximum(0, index - window), np.minimum(len(codes), index + window + 1)
    if abstain:
        out[seen_cumulative[high] == seen_cumulative[low]] = "X"  # no ball in the window, as in v1
    for start, stop in cuts:
        out[start:stop] = "D"
    return out


if __name__ == "__main__":
    import os
    import sys

    HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, HERE)
    import core

    match = core.bar_ath_match()
    codes = core.nearest_team(match)
    for version in ("v4", "v5", "v6"):
        for window in (50, 125):
            name = "saryarka_{}_smooth{}".format(version, window)
            path = core.write_submission(os.path.join(core.RESULTS_DIR, name + ".csv"),
                                         core.predict(match, version, window, codes))
            print("{:<22} -> {}".format(name, os.path.relpath(path, core.ROOT)))
