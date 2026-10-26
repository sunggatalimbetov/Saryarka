"""Team Saryarka: per-frame signals for a learned possession model, from tracking alone.

Every signal is from the home team's point of view, so a model only answers "does home have it?".
Time windows are in seconds, so a 10 fps source (SkillCorner) builds the same signals as a 25 fps
one. Where the ball is missing, distance signals use its last seen spot and `ball_age_s` says how
stale that is.

Team length (front-to-back spread) is measured on outfield players only: the keeper (each team's
deepest player in that frame) dominates any spread, and BAR-ATH tracks 9 or fewer players of a team
on ~10% of live frames and more than 11 (ghost tracks) on 4-8%. With the keeper in, length failed
the transfer check; without, the spread gap passes (and points to the team on the ball 72% of the
time vs 64%), and per team the middle-50% range is used because it shrugs off one stray track.

Which goal each team attacks is inferred from where the teams stand over the surrounding minutes
(DIRECTION_WINDOW_S), so it flips at half-time on any source without needing period metadata.

    .venv/bin/python saryarka/features.py      # transfer check: BAR-ATH vs noisy Metrica summaries
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import phases  # noqa: E402
import touches  # noqa: E402

# Params
NEAR_M = (2.0, 5.0)          # player counts within these distances of the ball
VELOCITY_S = 0.5             # player and team velocities over +/- this
PRESSERS = 3                 # nearest players per team whose closing speed is averaged
DIRECTION_WINDOW_S = 300     # +/- seconds for deciding which goal each team attacks
SMOOTH_S = (1.0, 3.0)        # centred means of the main signals over +/- these
MAX_AGE_S = 10.0             # cap on "seconds since" signals
VOTE_WINDOW_FRAMES = 50      # the v4 vote window, at 25 fps
SMOOTHED = ("near_gap", "count2_gap", "count5_gap", "press_gap", "push_gap", "touch_last")


def centred_mean(x, half):
    """NaN-aware centred mean over +/- `half` frames, per column."""
    x = np.asarray(x, dtype=float)
    flat = x.reshape(len(x), -1)
    ok = ~np.isnan(flat)
    sums = np.vstack([np.zeros((1, flat.shape[1])), np.cumsum(np.where(ok, flat, 0.0), axis=0)])
    counts = np.vstack([np.zeros((1, flat.shape[1])), np.cumsum(ok, axis=0)])
    index = np.arange(len(x))
    low, high = np.maximum(0, index - half), np.minimum(len(x), index + half + 1)
    with np.errstate(invalid="ignore"):
        out = (sums[high] - sums[low]) / (counts[high] - counts[low])
    return out.reshape(x.shape)


def forward_fill(xy):
    """Rows of NaN replaced by the last seen row; leading NaN rows stay NaN."""
    seen = ~np.isnan(xy[:, 0])
    last = np.where(seen, np.arange(len(xy)), 0)
    np.maximum.accumulate(last, out=last)
    out = xy[last].copy()
    out[:np.argmax(seen) if seen.any() else len(xy)] = np.nan
    return out


def seconds_since(event, fps):
    """Seconds since `event` (bool per frame) was last True, capped at MAX_AGE_S."""
    last = np.where(event, np.arange(len(event)), -10 ** 9)
    np.maximum.accumulate(last, out=last)
    return np.minimum((np.arange(len(event)) - last) / fps, MAX_AGE_S)


def velocity(xy, fps, seconds=VELOCITY_S):
    """Centred velocity (m/s) over +/- seconds, same shape as xy; NaN at the edges and gaps."""
    k = max(1, int(round(seconds * fps)))
    out = np.full(xy.shape, np.nan)
    out[k:-k] = (xy[2 * k:] - xy[:-2 * k]) / (2 * k / fps)
    return out


def home_attack_sign(match):
    """Per frame +1 when home attacks towards x = pitch length, -1 when towards x = 0."""
    x = match.players_xy[:, :, 0]
    with np.errstate(invalid="ignore"):
        home = np.nanmean(np.where(match.team_of_slot == 0, x, np.nan), axis=1)
        away = np.nanmean(np.where(match.team_of_slot == 1, x, np.nan), axis=1)
    side = centred_mean(home - away, int(DIRECTION_WINDOW_S * match.fps))
    return np.where(np.nan_to_num(side) <= 0, 1.0, -1.0)  # home stands deeper on the side it defends


def build(match, codes, touch_list, dead_spans, vote_share):
    """{name: (n,) float array} of signals for every frame of `match`."""
    fps, n = match.fps, len(match.ball)
    length, width = match.pitch
    sign = home_attack_sign(match)
    is_home = match.team_of_slot == 0
    ball_seen = ~np.isnan(match.ball[:, 0])
    ball = forward_fill(match.ball[:, :2])
    players = match.players_xy

    dist = np.hypot(players[:, :, 0] - ball[:, None, 0], players[:, :, 1] - ball[:, None, 1])
    dist_home = np.where(is_home, dist, np.nan)
    dist_away = np.where(~is_home, dist, np.nan)
    with np.errstate(all="ignore"):
        near_home = np.nanmin(dist_home, axis=1)
        near_away = np.nanmin(dist_away, axis=1)
    f = {"near_home": near_home, "near_away": near_away, "near_gap": near_away - near_home}
    for radius in NEAR_M:
        home_count = np.sum(dist_home < radius, axis=1).astype(float)
        away_count = np.sum(dist_away < radius, axis=1).astype(float)
        f["count{:g}_gap".format(radius)] = home_count - away_count
    f["ball_age_s"] = seconds_since(ball_seen, fps)

    last = touches.last_touch(touch_list, n, dead_spans)
    touched = np.zeros(n, bool)
    touched[[frame for frame, _ in touch_list]] = True
    f["touch_last"] = last
    f["touch_age_s"] = seconds_since(touched, fps)

    ball_v = velocity(np.where(ball_seen[:, None], match.ball[:, :2], np.nan), fps)
    f["ball_speed"] = np.hypot(ball_v[:, 0], ball_v[:, 1])
    f["ball_v_forward"] = ball_v[:, 0] * sign  # towards home's attacking goal
    f["ball_x_forward"] = (ball[:, 0] - length / 2) * sign / (length / 2)
    f["ball_y_abs"] = np.abs(ball[:, 1] - width / 2) / (width / 2)

    spread = {}
    with np.errstate(all="ignore"):
        for name, mask, team_sign in (("home", is_home, 1.0), ("away", ~is_home, -1.0)):
            team = players[:, mask]
            forward = sign * team_sign  # +1 = towards this team's attacking goal
            f["{}_centre_forward".format(name)] = (np.nanmean(team[:, :, 0], axis=1) - length / 2) * forward / (length / 2)
            f["{}_width".format(name)] = np.nanstd(team[:, :, 1], axis=1)
            depth = np.sort(np.where(np.isnan(team[:, :, 0]), np.inf, (team[:, :, 0] - length / 2) * forward[:, None]),
                            axis=1)
            outfield = np.where(np.isfinite(depth), depth, np.nan)[:, 1:]  # drop the deepest: the keeper
            spread[name] = np.nanstd(outfield, axis=1)
            f["{}_length".format(name)] = (np.nanpercentile(outfield, 75, axis=1)
                                           - np.nanpercentile(outfield, 25, axis=1))
            team_v = velocity(team, fps)
            f["{}_push".format(name)] = np.nanmean(team_v[:, :, 0], axis=1) * forward
            # closing speed of the PRESSERS nearest players: minus d(distance)/dt
            d = np.where(mask, dist, np.nan)
            closing = -velocity(d, fps)
            order = np.argsort(np.where(np.isnan(d), np.inf, d), axis=1)[:, :PRESSERS]
            f["{}_press".format(name)] = np.nanmean(np.take_along_axis(closing, order, axis=1), axis=1)
    f["push_gap"] = f["home_push"] - f["away_push"]
    f["press_gap"] = f["home_press"] - f["away_press"]
    f["width_gap"] = f["home_width"] - f["away_width"]
    f["length_gap"] = spread["home"] - spread["away"]
    f["vote_share"] = vote_share
    f["phase_team"] = phases.signal(match.phase_tags, n)  # Smart Tagging phases (faked on the harness)

    for seconds in SMOOTH_S:
        half = int(round(seconds * fps))
        for name in SMOOTHED:
            f["{}_m{:g}s".format(name, seconds)] = centred_mean(f[name], half)
    return f


def match_features(match, codes=None):
    """build() with the same touches, dead spans and vote share v6 uses."""
    import core
    import viterbi
    if codes is None:
        codes = core.nearest_team(match)
    spans = core.dead_spans(match, goals=True)
    touch_list = touches.detect(match, core.TOUCH_IMPULSE)
    window = int(round(VOTE_WINDOW_FRAMES * match.fps / core.REFERENCE_FPS))
    return build(match, codes, touch_list, spans, viterbi.vote_share(codes, window))


def summary(f, frames):
    """{name: (missing share, p10, p50, p90)} over the given frames."""
    out = {}
    for name, values in f.items():
        v = values[frames]
        ok = ~np.isnan(v)
        q = np.percentile(v[ok], (10, 50, 90)) if ok.any() else (np.nan,) * 3
        out[name] = (1 - ok.mean(), *q)
    return out


if __name__ == "__main__":
    import core
    import external
    import noise
    import noise_stats

    stats = noise_stats.load()
    sims = []
    for game in external.GAMES:
        match, labels, restarts = external.metrica_match(game)
        variant = noise.noisy_match(match, labels, restarts, 0, stats)[0]
        sims.append(summary(match_features(variant), np.isin(labels, ["H", "A"])))
    bar = summary(match_features(core.bar_ath_match()), noise_stats.live_mask())
    print("{:<22} {:>24}   {:>24}   flag".format("signal", "noisy Metrica p10/50/90", "BAR-ATH p10/50/90"))
    for name in bar:
        sim = np.mean([s[name] for s in sims], axis=0)
        spread = max(abs(bar[name][3] - bar[name][1]), 1e-9)
        shift = abs(sim[2] - bar[name][2]) / spread  # median gap in units of BAR-ATH's 10-90 range
        width_ratio = (abs(sim[3] - sim[1]) + 1e-9) / spread
        flag = "CHECK" if shift > 0.25 or not 0.67 < width_ratio < 1.5 else ""
        print("{:<22} {:>7.2f} {:>7.2f} {:>8.2f}   {:>7.2f} {:>7.2f} {:>8.2f}   {}".format(
            name, sim[1], sim[2], sim[3], bar[name][1], bar[name][2], bar[name][3], flag))
