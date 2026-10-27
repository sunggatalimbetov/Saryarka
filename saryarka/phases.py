"""Team Saryarka: Smart Tagging phases of play as possession evidence, real and faked.

Every attacking phase in BAR-ATH's Smart Tagging has a defending twin over the same interval
(BUILD UP / DEFENDING IN ATTACKING THIRD and so on), so one code per pair is enough. Its Team tag
names the team WITH the ball: against v6 on BAR-ATH it agrees 85% (build up), 80% (progression),
65% (transition), 56% (creating chances, where v6 itself struggles in a crowded box).

Metrica has no Smart Tagging, so the harness fakes it in the same situations: build up while the
team with the ball is in its own third, progression in the middle third, creating chances in the
final third, transition right after the ball changes team in play. Counts per live minute and
lengths copy BAR-ATH, but a fake phase stops when its possession ends (real ones evidently do:
fixed-length fakes ran into the next possession and agreed with v6 far less than real tags), and
spills into the next possession at a per-family rate ("a Build Up can ... cover two", DATA.md).
The spill rate, and a wrong-team rate only where spill cannot get there, are set so fake tags agree
with the harness's v6 as often as real tags agree with BAR-ATH's v6 -- matching what can be
observed, because v6's own errors depend on the situation.

Calibrated 26 Oct, recalibrated 27 Oct after fake starts were delayed 6-12 s (delay_starts):
build up spills 7%, progression 41%, transitions 82%, creating chances always (a fixed 30 s on
BAR-ATH) with 2% wrong team. Known gap: fake tags cover 26% of live play, BAR-ATH's 39% (Metrica possessions are
shorter, so clipped fakes are too). Where the vote share is unsure (|share| < 0.3) fake tags are
right 65-79% of the time vs 56-65% for the vote.

    .venv/bin/python saryarka/phases.py      # calibrate (prints PHASE_SPILL and PHASE_FLIP)
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "starter"))

# Params
FAMILIES = {"BUILD UP": "build", "PROGRESSION": "progression", "CREATING CHANCES": "chances",
            "ATTACKING TRANSITION": "transition"}
TEAM_CODE = {"barcelona": 0, "atletic club": 1}  # BAR-ATH Smart Tagging team names; home first
# BAR-ATH: phases per live minute and duration quantiles p10/p50/p90 (s), summaries only.
RATE_PER_MIN = {"build": 0.221, "progression": 0.664, "chances": 0.302, "transition": 0.241}
DURATION_S = {"build": (13.7, 20.5, 25.2), "progression": (11.1, 17.6, 27.7), "chances": (30.0, 30.0, 30.0),
              "transition": (10.7, 15.4, 20.5)}
BAR_ATH_AGREE_V6 = {"build": 0.852, "progression": 0.797, "chances": 0.555, "transition": 0.645}
ZONE_FAMILY = {0: "build", 1: "progression", 2: "chances"}
MIN_ZONE_S = 2.0     # a zone stay shorter than this does not start a phase
TRANSITION_GAP_S = 1.0  # a change of team with at most this much dead ball in between is a transition
START_DELAY_S = (6.0, 12.0)  # fake tags start this much after their trigger (delay_starts)


def smart_tagging_phases(game=None):
    """[(start, end, family, team)] from the game's Smart Tagging, team 0 home / 1 away."""
    import load_smart_tagging
    from games import GAME
    out = []
    for event in load_smart_tagging.load(game or GAME):
        family = FAMILIES.get(event["code"].strip().upper())
        team = TEAM_CODE.get((event["tags"].get("Team") or "").strip().lower())
        if family and team is not None:
            out.append((event["start_frame"], event["end_frame"], family, team))
    return out


def signal(phase_tags, n):
    """Per frame +1 (tags say home has the ball) / -1 / 0, overlapping tags summed then clipped."""
    out = np.zeros(n)
    for start, end, _, team in phase_tags:
        out[max(0, start):min(n, end)] += 1.0 if team == 0 else -1.0
    return np.clip(out, -1, 1)


def possession_end(labels):
    """Per frame: first frame after the H or A run it is in (the frame itself if not H/A)."""
    out = np.arange(len(labels))
    end = len(labels)
    for i in range(len(labels) - 1, -1, -1):
        if labels[i] not in ("H", "A"):
            end = i
        elif i + 1 < len(labels) and labels[i + 1] != labels[i]:
            end = i + 1
        out[i] = end
    return out


def candidates(match, labels):
    """{family: [(frame, team)]} moments where a phase of that family could start."""
    import features
    fps = match.fps
    length = match.pitch[0]
    team = np.where(labels == "H", 0, np.where(labels == "A", 1, -1))
    ball_x = features.forward_fill(match.ball[:, :2])[:, 0]
    forward = (ball_x - length / 2) / (length / 2) * features.home_attack_sign(match) * np.where(team == 1, -1, 1)
    zone = np.digitize(np.nan_to_num(forward), (-1 / 3, 1 / 3))
    zone = np.round(features.centred_mean(zone.astype(float), fps)).astype(int)
    key = np.where(team >= 0, team * 3 + zone, -1)
    out = {family: [] for family in RATE_PER_MIN}
    edges = np.flatnonzero(np.diff(np.concatenate(([-2], key, [-2]))))
    for start, stop in zip(edges[:-1], edges[1:]):
        if key[start] >= 0 and stop - start >= MIN_ZONE_S * fps:
            out[ZONE_FAMILY[key[start] % 3]].append((int(start), int(key[start] // 3)))
    last_team, last_frame = -1, -10 ** 9
    for frame in np.flatnonzero(team >= 0):
        if team[frame] != last_team and last_team >= 0 and frame - last_frame <= TRANSITION_GAP_S * fps + 1:
            out["transition"].append((int(frame), int(team[frame])))
        last_team, last_frame = team[frame], frame
    return out


def draw_duration(rng, quantiles):
    p10, p50, p90 = quantiles
    return float(np.interp(rng.random(), (0.0, 0.1, 0.5, 0.9, 1.0), (p10 * 0.8, p10, p50, p90, p90 * 1.1)))


def fake(match, labels, rng, flips, spills):
    """[(start, end, family, team)] Smart-Tagging-like phases for a match with known labels.

    Every random draw happens for every tag whatever the rates, so changing a rate changes only
    the tags it applies to (common random numbers, which keeps the calibration smooth).
    """
    fps = match.fps
    live_minutes = np.isin(labels, ["H", "A"]).sum() / fps / 60
    stop = possession_end(labels)
    tags = []
    for family, options in candidates(match, labels).items():
        count = min(len(options), int(round(RATE_PER_MIN[family] * live_minutes)))
        for index in sorted(rng.choice(len(options), size=count, replace=False)) if count else []:
            start, team = options[index]
            end = start + int(round(draw_duration(rng, DURATION_S[family]) * fps))
            spill, flip = rng.random(), rng.random()
            if spill >= spills.get(family, 0.0):
                end = min(end, max(start + 1, int(stop[start])))
            if flip < flips.get(family, 0.0):
                team = 1 - team
            tags.append((start, end, family, team))
    return sorted(tags)


def delay_starts(tags, rng, fps, delay_s=START_DELAY_S):
    """Fake phase tags with each start pushed later by U(delay_s); tags left under 1 s are dropped.

    Real phases start a median 7.6 s from the nearest change of team (vs v6 on BAR-ATH); fakes
    started on the change itself (~4 s), which handed a model the possession's first frame.
    A 6-12 s delay brings them to 7.3 s (27 Oct, summaries only).
    """
    out = []
    for (start, end, family, team), u in zip(tags, rng.random(len(tags))):
        start = start + int(round((delay_s[0] + u * (delay_s[1] - delay_s[0])) * fps))
        if start < end - fps:
            out.append((start, end, family, team))
    return out


def agreement(phase_tags, predicted, live):
    """{family: share of live H/A predicted frames inside its tags that match the tag's team}."""
    hits, totals = {}, {}
    for start, end, family, team in phase_tags:
        lab = predicted[start:end][live[start:end]]
        lab = lab[np.isin(lab, ["H", "A"])]
        hits[family] = hits.get(family, 0) + int(np.sum(lab == "HA"[team]))
        totals[family] = totals.get(family, 0) + len(lab)
    return {family: hits[family] / totals[family] for family in totals if totals[family]}


def calibrate(seeds=2):
    """Spill and flip rates per family so fake tags agree with harness v6 as real ones do with
    BAR-ATH's v6. Agreement is measured with no spill (a_clip) and full spill (a_full) on the same
    draws; spill is interpolated between them. If even full spill agrees too much, the rest is
    made up with flips: a(f) = a (1 - f) + (1 - a) f.
    """
    import core
    import external
    import noise
    import noise_stats
    stats = noise_stats.load()
    clip, full = {}, {}
    all_on = {family: 1.0 for family in RATE_PER_MIN}
    for game in external.GAMES:
        match, labels, restarts = external.metrica_match(game)
        live = np.isin(labels, ["H", "A"])
        for seed in range(seeds):
            predicted = core.predict(noise.noisy_match(match, labels, restarts, seed, stats)[0], "v6", 50)
            for store, spills in ((clip, {}), (full, all_on)):
                tags = delay_starts(fake(match, labels, noise.rng_for(match.name + "/phases-calibrate", seed), {}, spills),
                                    noise.rng_for(match.name + "/phase-delay", seed), match.fps)
                for family, value in agreement(tags, predicted, live).items():
                    store.setdefault(family, []).append(value)
    spills, flips = {}, {}
    for family in RATE_PER_MIN:
        a_clip, a_full = float(np.mean(clip[family])), float(np.mean(full[family]))
        target = BAR_ATH_AGREE_V6[family]
        if target >= a_clip:
            spills[family], flips[family], note = 0.0, 0.0, "fake tags agree less than real even unflipped"
        elif target >= a_full:
            spills[family], flips[family], note = round((a_clip - target) / (a_clip - a_full), 3), 0.0, ""
        else:
            spills[family] = 1.0
            flips[family] = round(float(np.clip((a_full - target) / (2 * a_full - 1), 0, 0.5)), 3)
            note = ""
        print("{:<12} agree with harness v6: clipped {:.3f}, full spill {:.3f}; BAR-ATH real {:.3f} -> "
              "spill {:.3f} flip {:.3f} {}".format(family, a_clip, a_full, target, spills[family], flips[family], note))
    return spills, flips


if __name__ == "__main__":
    spills, flips = calibrate()
    print("PHASE_SPILL =", spills)
    print("PHASE_FLIP =", flips)
