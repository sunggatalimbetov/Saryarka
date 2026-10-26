"""Team Saryarka, v7: a learned per-frame possession probability fed to the v4 decoder.

A logistic regression over features.py's signals answers "does home have the ball?" on live frames.
Home and away are arbitrary (and BAR-ATH's home side is wider all match), so every training frame is
also shown with the teams swapped and the answer flipped, and a prediction averages both views:
the model cannot learn "home usually has it". The probability p becomes the decoder's evidence
2p - 1 in place of the vote share; dead spans, X and the switch cost work as in v6.

Trained only on external matches (labels from their events); never on BAR-ATH.

Held-out on the harness (tune_v7.py): per-frame right team 85-87% vs 71% for the vote share;
spellF1 0.407 vs v6 0.355, wins 3/3. Without the phase tags 0.402, so the gain does not rest on
the faked tags. Largest weights: vote share, team width gap, phase tags, ball moving forward,
length gap, team push.

    .venv/bin/python saryarka/model.py      # trains, writes evaluation/results/saryarka_v7_*.csv
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Params
STRIDE = 5            # use every 5th live frame for training: neighbours are near copies
C = 1.0               # inverse regularisation strength
# Signals that change sign when home and away swap; home_* <-> away_* are swapped; the rest keep.
SIGN_FLIP = ("near_gap", "count2_gap", "count5_gap", "touch_last", "ball_v_forward", "ball_x_forward",
             "push_gap", "press_gap", "width_gap", "length_gap", "vote_share", "phase_team")


def mirror(f):
    """The same frames seen with home and away swapped."""
    out = {}
    for name, values in f.items():
        base = name.split("_m")[0] if "_m" in name and name.rsplit("_m", 1)[1].endswith("s") else name
        if name.startswith("home_"):
            out["away_" + name[5:]] = values
        elif name.startswith("away_"):
            out["home_" + name[5:]] = values
        elif name == "near_home":
            out["near_away"] = values
        elif name == "near_away":
            out["near_home"] = values
        elif base in SIGN_FLIP:
            out[name] = -values
        else:
            out[name] = values
    return out


def matrix(f, names):
    return np.column_stack([f[name] for name in names])


def train(feature_sets, label_sets, stride=STRIDE):
    """Fit on [(features dict, labels array)] from external matches; returns (model, names)."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    names = sorted(feature_sets[0])
    rows, targets = [], []
    for f, labels in zip(feature_sets, label_sets):
        frames = np.flatnonzero(np.isin(labels, ["H", "A"]))[::stride]
        home = (labels[frames] == "H").astype(int)
        rows += [matrix(f, names)[frames], matrix(mirror(f), names)[frames]]
        targets += [home, 1 - home]
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          LogisticRegression(C=C, max_iter=2000))
    model.fit(np.vstack(rows), np.concatenate(targets))
    return model, names


def home_probability(model, names, f):
    """Per-frame P(home has the ball), averaged over the normal and the mirrored view."""
    p = model.predict_proba(matrix(f, names))[:, 1]
    q = model.predict_proba(matrix(mirror(f), names))[:, 1]
    return (p + 1 - q) / 2


def labels(match, codes, probability, window, switch_cost):
    """v7 labels: the v4 decoder over 2p - 1, same dead spans and X rule as v6."""
    import core
    import viterbi
    frames = int(round(window * match.fps / core.REFERENCE_FPS))
    spans = core.dead_spans(match, goals=True)
    return viterbi.labels(codes, frames, switch_cost * match.fps / core.REFERENCE_FPS, spans,
                          touch=2 * probability - 1, touch_weight=1.0)


def weights(model, names, top=12):
    """[(name, weight)] largest standardised weights."""
    coef = model[-1].coef_[0]
    order = np.argsort(-np.abs(coef))[:top]
    return [(names[i], float(coef[i])) for i in order]


# Final settings, picked by tune_v7.py (26 Oct): window 50, switch cost 2 (costs 1-3 within 0.005).
WINDOW = 50
SWITCH_COST = 2.0
TRAIN_SEEDS = (100, 101, 102)


def train_on_external(seeds=TRAIN_SEEDS):
    """The model the submission uses: trained on every external match, BAR-ATH never seen."""
    import external
    import features
    import noise
    import noise_stats
    stats = noise_stats.load()
    feature_sets, label_sets = [], []
    for game in external.GAMES:
        match, labels, restarts = external.metrica_match(game)
        for seed in seeds:
            feature_sets.append(features.match_features(noise.noisy_match(match, labels, restarts, seed, stats)[0]))
            label_sets.append(labels)
    return train(feature_sets, label_sets)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    import core
    import features

    fit, names = train_on_external()
    match = core.bar_ath_match()
    codes = core.nearest_team(match)
    probability = home_probability(fit, names, features.match_features(match, codes))
    for window in (WINDOW, 125):
        name = "saryarka_v7_smooth{}".format(window)
        path = core.write_submission(os.path.join(core.RESULTS_DIR, name + ".csv"),
                                     labels(match, codes, probability, window, SWITCH_COST))
        print("{:<22} -> {}".format(name, os.path.relpath(path, core.ROOT)))
