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

    .venv/bin/python saryarka/model.py                 # v9: trains, writes evaluation/results/saryarka_v9_*.csv
    .venv/bin/python saryarka/model.py --version v8    # without the learned dead ball
    .venv/bin/python saryarka/model.py --version v7    # the logistic model, Metrica only
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Params
STRIDE = 5            # use every 5th live frame for training: neighbours are near copies
C = 1.0               # inverse regularisation strength (logistic)
KIND = "logistic"     # model kind used by train() unless told otherwise
# Gradient boosting (v8): a few hundred small trees, each fixing the previous ones' mistakes.
BOOSTING = {"max_iter": 300, "learning_rate": 0.1, "max_leaf_nodes": 31, "min_samples_leaf": 200,
            "l2_regularization": 1.0, "early_stopping": False}
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


def training_rows(f, labels, names, stride=STRIDE):
    """(X, y) for one match: every stride-th live frame, seen as is and with the teams swapped."""
    frames = np.flatnonzero(np.isin(labels, ["H", "A"]))[::stride]
    home = (labels[frames] == "H").astype(int)
    return (np.vstack([matrix(f, names)[frames], matrix(mirror(f), names)[frames]]),
            np.concatenate([home, 1 - home]))


def fit(X, y, kind=KIND, sample_weight=None):
    """A fitted classifier of the given kind: 'logistic' (v7) or 'boosting' (v8).

    sample_weight lets each training source count the same in total, whatever its match count."""
    if kind == "boosting":
        from sklearn.ensemble import HistGradientBoostingClassifier
        model = HistGradientBoostingClassifier(**BOOSTING, random_state=0)  # handles NaN itself
    else:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                              LogisticRegression(C=C, max_iter=2000))
    if sample_weight is None:
        model.fit(X, y)
    elif kind == "boosting":
        model.fit(X, y, sample_weight=sample_weight)
    else:
        model.fit(X, y, logisticregression__sample_weight=sample_weight)
    return model


def train(feature_sets, label_sets, stride=STRIDE, kind=KIND):
    """Fit on [(features dict, labels array)] from external matches; returns (model, names)."""
    names = sorted(feature_sets[0])
    parts = [training_rows(f, labels, names, stride) for f, labels in zip(feature_sets, label_sets)]
    return fit(np.vstack([X for X, _ in parts]), np.concatenate([y for _, y in parts]), kind), names


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


# v8, picked by tune_v8.py (26 Oct): gradient boosting, Metrica and SkillCorner weighted equally,
# trained on all 23 external matches. Held-out spellF1 with the sources counted equally 0.344
# (unweighted boosting 0.332, logistic 0.319, v6 0.310); Metrica 0.386, SkillCorner 0.303.
# Switch cost 3 picked on every held-out match (costs 1-3 within 0.004).
V8_SWITCH_COST = 3.0


def train_v8():
    """v8's model: balanced boosting on every external match's cached training rows."""
    import external
    import tune_v8
    games = external.all_games()
    jobs = [(g, s) for g in games for s in tune_v8.TRAIN_SEEDS]
    for job in jobs:
        tune_v8.build(job)  # no-op when cached
    parts = [np.load(tune_v8.cache_path(g, s)) for g, s in jobs]
    sources = np.concatenate([np.full(len(p["y"]), g in external.GAMES) for (g, _), p in zip(jobs, parts)])
    X = np.vstack([p["X"] for p in parts]).astype(float)
    y = np.concatenate([p["y"] for p in parts])
    return fit(X, y, "boosting", tune_v8.source_weights(sources)), list(parts[0]["names"])


# v9, picked by tune_v9.py (27 Oct): v8 + the learned dead-ball probability (deadball.py), tagged
# spans backdated (stage A) and untagged stoppages added (stage B). Held-out spellF1 over the 23
# external matches 0.424 vs v8 0.314, wins 23/23 (Metrica 0.486, SkillCorner 0.414).
V9_DEAD = {"t_low": 0.3, "t_high": 0.6, "min_dead_s": 2.0}


def train_dead():
    """v9's dead-ball model: balanced boosting on every external match's cached dead-ball rows."""
    import external
    import tune_v9
    games = external.all_games()
    for job in [(g, s) for g in games for s in tune_v9.TRAIN_SEEDS]:
        tune_v9.build_dead(job)  # no-op when cached
    return tune_v9.train(games, tune_v9.dead_path)


if __name__ == "__main__":
    import argparse
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    import core
    import features

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", choices=("v7", "v8", "v9"), default="v9")
    version = parser.parse_args().version
    fit_, names = train_on_external() if version == "v7" else train_v8()
    cost = SWITCH_COST if version == "v7" else V8_SWITCH_COST
    match = core.bar_ath_match()
    codes = core.nearest_team(match)
    probability = home_probability(fit_, names, features.match_features(match, codes))
    dead = None
    if version == "v9":
        import deadball
        dead_fit, dead_names = train_dead()
        p_dead = deadball.probability(dead_fit, dead_names, deadball.build(match))
        dead = deadball.spans(match, p_dead, **V9_DEAD)
        np.save(os.path.join(core.ROOT, "data", "bar_ath_p_dead.npy"), p_dead)  # for summary checks only
    for window in (WINDOW, 125):
        name = "saryarka_{}_smooth{}".format(version, window)
        out = (labels(match, codes, probability, window, cost) if dead is None else
               deadball.labels(match, codes, probability, dead, window, cost))
        path = core.write_submission(os.path.join(core.RESULTS_DIR, name + ".csv"), out)
        print("{:<22} -> {}".format(name, os.path.relpath(path, core.ROOT)))
