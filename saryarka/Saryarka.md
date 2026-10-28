# Team Saryarka: technical document

More than a Hack 2027, Round 1 (ball possession on BAR-ATH). Team: Sunggat Alimbetov, Yerasyl
Sailaubai, Aliyar Askergali. Code: https://github.com/sunggatalimbetov/Saryarka (`saryarka/`).
Submission: `Saryarka.csv`, sha256 `344244bd...e9e4`, built by version v11 of our pipeline.

## 1. Approach

We treat the match as a sequence of states and decode it as a whole, not frame by frame. Three
parts:

1. **Possession probability.** A gradient-boosted classifier (scikit-learn
   `HistGradientBoostingClassifier`) gives, for every live frame, the probability that home has the
   ball. Its inputs are 39 signals computed from tracking alone (nearest-player distance gap,
   players within 2 m and 5 m of the ball, which team is pressing and pushing, team width and
   length gaps, ball speed and direction relative to each team's attacking goal, a last-touch
   detector, the organisers' ±2 s vote share) plus the team of the current Smart Tagging phase.
   Every training frame is also shown with the teams swapped and the answer flipped, and every
   prediction averages both views, so the model cannot learn "home usually has it".
2. **Dead ball.** A second gradient-boosted classifier gives P(ball dead) from team-free tracking
   signals: where the ball was last seen relative to the lines, how long it has been missing,
   whether it is still, and whether players have slowed to a walk or gathered. It is used to move
   the start of each Smart Tagging restart back to the moment the ball went out, and to add
   stoppages Smart Tagging does not tag (fouls, offsides).
3. **Decoder.** A two-state Viterbi decoder (home / away) over the evidence 2p − 1, where every
   change of team costs 8. Dead spans cut the match into live pieces and are written as `D`.
   This is what turns a noisy per-frame probability into whole possessions with sharp boundaries.

**Why this and not something else.** The ranking metric (spellF1) matches whole possessions by
overlap, so flicker and merged possessions cost far more than a wrong frame here and there. A
decoder with one switch cost controls the possession count directly. We learned the per-frame
probability instead of hand-weighting signals because the hand-weighted versions plateaued
(section 6), and we kept the model small (boosted trees on hand-built signals) because we have 23
labelled external matches, not thousands.

**How we chose between versions.** The public 5 minutes have 17 possessions, far too few to rank
versions, and tuning on them is not allowed. So every decision was made on our own harness
(section 3): 23 external matches, made to look like BAR-ATH's tracking, scored leave-one-match-out.
Before each experiment we fixed the rule for "better": the all-23 mean goes up, it wins on at least
16 of 23 matches, and neither data source drops by more than 0.01. The public window was only used
to check that files are aligned and well-formed.

## 2. Assets used

| asset | used? | how |
|---|---|---|
| Tactical ATD tracking (`*_tactical_atd.txt`) | yes | all player and ball signals, in pitch metres |
| Tactical ATD metadata (`*_tactical_atd.xml`) | yes | which track slots belong to which team, goalkeeper slots |
| Smart Tagging (`*_tactical_smart_tagging.xml`) | yes | restart tags (start of a dead span, and team for corners, free kicks, goal kicks), GOALS tags (dead from goal to kickoff), phase tags (team as a model input), the 2nd-half kickoff check |
| Published half boundaries (FAQ, DATA) | yes | graded halves; frames 110,808 to 113,741 before the 2nd-half kickoff are `D` (organiser metadata, checked against the Smart Tagging kickoff tag) |
| Tactical and broadcast video | no | a ball detector would need external footage to train on; out of our time budget |
| Broadcast ATD, homographies | no | the tactical tracking is already in pitch coordinates and covers the whole match |
| ATD half markers | tried, dropped | the first half spans the whole video and the second is empty, so they say nothing |

## 3. Training data

No label of any kind comes from BAR-ATH. Every labelled frame is from another match:

| dataset | source, version | labels: who and how | amount |
|---|---|---|---|
| Metrica Sports sample data | github.com/metrica-sports/sample-data, commit `e706dd5` | Metrica's own event logs; our code turns them into frame labels by fixed rules (below), nobody labelled anything by hand | 3 anonymised full matches, 25 fps, ~350 to 400 possessions each |
| SkillCorner open data | github.com/SkillCorner/opendata, commit `4340d27` | SkillCorner's own per-frame `possession.group` field as published; our code fills the frames where it names nobody | 20 A-League 2024/25 matches, broadcast tracking at 10 fps (upsampled to 25) |
| DFL open data | 7 Bundesliga matches | tried as extra dead-ball training data, dropped (no clear change, 11/23 wins); its possession flips on every duel, so it was never used for team labels | 0 in the submission |

**Rules that turn events into labels (Metrica).** `D` from a ball-killing event (ball out, shot out
or in, goal, foul, offside, and each period before its kickoff) to the next on-ball event. Otherwise
the team of the latest on-ball event (pass, carry, set piece, recovery, shot, ball lost), held until
the other team's next on-ball event, so a loose ball stays with the last team. **SkillCorner:**
where `possession.group` names nobody, `D` before a kickoff, inside a logged stoppage, or for an
unnamed stretch of 3 s or more; otherwise the last team keeps it. Replays with no tracking are not
scored. The 3-second-rule `D` frames are left out when training the dead-ball model, because no
logged stoppage backs them.

These are not the organisers' conventions, which come from corrected tracking. The harness only
compares our versions against each other; it does not predict our real score.

**Making external matches look like BAR-ATH.** Clean external tracking gives the answer away (in
Metrica's sample the ball disappears almost only when it is dead). Before training or scoring, each
external match gets noise fitted to unlabelled summary numbers of BAR-ATH's own tracking: the live
ball missing about 40% of the time in long stretches, mostly when play is crowded and slow; the ball
still visible on about 24% of dead frames; a centred 0.3 s blur (0.2 s for SkillCorner) and a slow
0.75 m drift, fitted to 13 BAR-ATH ball-position summaries; Smart Tagging tags faked from the true
events with BAR-ATH's per-type miss rates, restart leads drawn from BAR-ATH's spread, and phase
starts delayed 6 to 12 s so their timing cannot leak the answer. Each match is used with two noise
seeds for training and two other seeds for testing.

What we took from BAR-ATH itself, all unlabelled: those tracking summaries, the Smart Tagging tags as
inputs, and a sanity guard on our own output fixed before each run (dead share between 25% and 45%,
at most 60 stoppages that no tag backs). v11 gives 42.6% dead and 60 such stoppages.

## 4. Possession rules we implemented

- **Loose balls, deflections, challenges.** Possession stays with the last team until the evidence
  for the other team outweighs the switch cost. A contested ball or a one-touch deflection does not
  flip it; a sustained change does. The training labels follow the same idea (section 3).
- **Ball missing.** Not `D` and not `X`. The decoder keeps going on the player signals and the last
  seen ball spot. We never write `X` on a graded frame.
- **Restarts.** A dead span ends when, after a restart tag, the ball is inside the pitch (1 m margin)
  and moving at 3 m/s or more within 12 s of the tag's start. It starts when the dead-ball
  probability says the ball went out: we walk the start back from the tag while P(dead) > 0.3, up to
  25 s. For corners, free kicks and goal kicks, the tag's team tilts the evidence (+0.5) for 6 s
  after the restart. Throw-ins carry no team in Smart Tagging, so they get no tilt.
- **Untagged stoppages** (fouls, offsides, injuries): P(dead) > 0.8 for at least 3 s, widened to where
  it drops below 0.3.
- **Goals.** `D` from the start of each GOALS tag until the ball is back in play at the kickoff.
- **Half boundaries.** The graded gap before the 2nd-half kickoff (frames 110,808 to 113,741) is
  `D`. Ungraded frames get whatever the pipeline outputs; they cost nothing.

## 5. Results

Held-out spellF1 on our harness (noisy external matches, leave-one-match-out). The harness changed
twice (a more realistic ball on 26 Oct, then 20 SkillCorner matches), so numbers are only comparable
within a column group.

| version | what it added | Metrica 3, ball-fixed harness | all 23 matches |
|---|---|---|---|
| organisers' baseline (±2 s vote) | | 0.220 | |
| v6 | missing ball is `X` not `D`, kickoff gap and set pieces as `D`, Viterbi decoder, goal spans, last-touch evidence | 0.355 | |
| v7 | logistic regression replaces the vote share | 0.407 | |
| v8 | gradient boosting, trained on all 23 matches | | 0.305 |
| v9 | learned dead ball | | 0.409 (23/23 wins) |
| v10 | no forced `X`, restart-tag team tilt | | 0.451 |
| **v11** | short possessions (under 5 s) count 3x in training, switch cost 8 | | **0.459** (20/23 vs v10) |

On the public 5 minutes, v11 scores spellF1 0.581 and macroF1 0.707 (baseline 0.326 and 0.621). We
report it only as a pipeline check: with 17 possessions, differences under about 0.04 are noise.

## 6. What we tried and dropped

- Per-frame nearest team with no decoder: merged short possessions.
- Weighting Metrica and SkillCorner equally vs plainly: first reported as a gain, then withdrawn,
  because it won only 10 of 23 matches under a rule fixed in advance.
- A three-state (home / away / dead) decoder: no clear change (14/23).
- DFL matches as dead-ball training data: no clear change (11/23).
- Eight extra fast signals for short possessions (next touch, weak touches, before/after gaps,
  closest approach, control): added nothing over v10's signals.
- Reshaping the decoder evidence (power and logit transforms): no clear change (15/23).
- An out-of-play rule ("the restart team is not the last toucher"): the external labels do not follow
  it (only ~39% flip around restarts).

## 7. Limitations

- **Short possessions are mostly invisible.** Even in clean external tracking, the nearest player is
  on the team with the ball in only 60 to 64% of the frames of a possession under 5 s (77 to 82% for
  longer ones). Our harness has about 159 such possessions per match and we miss most of them. This
  is the largest remaining loss.
- **We report too few possessions.** On the harness v9 made 206 possessions per match against 338
  true; v11 raises the count on BAR-ATH (114 and 135 per half vs 103 and 139 for v10) but is still
  well under the truth on the harness.
- **The harness is a proxy.** Its labels follow Metrica's events and SkillCorner's own definition,
  not the organisers' corrected-tracking flags. Its window-ranking check fails on SkillCorner (the
  plain baseline prefers ±5 s there, the real match prefers ±2 s); we accepted this and choose
  windows from Metrica.
- **Player tracking noise is not simulated.** BAR-ATH has ghost tracks and missing players (9 or
  fewer of a team on ~10% of live frames); the harness adds noise to the ball only.
- **Throw-ins** carry no team in Smart Tagging, so their restarts get no team prior.

With more time: simulate player-track noise on the harness, a ball detector trained on external
footage to fill the 41% of frames without a ball, and duration-aware decoding (HSMM) aimed at short
possessions.

## 8. Reproduction

```bash
git clone https://github.com/sunggatalimbetov/Saryarka.git && cd Saryarka
# unpack the organisers' data bundle into data/ as their README says
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r saryarka/requirements.txt
bash saryarka/reproduce.sh
```

`reproduce.sh` fetches the external matches at pinned commits, builds their signal caches, trains
both models (fixed seeds, one thread per numeric library), labels BAR-ATH, runs the organisers'
`validate_submission.py` and fails unless `Saryarka.csv` is byte-identical to the submitted file.
After the ~1.9 GB download, about 5 minutes on 8 cores; ~10 GB of disk. Tested on macOS arm64, Python 3.12.13, from a fresh clone.
