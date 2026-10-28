# Team Saryarka

Our Round 1 submission for More than a Hack 2027: `Saryarka.csv`, one possession label per frame of
BAR-ATH. The approach, data and limitations are in the technical document (`Saryarka.pdf`, submitted
with the form). This page is only how to rebuild the file.

## Reproduce

Tested on macOS arm64 with Python 3.12.13. Needs git, curl, ~10 GB of free disk and about an hour
on 8 cores.

```bash
git clone https://github.com/sunggatalimbetov/Saryarka.git && cd Saryarka

# 1. the organisers' data bundle, unpacked as their README says (data/tactical/ is what is read)
#    data/tactical/BAR-ATH_tactical_atd.txt, ..._atd.xml, ..._smart_tagging.xml, ...

# 2. pinned environment
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r saryarka/requirements.txt

# 3. everything else
bash saryarka/reproduce.sh
```

`reproduce.sh` downloads the external training matches at pinned commits (Metrica Sports sample
data, SkillCorner open data), builds their signal caches, trains on them, labels BAR-ATH and writes
`Saryarka.csv`. It then runs the organisers' `validate_submission.py` and fails unless the file's
sha256 is `344244bdff3aed6f7b541d3681e69c6da4ad71968a538d1195e114af0ee3e9e4`, the file we submitted.

Set `WORKERS=4` before the command on a smaller machine; the output does not depend on it.

Nothing is trained on BAR-ATH. Every match file, external download and cache stays in `data/`,
which git ignores.

## Files

| file | role |
|---|---|
| `model.py` | entry point: trains the possession and dead-ball models, writes `evaluation/results/saryarka_v11.csv` |
| `core.py`, `tracking.py`, `features.py` | BAR-ATH loading and the per-frame signals |
| `deadball.py`, `set_pieces.py`, `kickoff_d.py`, `restart_prior.py`, `phases.py`, `touches.py` | dead ball, restarts, Smart Tagging inputs, touches |
| `viterbi.py` | the decoder that turns per-frame probabilities into possessions |
| `external.py`, `skillcorner.py`, `fetch_external.py` | external matches and their labels |
| `noise.py`, `harness.py` | makes external matches look like BAR-ATH's tracking, and scores a version on them |
| `tune_v*.py`, `study_*.py` | the experiments behind each version, kept as the record |
