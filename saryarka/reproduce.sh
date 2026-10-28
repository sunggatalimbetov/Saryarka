#!/usr/bin/env bash
# Team Saryarka: from the data bundle to Saryarka.csv in one command.
#
#   bash saryarka/reproduce.sh
#
# Before running (see saryarka/README.md): a Python 3.12 venv in .venv with saryarka/requirements.txt,
# the organisers' data bundle unpacked into data/, git and curl on PATH. Downloads ~1.9 GB of
# external matches (the slow part), needs ~10 GB of free disk; the rest takes ~5 min on 8 cores.
# Fails unless the file it writes is byte-identical to the one we submitted.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
WORKERS=${WORKERS:-8}
EXPECTED=344244bdff3aed6f7b541d3681e69c6da4ad71968a538d1195e114af0ee3e9e4

# One thread per numeric library: the same file on any machine, whatever its core count.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1

echo "1/4 external matches at pinned commits"
$PY saryarka/fetch_external.py

echo "2/4 BAR-ATH noise summary (unlabelled tracking + Smart Tagging), then signal caches for the 23"
echo "    external matches, $WORKERS at a time"
# Built once up front: every worker's noise reads data/BAR-ATH_noise_stats.json, and workers creating
# it together read each other's half-written file.
$PY saryarka/noise_stats.py
# model.py would build these one by one; this only does it in parallel. Kept out of saryarka/*.py
# because every file there is part of the cache fingerprint (tune_v8.code_fingerprint).
PREBUILD=data/_saryarka_prebuild.py
trap 'rm -f "$PREBUILD"' EXIT
cat > "$PREBUILD" <<'EOF'
import multiprocessing
import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "saryarka"))

if __name__ == "__main__":
    import external
    import tune_v9
    import tune_v11
    games = external.all_games()
    with multiprocessing.get_context("spawn").Pool(int(os.environ["WORKERS"])) as pool:
        for builder in (tune_v11.build, tune_v9.build_dead):
            # one seed at a time: two jobs must never parse the same SkillCorner match into its cache together
            for seed in tune_v11.TRAIN_SEEDS:
                list(pool.imap_unordered(builder, [(game, seed) for game in games]))
EOF
WORKERS=$WORKERS $PY "$PREBUILD"

echo "3/4 train on the external matches, label BAR-ATH"
$PY saryarka/model.py
cp evaluation/results/saryarka_v11.csv Saryarka.csv

echo "4/4 check"
$PY evaluation/validate_submission.py Saryarka.csv
GOT=$($PY -c "import hashlib; print(hashlib.sha256(open('Saryarka.csv', 'rb').read()).hexdigest())")
if [ "$GOT" != "$EXPECTED" ]; then
    echo "Saryarka.csv sha256 $GOT, expected $EXPECTED" >&2
    exit 1
fi
echo "Saryarka.csv is byte-identical to the submitted file ($EXPECTED)"
