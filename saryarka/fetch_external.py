"""Download the external matches the harness scores on, into data/external/ (git ignores data/).

Metrica Sports sample data: three anonymised full matches, 25 fps, github.com/metrica-sports/sample-data.
Credit: Metrica Sports. The repository ships no licence; it asks that the source be acknowledged.

SkillCorner open data: 20 A-League 2024/25 matches of broadcast tracking, 10 fps,
github.com/SkillCorner/opendata (SkillCorner x PySport; please credit SkillCorner). Its tracking
files sit in git LFS, so they are fetched from GitHub's media host instead (~90 MB each).

    .venv/bin/python saryarka/fetch_external.py
"""

import json
import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTERNAL = os.path.join(ROOT, "data", "external")
METRICA_URL = "https://github.com/metrica-sports/sample-data.git"
METRICA_DIR = os.path.join(EXTERNAL, "metrica-sample-data")
SKILLCORNER_URL = "https://github.com/SkillCorner/opendata.git"
SKILLCORNER_DIR = os.path.join(EXTERNAL, "skillcorner-opendata")
SKILLCORNER_MEDIA = "https://media.githubusercontent.com/media/SkillCorner/opendata/master/data/matches/{0}/{0}_tracking_extrapolated.jsonl"


def clone(url, folder):
    if os.path.isdir(folder):
        print("already there:", os.path.relpath(folder, ROOT))
    else:
        subprocess.run(["git", "clone", "--depth", "1", url, folder], check=True,
                       env=dict(os.environ, GIT_LFS_SKIP_SMUDGE="1"))
    return subprocess.run(["git", "-C", folder, "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()


def skillcorner_tracking():
    """Replace each LFS pointer with the real tracking file."""
    with open(os.path.join(SKILLCORNER_DIR, "data", "matches.json")) as handle:
        ids = sorted(m["id"] for m in json.load(handle))
    for game in ids:
        target = os.path.join(SKILLCORNER_DIR, "data", "matches", str(game), "{}_tracking_extrapolated.jsonl".format(game))
        with open(target, "rb") as handle:
            if not handle.read(200).startswith(b"version https://git-lfs"):
                continue
        print("fetching tracking", game, flush=True)
        subprocess.run(["curl", "-sSL", "--retry", "3", "-o", target + ".part", SKILLCORNER_MEDIA.format(game)], check=True)
        os.replace(target + ".part", target)


if __name__ == "__main__":
    os.makedirs(EXTERNAL, exist_ok=True)
    print("metrica sample data at commit", clone(METRICA_URL, METRICA_DIR))
    print("skillcorner open data at commit", clone(SKILLCORNER_URL, SKILLCORNER_DIR))
    skillcorner_tracking()
