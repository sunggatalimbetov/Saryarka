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
SKILLCORNER_MEDIA = "https://media.githubusercontent.com/media/SkillCorner/opendata/{1}/data/matches/{0}/{0}_tracking_extrapolated.jsonl"
# Pinned 27 Oct: the exact data every number in the repo was built from. Upstream may add matches
# or fix files later; the harness and the trained models must not change with it.
METRICA_COMMIT = "e706dd506b360d69d9d123d5b8026e7294b13996"
SKILLCORNER_COMMIT = "4340d274572876239c154c90bc507a9b3250a656"


def clone(url, folder, commit):
    """Shallow clone of exactly `commit`; refuses a folder already at another commit."""
    env = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1")
    if os.path.isdir(folder):
        print("already there:", os.path.relpath(folder, ROOT))
    else:
        os.makedirs(folder)
        for args in (["init", "-q"], ["remote", "add", "origin", url], ["fetch", "-q", "--depth", "1", "origin", commit],
                     ["checkout", "-q", "FETCH_HEAD"]):
            subprocess.run(["git", "-C", folder] + args, check=True, env=env)
    head = subprocess.run(["git", "-C", folder, "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()
    if head != commit:
        raise SystemExit("{} is at {}, expected {}".format(folder, head, commit))
    return head


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
        subprocess.run(["curl", "-sSL", "--retry", "3", "-o", target + ".part",
                        SKILLCORNER_MEDIA.format(game, SKILLCORNER_COMMIT)], check=True)
        os.replace(target + ".part", target)


if __name__ == "__main__":
    os.makedirs(EXTERNAL, exist_ok=True)
    print("metrica sample data at commit", clone(METRICA_URL, METRICA_DIR, METRICA_COMMIT))
    print("skillcorner open data at commit", clone(SKILLCORNER_URL, SKILLCORNER_DIR, SKILLCORNER_COMMIT))
    skillcorner_tracking()
