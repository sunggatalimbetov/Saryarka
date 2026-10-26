"""Download the external matches the harness scores on, into data/external/ (git ignores data/).

Metrica Sports sample data: three anonymised full matches, 25 fps, github.com/metrica-sports/sample-data.
Credit: Metrica Sports. The repository ships no licence; it asks that the source be acknowledged.

    .venv/bin/python saryarka/fetch_external.py
"""

import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTERNAL = os.path.join(ROOT, "data", "external")
METRICA_URL = "https://github.com/metrica-sports/sample-data.git"
METRICA_DIR = os.path.join(EXTERNAL, "metrica-sample-data")

if __name__ == "__main__":
    os.makedirs(EXTERNAL, exist_ok=True)
    if os.path.isdir(METRICA_DIR):
        print("already there:", os.path.relpath(METRICA_DIR, ROOT))
    else:
        subprocess.run(["git", "clone", "--depth", "1", METRICA_URL, METRICA_DIR], check=True)
    commit = subprocess.run(["git", "-C", METRICA_DIR, "rev-parse", "HEAD"], check=True,
                            capture_output=True, text=True).stdout.strip()
    print("metrica sample data at commit", commit)
