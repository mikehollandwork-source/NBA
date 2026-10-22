"""Download bulk nba.com-derived datasets from github.com/shufinskiy/nba_data.

stats.nba.com blocks most cloud IPs, so we use this public mirror of nba.com
play-by-play (cdn + v3), defender matchups and shot charts. Only the files we
need are fetched via a sparse, blob-less git clone.
"""
import subprocess
import tarfile
from pathlib import Path

from nba.config import RAW

REPO_URL = "https://github.com/shufinskiy/nba_data"
DATASETS = ["nbastatsv3", "cdnnba", "matchups", "shotdetail"]


def _run(cmd, cwd=None):
    import os
    subprocess.run(cmd, cwd=cwd, check=True, env={**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"})


def fetch(season: int, datasets=DATASETS, playoffs: bool = False) -> dict[str, Path]:
    repo = RAW / "nba_data_repo"
    names = [f"{d}_{'po_' if playoffs else ''}{season}" for d in datasets]
    if not (repo / ".git").exists():
        RAW.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", REPO_URL, str(repo)])
    listed = subprocess.run(["git", "sparse-checkout", "list"], cwd=repo, capture_output=True, text=True).stdout.split()
    pats = sorted(set(p for p in listed if p.startswith("/datasets/")) | {f"/datasets/{n}.tar.xz" for n in names})
    _run(["git", "sparse-checkout", "set", "--no-cone"] + pats, cwd=repo)
    out = {}
    for n in names:
        dest = RAW / "bulk" / n
        csv = dest / f"{n}.csv"
        if not csv.exists():
            dest.mkdir(parents=True, exist_ok=True)
            with tarfile.open(repo / "datasets" / f"{n}.tar.xz") as tf:
                tf.extractall(dest, filter="data")
        out[n] = csv
    return out


if __name__ == "__main__":
    import sys
    print(fetch(int(sys.argv[1]) if len(sys.argv) > 1 else 2025))
