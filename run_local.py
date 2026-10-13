"""Run Day 10 stages from a local Python environment.

Examples:
    .venv\\Scripts\\python run_local.py prepare
    .venv\\Scripts\\python run_local.py e0
    .venv\\Scripts\\python run_local.py all
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")


def load_env_file(path):
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.split("#", 1)[0].strip()
        if key in ("KAGGLE_USERNAME", "KAGGLE_KEY", "HF_TOKEN") and value:
            os.environ.setdefault(key, value)


def prepare(source=None, archive=None):
    if (ROOT / "data/splits/meta.json").exists():
        print("Data already prepared; remove data/ only if a fresh split is required.")
        return
    cmd = [sys.executable, str(ROOT / "scripts/prepare_data.py")]
    if source:
        cmd += ["--source", source]
    elif archive:
        cmd += ["--zip", archive]
    else:
        load_env_file(ROOT / ".env")
        if not os.getenv("KAGGLE_USERNAME") or not os.getenv("KAGGLE_KEY"):
            raise RuntimeError("Kaggle credentials missing in .env; pass --source or --zip")
        cmd += ["--kaggle"]
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
    subprocess.run(cmd, cwd=ROOT, check=True)


def run(stage):
    load_env_file(ROOT / ".env")
    os.environ.setdefault("HF_HOME", str(ROOT / "work/hf_cache"))
    from student_lab import inspect_splits, run_e0, run_trained, save_illustrations

    rows = inspect_splits()
    if stage in ("e0", "all"):
        run_e0(rows["test"])
    if stage in ("e1", "e2", "e3", "all"):
        names = ("E1", "E2", "E3") if stage == "all" else (stage.upper(),)
        for name in names:
            run_trained(name, rows["train"], rows["test"])
    if stage in ("e4", "all"):
        evidence = ROOT / "report/evidence"
        base = max(("E1", "E2", "E3"), key=lambda name:
                   json.loads((evidence / f"{name.lower()}.json").read_text(encoding="utf-8"))["micro"]["f1"])
        for rank in (4, 64):
            name = f"E4_rank_{rank}"
            run_trained(name, rows["train"], rows["test"], setup=base, rank=rank)
    if stage in ("figures", "all"):
        save_illustrations(rows["test"], ROOT / "work/e3.jsonl", "e3")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "e0", "e1", "e2", "e3", "e4", "figures", "all"))
    parser.add_argument("--source", help="Extracted dataset directory")
    parser.add_argument("--zip", dest="archive", help="Dataset ZIP path")
    args = parser.parse_args()
    if args.stage in ("prepare", "all"):
        prepare(args.source, args.archive)
    if args.stage != "prepare":
        run(args.stage)


if __name__ == "__main__":
    main()
