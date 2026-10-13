"""Run the remaining GPU stages in a persistent local process.

Start with the project's Python environment. Progress and failures are written
to work/logs/remaining.log and work/remaining_status.json.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WORK = ROOT / "work"
LOG = WORK / "logs" / "remaining.log"
STATUS = WORK / "remaining_status.json"


def record(stage, state, **extra):
    payload = {"stage": stage, "state": state,
               "updated_utc": datetime.now(timezone.utc).isoformat(), **extra}
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(STATUS)


def completed(stage):
    evidence = ROOT / "report" / "evidence"
    if stage == "e4":
        return all((evidence / f"e4_rank_{rank}.json").is_file() for rank in (4, 64))
    return (evidence / f"{stage}.json").is_file()


def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    for stage in ("e2", "e3", "e4"):
        if completed(stage):
            record(stage, "already_complete")
            continue
        record(stage, "running")
        with LOG.open("a", encoding="utf-8", buffering=1) as log:
            log.write(f"\n=== {stage.upper()} started {datetime.now(timezone.utc).isoformat()} ===\n")
            process = subprocess.run([sys.executable, "-u", "run_local.py", stage],
                                     cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                     env={**os.environ, "PYTHONUNBUFFERED": "1"}, check=False)
            log.write(f"=== {stage.upper()} exited {process.returncode} ===\n")
        if process.returncode or not completed(stage):
            record(stage, "failed", exit_code=process.returncode, log=str(LOG))
            return 1
        record(stage, "complete")
    with LOG.open("a", encoding="utf-8", buffering=1) as log:
        check = subprocess.run([sys.executable, "scripts/check_submission.py"],
                               cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
        log.write(f"=== checker exited {check.returncode} ===\n")
    record("all", "complete" if check.returncode == 0 else "checker_failed",
           exit_code=check.returncode, log=str(LOG))
    return check.returncode


if __name__ == "__main__":
    raise SystemExit(main())
