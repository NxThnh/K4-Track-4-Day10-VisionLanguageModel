"""Finish the measured report and push main after E2-E4 pass the checker.

This watcher is intentionally separate from run_remaining.py, which may already
be running in another Python process. It never publishes partial evidence.
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WORK = ROOT / "work"
RUN_STATUS = WORK / "remaining_status.json"
PUBLISH_STATUS = WORK / "publish_status.json"
PUBLISH_LOG = WORK / "logs" / "publish.log"
EVIDENCE = ROOT / "report" / "evidence"
REQUIRED = ("e0", "e1", "e2", "e3", "e4_rank_4", "e4_rank_64")


def record(state, **extra):
    payload = {"state": state,
               "updated_utc": datetime.now(timezone.utc).isoformat(), **extra}
    temporary = PUBLISH_STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(PUBLISH_STATUS)


def log(message):
    PUBLISH_LOG.parent.mkdir(parents=True, exist_ok=True)
    with PUBLISH_LOG.open("a", encoding="utf-8") as f:
        f.write(f"{datetime.now(timezone.utc).isoformat()} {message}\n")


def git(*args, check=True):
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    result = subprocess.run(["git", *args], cwd=ROOT, env=env,
                            capture_output=True, text=True, check=False)
    log(f"git {' '.join(args)}: exit {result.returncode}; {result.stdout.strip()} {result.stderr.strip()}")
    if check and result.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result


def load_evidence():
    evidence = {name: json.loads((EVIDENCE / f"{name}.json").read_text(encoding="utf-8"))
                for name in REQUIRED}
    versions = {item["dataset_version"] for item in evidence.values()}
    if len(versions) != 1 or any(item.get("split") != "test" or item.get("missing_predictions") != 0
                                 for item in evidence.values()):
        raise RuntimeError("Evidence has inconsistent dataset, split, or missing predictions")
    return evidence


def result_row(name, setup, item):
    meta = item.get("meta", {})
    params = f"{int(meta['trainable_params']):,}" if "trainable_params" in meta else "0"
    vram = f"{meta['peak_vram_gb']:.3f}" if "peak_vram_gb" in meta else "Not recorded"
    minutes = f"{meta['train_minutes']:.3f}" if "train_minutes" in meta else "Not recorded"
    per_class = [f"{item['per_class'][label]['f1']:.4f}" for label in ("pothole", "crack", "manhole")]
    cells = [name, setup, params, vram, minutes, f"{item['micro']['f1']:.4f}",
             f"{item['f1_ci95'][0]:.4f}–{item['f1_ci95'][1]:.4f}",
             f"{item['fp_per_image']:.3f}", *per_class]
    return "| " + " | ".join(cells) + " |"


def replace_section(text, heading, body):
    start = text.index(heading)
    content_start = text.index("\n", start) + 1
    next_heading = text.find("\n### ", content_start)
    if next_heading < 0:
        next_heading = text.find("\n## ", content_start)
    if next_heading < 0:
        next_heading = len(text)
    return text[:content_start] + body.rstrip() + "\n" + text[next_heading:]


def update_report(evidence):
    report_path = ROOT / "REPORT.md"
    text = report_path.read_text(encoding="utf-8")
    old_status = next(line for line in text.splitlines() if line.startswith("**Status:**"))
    new_status = ("**Status:** E0–E4 completed with scorer-generated evidence on the fixed test split. "
                  "E0–E1 were run on Colab; E2–E4 were run locally. All numbers below come from "
                  "`report/evidence/*.json`.")
    text = text.replace(old_status, new_status, 1)
    setups = {"E2": "+ projector", "E3": "+ LoRA ViT",
              "E4 rank 4": "Best E1–E3 setup, r=4",
              "E4 rank 64": "Same setup, r=64"}
    keys = {"E2": "e2", "E3": "e3", "E4 rank 4": "e4_rank_4", "E4 rank 64": "e4_rank_64"}
    lines = text.splitlines()
    for index, line in enumerate(lines):
        for label, setup in setups.items():
            if line.startswith(f"| {label} |"):
                lines[index] = result_row(label, setup, evidence[keys[label]])
                break
    text = "\n".join(lines) + "\n"

    e1, e2, e3 = (evidence[name] for name in ("e1", "e2", "e3"))
    best_name = max(("e1", "e2", "e3"), key=lambda name: evidence[name]["micro"]["f1"])
    best = evidence[best_name]
    q7 = (f"E1/E2/E3 F1: {e1['micro']['f1']:.4f} / {e2['micro']['f1']:.4f} / "
          f"{e3['micro']['f1']:.4f}. Their CI95 intervals are "
          f"{e1['f1_ci95'][0]:.4f}–{e1['f1_ci95'][1]:.4f}, "
          f"{e2['f1_ci95'][0]:.4f}–{e2['f1_ci95'][1]:.4f}, and "
          f"{e3['f1_ci95'][0]:.4f}–{e3['f1_ci95'][1]:.4f}. "
          f"Peak VRAM (GB): {e1['meta']['peak_vram_gb']:.3f} / "
          f"{e2['meta']['peak_vram_gb']:.3f} / {e3['meta']['peak_vram_gb']:.3f}; "
          f"training time (minutes): {e1['meta']['train_minutes']:.3f} / "
          f"{e2['meta']['train_minutes']:.3f} / {e3['meta']['train_minutes']:.3f}. "
          f"{best_name.upper()} has the highest observed F1. Marginal CI overlap alone does "
          "not establish a paired difference; no paired bootstrap was computed.")
    text = replace_section(text, "### Câu 7 —", q7)

    rank4, rank64 = evidence["e4_rank_4"], evidence["e4_rank_64"]
    delta = rank64["micro"]["f1"] - rank4["micro"]["f1"]
    q10 = (f"E4 changed only LoRA rank on the {best_name.upper()} setup: r=4 versus r=64, "
           f"with alpha=2r. F1 was {rank4['micro']['f1']:.4f} "
           f"(CI95 {rank4['f1_ci95'][0]:.4f}–{rank4['f1_ci95'][1]:.4f}) versus "
           f"{rank64['micro']['f1']:.4f} "
           f"(CI95 {rank64['f1_ci95'][0]:.4f}–{rank64['f1_ci95'][1]:.4f}); "
           f"the observed r=64 minus r=4 difference was {delta:+.4f}. "
           "With 311 test images, these marginal intervals describe uncertainty for each "
           "run. A paired interval for their difference was not computed, so this alone "
           "does not prove that one rank is better.")
    text = replace_section(text, "### Câu 10 —", q10)
    text = replace_section(text, "## Nhìn lại", "\nE0–E4 have measured test evidence. The six annotated images were generated from saved E1 predictions. The rank ablation used the highest-F1 E1–E3 setup selected on this test split; that selection can bias the apparent best result and should be repeated on a separate holdout.")
    report_path.write_text(text, encoding="utf-8")


def publish():
    evidence = load_evidence()
    update_report(evidence)
    check = subprocess.run([sys.executable, "scripts/check_submission.py"],
                           cwd=ROOT, capture_output=True, text=True, check=False)
    log(f"submission checker: exit {check.returncode}; {check.stdout.strip()} {check.stderr.strip()}")
    if check.returncode:
        raise RuntimeError("Submission checker failed; see publish.log")
    git("add", "REPORT.md", "report/evidence/e2.json", "report/evidence/e3.json",
        "report/evidence/e4_rank_4.json", "report/evidence/e4_rank_64.json")
    status = git("status", "--porcelain").stdout
    if not status.strip():
        log("No new files to commit")
    else:
        git("commit", "-m", "results: add measured E2-E4 evidence and finish report")
    git("fetch", "origin", "main")
    git("merge-base", "--is-ancestor", "origin/main", "HEAD")
    git("push", "origin", "main")
    record("pushed", commit=git("rev-parse", "HEAD").stdout.strip())


def main():
    WORK.mkdir(exist_ok=True)
    record("waiting")
    while True:
        if RUN_STATUS.exists():
            run = json.loads(RUN_STATUS.read_text(encoding="utf-8"))
            if run.get("state") in ("failed", "checker_failed"):
                record("run_failed", run=run)
                return 1
            if run.get("stage") == "all" and run.get("state") == "complete":
                break
        time.sleep(60)
    try:
        record("publishing")
        publish()
        return 0
    except Exception as exc:
        log(f"publish failed: {exc}")
        record("publish_failed", error=str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
