"""Generic multi-GPU job scheduler.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence


@dataclass
class Job:
    """A single unit of parallel work."""

    job_id: str
    command: Sequence[str]
    payload: dict = field(default_factory=dict)
    # Optional callable that returns the expected output path for this
    # job; used when ``skip_existing=True``.
    output_path: Optional[Callable[[dict], Path]] = None


@dataclass
class _JobStatus:
    job_id: str
    status: str  # pending | running | done | skipped | fail(<rc>)
    gpu: str = ""
    pid: str = ""
    retcode: str = ""

    def as_row(self) -> dict:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "gpu": self.gpu,
            "pid": self.pid,
            "retcode": self.retcode,
        }


def _load_status(path: Path) -> dict[str, _JobStatus]:
    if not path.exists():
        return {}
    with path.open() as f:
        reader = csv.DictReader(f)
        return {
            row["job_id"]: _JobStatus(
                job_id=row["job_id"], status=row["status"],
                gpu=row.get("gpu", ""), pid=row.get("pid", ""),
                retcode=row.get("retcode", ""),
            )
            for row in reader
        }


def _write_status(path: Path, statuses: dict[str, _JobStatus]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["job_id", "status", "gpu", "pid", "retcode"])
        writer.writeheader()
        for st in statuses.values():
            writer.writerow(st.as_row())


def run_parallel(
    jobs: list[Job],
    gpus: list[int],
    *,
    poll: float = 5.0,
    log_dir: Path | str = Path("logs"),
    status_path: Optional[Path | str] = None,
    skip_existing: bool = False,
    env_extras: Optional[dict] = None,
) -> int:
    """Run ``jobs`` across ``gpus``. Returns 0 if every job succeeded."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = Path(status_path) if status_path else None

    statuses: dict[str, _JobStatus] = _load_status(status_path) if status_path else {}

    for job in jobs:
        if job.job_id in statuses and statuses[job.job_id].status in ("done", "skipped"):
            continue
        if skip_existing and job.output_path is not None and job.output_path(job.payload).exists():
            statuses[job.job_id] = _JobStatus(job_id=job.job_id, status="skipped")
        else:
            statuses[job.job_id] = _JobStatus(job_id=job.job_id, status="pending")

    if status_path:
        _write_status(status_path, statuses)

    pending = [j for j in jobs if statuses[j.job_id].status == "pending"]
    print(f"Total jobs: {len(jobs)} | pending: {len(pending)} | "
          f"already done/skipped: {len(jobs) - len(pending)}")
    print(f"Using GPUs: {gpus} | logs: {log_dir}")

    gpu_free = set(gpus)
    running: dict[str, tuple[subprocess.Popen, int, object, Job]] = {}
    done = failed = 0

    def _launch(job: Job, gpu: int) -> None:
        nonlocal running
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["VLLM_LOGGING_LEVEL"] = env.get("VLLM_LOGGING_LEVEL", "ERROR")
        env["TRANSFORMERS_VERBOSITY"] = env.get("TRANSFORMERS_VERBOSITY", "error")
        if env_extras:
            env.update(env_extras)
        log_file = open(log_dir / f"{job.job_id}.log", "w")
        proc = subprocess.Popen(
            list(job.command), env=env, stdout=log_file, stderr=log_file,
        )
        statuses[job.job_id] = _JobStatus(
            job_id=job.job_id, status="running", gpu=str(gpu), pid=str(proc.pid),
        )
        running[job.job_id] = (proc, gpu, log_file, job)
        print(f"[GPU {gpu}] launching {job.job_id}")

    while pending or running:
        while gpu_free and pending:
            gpu = gpu_free.pop()
            _launch(pending.pop(0), gpu)
            if status_path:
                _write_status(status_path, statuses)

        finished = []
        for job_id, (proc, gpu, log_file, job) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            log_file.close()
            statuses[job_id].retcode = str(rc)
            if rc == 0:
                statuses[job_id].status = "done"
                done += 1
                print(f"[GPU {gpu}] done {job_id}")
            else:
                statuses[job_id].status = f"fail({rc})"
                failed += 1
                print(f"[GPU {gpu}] failed {job_id} (rc={rc}) -> {log_dir}/{job_id}.log")
            finished.append(job_id)
            gpu_free.add(gpu)

        for job_id in finished:
            running.pop(job_id, None)

        if finished and status_path:
            _write_status(status_path, statuses)

        if pending or running:
            time.sleep(poll)

    skipped = sum(1 for s in statuses.values() if s.status == "skipped")
    print(f"\nAll jobs complete. done={done} skipped={skipped} failed={failed}")
    return 1 if failed else 0
