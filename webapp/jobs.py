"""In-process background jobs for on-demand bulk / Nasdaq screens.

A screen of 50 symbols can take a couple of minutes, so the request kicks off a
worker thread and returns a job id. The page then polls the job status and
reloads itself when the job finishes.
"""

import threading
import time
import uuid

import engine

_JOBS: dict[str, dict] = {}
_LOCK = threading.Lock()
_MAX_JOBS = 40


def _prune() -> None:
    if len(_JOBS) <= _MAX_JOBS:
        return
    done = sorted(
        ((jid, j) for jid, j in _JOBS.items() if j["state"] in ("done", "error")),
        key=lambda kv: kv[1]["finished_at"] or 0,
    )
    for jid, _ in done[: len(_JOBS) - _MAX_JOBS]:
        _JOBS.pop(jid, None)


def get(job_id: str) -> dict | None:
    return _JOBS.get(job_id)


def start_bulk(symbols: list[str], sessions: dict | None = None,
               label: str = "", meta: dict | None = None) -> str:
    job_id = uuid.uuid4().hex[:12]
    job = {
        "id": job_id,
        "state": "running",          # running | done | error
        "label": label,
        "total": len(symbols),
        "done": 0,
        "current": symbols[0] if symbols else "",
        "rows": None,
        "error": None,
        "started_at": time.time(),
        "finished_at": None,
        "meta": meta or {},
    }
    with _LOCK:
        _JOBS[job_id] = job
        _prune()

    def progress_cb(done, total, symbol):
        job["done"] = done
        job["current"] = symbol

    def worker():
        try:
            rows = engine.compute_bulk(
                symbols, progress_cb=progress_cb, use_cache=True,
                sessions=sessions or {},
            )
            job["rows"] = rows
            job["state"] = "done"
        except Exception as e:  # noqa: BLE001
            job["error"] = str(e)
            job["state"] = "error"
        finally:
            job["finished_at"] = time.time()

    threading.Thread(target=worker, daemon=True).start()
    return job_id


def public_status(job: dict) -> dict:
    """The subset safe to hand to the polling client."""
    return {
        "id": job["id"],
        "state": job["state"],
        "total": job["total"],
        "done": job["done"],
        "current": job["current"],
        "error": job["error"],
        "label": job["label"],
    }
