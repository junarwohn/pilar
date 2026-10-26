"""Shared logging and durable background-job records for CLI and Web workers."""
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from datetime import datetime, timedelta, timezone
from functools import wraps
import json
import logging
import os
from pathlib import Path
import sys
import threading
import time
from uuid import uuid4


request_id = ContextVar("pilar_request_id", default="-")
current_job = ContextVar("pilar_job", default=None)
logger = logging.getLogger(__name__)


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def error_summary(exc):
    return (str(exc).split("Stacktrace:", 1)[0].strip() or type(exc).__name__)[:700]


class JsonFormatter(logging.Formatter):
    def format(self, record):
        job = current_job.get() or {}
        data = {
            "time": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname, "logger": record.name,
            "message": record.getMessage(), "pid": record.process,
            "request_id": request_id.get(), "run_id": job.get("run_id", "-"),
            "stage": job.get("stage", "-"),
        }
        for key in ("event", "stage", "status", "trigger", "duration_seconds", "image_count",
                    "title", "method", "path", "status_code", "run_id", "request_id"):
            if hasattr(record, key):
                data[key] = getattr(record, key)
        if record.exc_info:
            data["exception"] = self.formatException(record.exc_info)
        return json.dumps(data, ensure_ascii=False, default=str)


def configure_logging():
    level = getattr(logging, os.getenv("PILAR_LOG_LEVEL", "INFO").upper(), logging.INFO)
    root = logging.getLogger()
    if not any(getattr(h, "_pilar_handler", False) for h in root.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler._pilar_handler = True
        handler.setFormatter(JsonFormatter())
        root.handlers.clear()
        root.addHandler(handler)
    root.setLevel(level)
    logging.getLogger("ppocr").setLevel(logging.DEBUG if os.getenv("PILAR_OCR_DEBUG") == "1" else logging.WARNING)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(line_buffering=True, write_through=True)

    def thread_failed(args):
        logger.error("Uncaught background thread exception: %s", args.thread.name,
                     exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
                     extra={"event": "thread_failed"})
    threading.excepthook = thread_failed


def background_thread(*, target, args=(), **kwargs):
    """Keep request/job correlation when work leaves the HTTP request thread."""
    context = copy_context()
    return threading.Thread(target=context.run, args=(target, *args), **kwargs)


def record_failure(message, exc, log=None):
    job = current_job.get()
    if job is not None:
        job.update(status="failed", error=error_summary(exc))
    (log or logger).exception(message, extra={"event": "job_error"})


class JobHistory:
    def __init__(self, directory=None, retention_days=None):
        default_dir = Path(__file__).resolve().parent / "logs"
        self.directory = Path(directory or os.getenv("PILAR_LOG_DIR") or default_dir)
        self.retention_days = max(1, int(retention_days or os.getenv("PILAR_LOG_RETENTION_DAYS", "30")))
        self._lock = threading.RLock()
        self._cleaned_date = None

    def append(self, record):
        try:
            with self._lock:
                self.directory.mkdir(parents=True, exist_ok=True)
                today = datetime.now().date()
                if self._cleaned_date != today:
                    cutoff = today - timedelta(days=self.retention_days - 1)
                    for path in self.directory.glob("jobs-????-??-??.jsonl"):
                        try:
                            day = datetime.strptime(path.stem[5:], "%Y-%m-%d").date()
                        except ValueError:
                            continue
                        if day < cutoff:
                            path.unlink()
                    self._cleaned_date = today
                path = self.directory / f"jobs-{today.isoformat()}.jsonl"
                with path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except OSError:
            logger.exception("Could not persist job history", extra={"event": "history_write_failed"})

    def recent(self, stage=None, limit=20):
        records = []
        with self._lock:
            for path in sorted(self.directory.glob("jobs-????-??-??.jsonl"), reverse=True):
                try:
                    lines = path.read_text(encoding="utf-8").splitlines()
                except OSError:
                    logger.exception("Could not read job history")
                    continue
                for line in reversed(lines):
                    try:
                        record = json.loads(line)
                    except ValueError:
                        logger.warning("Ignoring incomplete job-history record in %s", path.name)
                        continue
                    if record.get("event") == "job_finished" and (stage is None or record.get("stage") == stage):
                        records.append(record)
                        if len(records) >= limit:
                            return records
        return records

    @contextmanager
    def run(self, stage, trigger=None):
        parent = current_job.get()
        job = {"run_id": uuid4().hex, "request_id": request_id.get(), "stage": stage,
               "trigger": trigger or (parent or {}).get("trigger") or ("web" if request_id.get() != "-" else "cli"),
               "parent_run_id": (parent or {}).get("run_id"),
               "started_at": timestamp(), "status": "running"}
        token = current_job.set(job)
        started = time.monotonic()
        self.append({**job, "event": "job_started", "time": job["started_at"]})
        logger.info("%s started", stage, extra={"event": "job_started", "trigger": job["trigger"]})
        try:
            yield job
        except Exception as exc:
            record_failure(f"{stage} failed", exc)
            raise
        finally:
            job.update(finished_at=timestamp(), duration_seconds=round(time.monotonic() - started, 3))
            if job["status"] == "running":
                job["status"] = "completed"
            self.append({**job, "event": "job_finished", "time": job["finished_at"]})
            level = logging.ERROR if job["status"] == "failed" else logging.INFO
            logger.log(level, "%s %s (%.3fs)", stage, job["status"], job["duration_seconds"],
                       extra={"event": "job_finished", "status": job["status"], "duration_seconds": job["duration_seconds"]})
            if parent is not None and job["status"] == "failed":
                parent.update(status="failed", error=job.get("error", f"{stage} failed"))
            current_job.reset(token)

    def track(self, stage, trigger=None):
        def decorate(function):
            @wraps(function)
            def tracked(*args, **kwargs):
                with self.run(stage, trigger) as job:
                    result = function(*args, **kwargs)
                    if isinstance(result, dict):
                        job.update({k: result[k] for k in ("status", "image_count", "title", "reason") if k in result})
                    return result
            return tracked
        return decorate
