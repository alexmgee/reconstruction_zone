"""Locked, process-local representative jobs for the localhost web API."""

from __future__ import annotations

import re
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Callable

__all__ = ["JobRecord", "JobRegistry", "JobRegistryError", "JobState"]

MAX_ACTIVE_JOBS = 4
MAX_TERMINAL_JOBS = 100
MAX_LOG_ENTRIES = 200
REPRESENTATIVE_STEPS = 20

_JOB_ID_RE = re.compile(r"[0-9a-f]{32}\Z")
_TERMINAL_STATES = frozenset({"completed", "failed", "cancelled"})

_LEGAL_TRANSITIONS = {
    "queued": frozenset({"running", "cancel_requested", "failed"}),
    "running": frozenset({"completed", "cancel_requested", "failed"}),
    "cancel_requested": frozenset({"cancelled", "failed"}),
    "completed": frozenset(),
    "failed": frozenset(),
    "cancelled": frozenset(),
}

_STEP_LABELS = {
    "representative_work": "representative_work",
    "representative_finishing": "representative_finishing",
}

_LOG_CATALOG = {
    "representative_started": "Representative job started.",
    "representative_progress": "Representative job reached a progress checkpoint.",
    "representative_completed": "Representative job completed.",
    "representative_failure": "Representative job failed as requested.",
    "cancel_requested": "Cancellation was requested.",
    "representative_cancelled": "Representative job was cancelled.",
    "job_start_failure": "Representative job could not be started.",
    "internal_job_error": "Representative job failed safely.",
}

_ERROR_CATALOG = {
    "representative_failure": "Representative job failed as requested.",
    "job_start_failure": "Representative job could not be started.",
    "internal_job_error": "Representative job failed safely.",
}


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    CANCEL_REQUESTED = "cancel_requested"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobRegistryError(RuntimeError):
    """A fixed public registry error safe for an HTTP response."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


class _TransitionError(RuntimeError):
    pass


@dataclass
class JobRecord:
    job_id: str
    behavior: str
    sequence: int
    state: JobState = JobState.QUEUED
    progress: int = 0
    current_step: str | None = None
    created_at: str = field(default_factory=lambda: _utc_timestamp())
    started_at: str | None = None
    finished_at: str | None = None
    logs: list[dict[str, str]] = field(default_factory=list)
    result: dict[str, object] | None = None
    error: dict[str, str] | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)


Runner = Callable[["JobRegistry", str, str, threading.Event], None]
ThreadStart = Callable[[threading.Thread], None]


class JobRegistry:
    """A single-lock, in-memory registry for representative jobs."""

    def __init__(
        self,
        *,
        _runner: Runner | None = None,
        _thread_start: ThreadStart | None = None,
        _step_wait: float = 0.01,
    ) -> None:
        if _step_wait < 0:
            raise ValueError("step wait must be non-negative")
        self._lock = threading.Lock()
        self._jobs: dict[str, JobRecord] = {}
        self._workers: dict[str, threading.Thread] = {}
        self._accepting = True
        self._next_sequence = 0
        self._runner = _runner or _representative_runner
        self._thread_start = _thread_start or _start_thread
        self._step_wait = _step_wait

    def create_representative(self, *, behavior: str) -> dict[str, object]:
        if behavior not in {"complete", "fail"}:
            raise JobRegistryError(400, "invalid_request", "Invalid request.")

        with self._lock:
            self._prune_terminals_locked()
            if not self._accepting:
                raise JobRegistryError(409, "server_shutting_down", "Server is shutting down.")
            active_count = sum(
                record.state.value not in _TERMINAL_STATES for record in self._jobs.values()
            )
            if active_count >= MAX_ACTIVE_JOBS:
                raise JobRegistryError(409, "job_limit_reached", "Active job limit reached.")

            job_id = uuid.uuid4().hex
            self._next_sequence += 1
            record = JobRecord(job_id=job_id, behavior=behavior, sequence=self._next_sequence)
            worker = threading.Thread(
                target=self._worker_entry,
                args=(job_id,),
                name="reconstruction-web-representative-job",
                daemon=True,
            )
            self._jobs[job_id] = record
            self._workers[job_id] = worker
            try:
                self._thread_start(worker)
            except Exception:
                self._transition_locked(record, JobState.FAILED)
                record.finished_at = _utc_timestamp()
                record.error = _error_payload("job_start_failure")
                self._append_log_locked(record, "error", "job_start_failure")
                self._workers.pop(job_id, None)
                self._prune_terminals_locked()
            return self._snapshot_locked(record, full=True)

    def list_summaries(self) -> list[dict[str, object]]:
        with self._lock:
            records = sorted(self._jobs.values(), key=lambda item: item.sequence, reverse=True)
            return [self._snapshot_locked(record, full=False) for record in records]

    def get_snapshot(self, job_id: str) -> dict[str, object]:
        with self._lock:
            record = self._lookup_locked(job_id)
            return self._snapshot_locked(record, full=True)

    def request_cancel(self, job_id: str) -> tuple[bool, dict[str, object]]:
        with self._lock:
            record = self._lookup_locked(job_id)
            if record.state.value in _TERMINAL_STATES:
                return False, self._snapshot_locked(record, full=True)
            if record.state is JobState.CANCEL_REQUESTED:
                return True, self._snapshot_locked(record, full=True)
            self._transition_locked(record, JobState.CANCEL_REQUESTED)
            record.cancel_event.set()
            self._append_log_locked(record, "info", "cancel_requested")
            return True, self._snapshot_locked(record, full=True)

    def update_progress(self, job_id: str, *, progress: int, step_code: str) -> None:
        if isinstance(progress, bool) or not isinstance(progress, int) or not 0 <= progress <= 100:
            raise ValueError("progress must be an integer from 0 through 100")
        if step_code not in _STEP_LABELS:
            raise ValueError("unknown representative step")
        with self._lock:
            record = self._lookup_locked(job_id)
            if record.state is not JobState.RUNNING:
                raise _TransitionError("progress requires a running job")
            if progress < record.progress:
                raise ValueError("progress cannot decrease")
            if progress == 100:
                raise ValueError("progress 100 is reserved for completion")
            record.progress = progress
            record.current_step = _STEP_LABELS[step_code]
            self._append_log_locked(record, "info", "representative_progress")

    def append_log(self, job_id: str, *, level: str, code: str) -> None:
        with self._lock:
            record = self._lookup_locked(job_id)
            if record.state.value in _TERMINAL_STATES:
                raise _TransitionError("terminal jobs cannot be updated")
            self._append_log_locked(record, level, code)

    def begin_shutdown(self) -> None:
        """Stop creation and request cancellation without waiting for workers."""
        with self._lock:
            self._accepting = False
            for record in self._jobs.values():
                if record.state in {JobState.QUEUED, JobState.RUNNING}:
                    self._transition_locked(record, JobState.CANCEL_REQUESTED)
                    record.cancel_event.set()
                    self._append_log_locked(record, "info", "cancel_requested")
                elif record.state is JobState.CANCEL_REQUESTED:
                    record.cancel_event.set()

    def shutdown(self, *, join_timeout: float) -> None:
        """Cancel and join all representative workers outside the registry lock."""
        if join_timeout < 0:
            raise ValueError("join timeout must be non-negative")
        self.begin_shutdown()
        with self._lock:
            workers = list(self._workers.values())
        deadline = _monotonic() + join_timeout
        for worker in workers:
            remaining = max(0.0, deadline - _monotonic())
            worker.join(remaining)
        if any(worker.is_alive() for worker in workers):
            raise RuntimeError("Representative worker did not stop during shutdown.")
        with self._lock:
            self._prune_terminals_locked()

    def has_live_workers(self) -> bool:
        with self._lock:
            return any(worker.is_alive() for worker in self._workers.values())

    def _worker_entry(self, job_id: str) -> None:
        try:
            with self._lock:
                record = self._jobs[job_id]
                if record.state is JobState.QUEUED:
                    self._transition_locked(record, JobState.RUNNING)
                    record.started_at = _utc_timestamp()
                    record.current_step = "representative_work"
                    self._append_log_locked(record, "info", "representative_started")
                cancelled_before_claim = record.state is JobState.CANCEL_REQUESTED
                behavior = record.behavior
                cancel_event = record.cancel_event
            if cancelled_before_claim:
                self._acknowledge_cancel(job_id)
            else:
                self._runner(self, job_id, behavior, cancel_event)
                self._terminalize_returned_runner(job_id)
        except Exception:
            self._fail_unexpected(job_id)
        finally:
            with self._lock:
                self._workers.pop(job_id, None)
                self._prune_terminals_locked()

    def _terminalize_returned_runner(self, job_id: str) -> None:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None or record.state.value in _TERMINAL_STATES:
                return
            if record.state is JobState.CANCEL_REQUESTED:
                self._cancel_locked(record)
                return
            self._fail_locked(record, "internal_job_error")

    def _complete(self, job_id: str) -> None:
        with self._lock:
            record = self._lookup_locked(job_id)
            if record.state is JobState.CANCEL_REQUESTED:
                self._cancel_locked(record)
                return
            if record.state is not JobState.RUNNING:
                raise _TransitionError("completion requires a running job")
            self._transition_locked(record, JobState.COMPLETED)
            record.progress = 100
            record.current_step = "representative_finishing"
            record.finished_at = _utc_timestamp()
            record.result = {"kind": "representative", "steps_completed": REPRESENTATIVE_STEPS}
            record.error = None
            self._append_log_locked(record, "info", "representative_completed")

    def _fail_requested(self, job_id: str) -> None:
        with self._lock:
            record = self._lookup_locked(job_id)
            if record.state is JobState.CANCEL_REQUESTED:
                self._cancel_locked(record)
                return
            if record.state is not JobState.RUNNING:
                raise _TransitionError("failure requires a running job")
            self._fail_locked(record, "representative_failure")

    def _acknowledge_cancel(self, job_id: str) -> None:
        with self._lock:
            record = self._lookup_locked(job_id)
            if record.state is JobState.CANCEL_REQUESTED:
                self._cancel_locked(record)
            elif record.state.value not in _TERMINAL_STATES:
                raise _TransitionError("cancellation was not requested")

    def _fail_unexpected(self, job_id: str) -> None:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None or record.state.value in _TERMINAL_STATES:
                return
            self._fail_locked(record, "internal_job_error")

    def _fail_locked(self, record: JobRecord, code: str) -> None:
        self._transition_locked(record, JobState.FAILED)
        record.finished_at = _utc_timestamp()
        record.current_step = None
        record.result = None
        record.error = _error_payload(code)
        self._append_log_locked(record, "error", code)

    def _cancel_locked(self, record: JobRecord) -> None:
        self._transition_locked(record, JobState.CANCELLED)
        record.finished_at = _utc_timestamp()
        record.current_step = None
        record.result = None
        record.error = None
        self._append_log_locked(record, "info", "representative_cancelled")

    def _lookup_locked(self, job_id: str) -> JobRecord:
        if not isinstance(job_id, str) or _JOB_ID_RE.fullmatch(job_id) is None:
            raise JobRegistryError(404, "job_not_found", "Job not found.")
        record = self._jobs.get(job_id)
        if record is None:
            raise JobRegistryError(404, "job_not_found", "Job not found.")
        return record

    def _transition_locked(self, record: JobRecord, new_state: JobState) -> None:
        if new_state.value not in _LEGAL_TRANSITIONS[record.state.value]:
            raise _TransitionError("illegal job state transition")
        record.state = new_state

    def _append_log_locked(self, record: JobRecord, level: str, code: str) -> None:
        if level not in {"info", "warning", "error"} or code not in _LOG_CATALOG:
            raise ValueError("unknown public log entry")
        if len(record.logs) >= MAX_LOG_ENTRIES:
            del record.logs[0]
        record.logs.append(
            {
                "time": _utc_timestamp(),
                "level": level,
                "code": code,
                "message": _LOG_CATALOG[code],
            }
        )

    def _snapshot_locked(self, record: JobRecord, *, full: bool) -> dict[str, object]:
        snapshot: dict[str, object] = {
            "job_id": record.job_id,
            "job_type": "representative",
            "state": record.state.value,
            "progress": record.progress,
            "current_step": record.current_step,
            "created_at": record.created_at,
            "started_at": record.started_at,
            "finished_at": record.finished_at,
        }
        if full:
            snapshot["logs"] = [dict(entry) for entry in record.logs]
            snapshot["result"] = dict(record.result) if record.result is not None else None
            snapshot["error"] = dict(record.error) if record.error is not None else None
        return snapshot

    def _prune_terminals_locked(self) -> None:
        terminals = [
            record for record in self._jobs.values() if record.state.value in _TERMINAL_STATES
        ]
        excess = len(terminals) - MAX_TERMINAL_JOBS
        if excess <= 0:
            return
        eligible = sorted(
            (record for record in terminals if record.job_id not in self._workers),
            key=lambda item: item.sequence,
        )
        for record in eligible[:excess]:
            self._workers.pop(record.job_id, None)
            self._jobs.pop(record.job_id, None)


def _representative_runner(
    registry: JobRegistry,
    job_id: str,
    behavior: str,
    cancel_event: threading.Event,
) -> None:
    if cancel_event.is_set():
        registry._acknowledge_cancel(job_id)
        return
    for step in range(1, REPRESENTATIVE_STEPS + 1):
        if cancel_event.is_set():
            registry._acknowledge_cancel(job_id)
            return
        if cancel_event.wait(registry._step_wait):
            registry._acknowledge_cancel(job_id)
            return
        if cancel_event.is_set():
            registry._acknowledge_cancel(job_id)
            return
        if step < REPRESENTATIVE_STEPS:
            registry.update_progress(
                job_id,
                progress=step * (100 // REPRESENTATIVE_STEPS),
                step_code="representative_work",
            )
        if behavior == "fail" and step == REPRESENTATIVE_STEPS // 2:
            registry._fail_requested(job_id)
            return
    if cancel_event.is_set():
        registry._acknowledge_cancel(job_id)
        return
    registry._complete(job_id)


def _error_payload(code: str) -> dict[str, str]:
    return {"code": code, "message": _ERROR_CATALOG[code]}


def _start_thread(thread: threading.Thread) -> None:
    thread.start()


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _monotonic() -> float:
    # Kept behind a tiny seam so bounded shutdown tests do not need wall-clock sleeps.
    import time

    return time.monotonic()
