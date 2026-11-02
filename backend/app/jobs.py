"""Bounded, process-local background jobs for a single-host dashboard."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from threading import RLock
from uuid import uuid4

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='nyc-job')
_lock = RLock()
_jobs: dict[str, dict] = {}


def submit(kind, work):
    with _lock:
        if any(j['kind'] == kind and j['status'] in ('queued', 'running') for j in _jobs.values()):
            raise ValueError(f'A {kind} job is already running')
        while len(_jobs) >= 8:
            oldest = next((key for key, j in _jobs.items() if j['status'] in ('complete', 'failed')), None)
            if oldest is None:
                raise ValueError('Job queue is full')
            del _jobs[oldest]
        key = uuid4().hex
        _jobs[key] = {'id': key, 'kind': kind, 'status': 'queued', 'created_at': datetime.now(timezone.utc).isoformat()}

    def run():
        with _lock:
            _jobs[key]['status'] = 'running'
        try:
            result = work()
            with _lock:
                _jobs[key].update(status='complete', result=result)
        except Exception as exc:
            with _lock:
                _jobs[key].update(status='failed', error=str(exc))
    _executor.submit(run)
    return {'id': key, 'kind': kind, 'status': 'queued'}


def get_job(key):
    with _lock:
        if key not in _jobs:
            raise KeyError(key)
        return _jobs[key].copy()


def active_jobs():
    with _lock:
        return [{k: v for k, v in j.items() if k != 'result'} for j in _jobs.values()]
