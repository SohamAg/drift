"""JSONL run logger.

A run writes 4 streams into `runs/<timestamp>/`:
  - events.jsonl     (one EventRecord per line)
  - actions.jsonl    (one Action per line)
  - snapshots.jsonl  (WorldState per timestep)
  - failures.jsonl   (FailureRecord per detection)

This is the persistent audit trail. The console summary is built by
re-reading these streams (or directly from in-memory equivalents).
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import IO

from pydantic import BaseModel


class RunLogger:
    def __init__(self, base_dir: Path | str = "runs", run_id: str | None = None) -> None:
        run_id = run_id or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(base_dir) / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._events: IO = (self.run_dir / "events.jsonl").open("w", encoding="utf-8")
        self._actions: IO = (self.run_dir / "actions.jsonl").open("w", encoding="utf-8")
        self._snapshots: IO = (self.run_dir / "snapshots.jsonl").open("w", encoding="utf-8")
        self._failures: IO = (self.run_dir / "failures.jsonl").open("w", encoding="utf-8")

    def _write(self, fh: IO, model: BaseModel) -> None:
        fh.write(model.model_dump_json() + "\n")

    def log_event(self, record: BaseModel) -> None: self._write(self._events, record)
    def log_action(self, record: BaseModel) -> None: self._write(self._actions, record)
    def log_snapshot(self, record: BaseModel) -> None: self._write(self._snapshots, record)
    def log_failure(self, record: BaseModel) -> None: self._write(self._failures, record)

    def close(self) -> None:
        for fh in (self._events, self._actions, self._snapshots, self._failures):
            try:
                fh.close()
            except Exception:
                pass

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, *args) -> None:
        self.close()
