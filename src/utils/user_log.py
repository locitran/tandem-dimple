"""User-facing structured log writer for web jobs."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from threading import Lock

__all__ = ["UserLog"]


class UserLog:
    """Append-only JSONL logger for UI-friendly job events."""

    def __init__(self, path: str, defaults: dict | None = None):
        self.path = os.path.abspath(path)
        self.defaults = defaults.copy() if defaults else {}
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._lock = Lock()

    def emit(
        self,
        level: str,
        code: str,
        stage: str,
        message: str,
        action: str | None = None,
        context: dict | None = None,
    ) -> None:
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": str(level),
            "code": str(code),
            "stage": str(stage),
            "message": str(message),
            "action": action,
            "context": {**self.defaults, **(context or {})},
        }
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

