"""User-facing structured log writer for web jobs."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from threading import Lock

__all__ = ["UserLog"]

USERLOG_MESSAGES = {
    # Mapping SAVs -> structures
    "SAV2PDB_NO_HITS": {
        "message": "Cannot find an experimental structure or AlphaFold2 structure for these SAVs:",
        "action": "If you want to predict the pathogenicity of these SAVs, please upload your own structure.",
    },
    "SAV2PDB_WT_MISMATCH": {
        "message": "Cannot map these SAVs due to residue mismatch:",
        "action": "Please ensure that the mutation is defined on the UniProt canonical sequence.",
    },
    "SAV2PDB_LOW_CONFIDENCE": {
        "message": "These SAVs fall in low-confidence regions (pLDDT < 50):",
        "action": "",
    },

    # PDB / structure preparation
    "PDB_PREP_FAILED": {
        "message": "Failed to prepare structure '{pdbID}' ({format}).",
        "action": "Verify structure source or provide a valid custom structure.",
    },
    "PDB_NOT_FOUND": {
        "message": "Prepared structure file not found for '{pdbID}' ({format}).",
        "action": "Try rerunning with refresh or verify external structure availability.",
    },
    "PDB_READ_FAILED": {
        "message": "Failed to read structure '{pdbID}' ({format}) for feature calculation.",
        "action": "Check if the structure file is complete and readable.",
    },
}

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
        stage: str,
        message: str,
        action: str | None = None,
        context: dict | None = None,
    ) -> None:
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": str(level),
            "stage": str(stage),
            "message": str(message),
            "action": action,
            "context": {**self.defaults, **(context or {})},
        }
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

