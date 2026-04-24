"""User-facing structured log writer for web jobs."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from threading import Lock

__all__ = ["UserLog"]

VALIDATING_STAGE = "Validating SAVs"
MAPPING_STAGE = "Mapping SAVs to structures"
FEATURE_STAGE = "Feature calculation"
MODEL_STAGE = "Model inferencing/Training"
REPORT_STAGE = "Summary"

STAGE_LABELS = [
    VALIDATING_STAGE,
    MAPPING_STAGE,
    FEATURE_STAGE,
    MODEL_STAGE,
    REPORT_STAGE,
]

USERLOG_MESSAGES = {
    # VALIDATING_STAGE = "Validating SAVs"

    # MAPPING_STAGE = "Mapping SAVs to structures"
    "SAV2PDB_NO_HITS": {
        "message": "Cannot find an experimental structure or AlphaFold2 structure for these SAVs:",
        "action": "If you want to predict the pathogenicity of these SAVs, please upload your own structure.",
        "example": ["Q9P2D1 Y72C", "Q9P2D1 P86R"]
    },
    "SAV2PDB_WT_MISMATCH": {
        "message": "Cannot map these SAVs due to residue mismatch:",
        "action": "Please ensure that the mutation is defined on the UniProt canonical sequence.",
        "example": ["O00255 R176Q", "O00255 D177Y"]
    },
    "SAV2PDB_LOW_CONFIDENCE": {
        "message": "These SAVs fall in low-confidence regions (pLDDT < 50):",
        "action": "",
        "example": ["Q8TDI8 S2P", "Q8TDI8 K4Q", "Q8TDI8 I8V", "Q8TDI8 I8N"]
    },
    "SAV2PDB_FAILED": {
        "message": "None could be mapped. Your job is stopped.", 
        "action": "",
    },
    "NOT_RECOGNIZE_UNIPROT": {
        "message": "a",
        "action": "a",
        "example": []
    },
    "NOT_RECOGNIZE_RESID": {
        "message": "a",
        "action": "a",
        "example": []
    },
    "SAV2PDB_INVALID_CUSTOM_ID": {
        "message": "The provided custom PDB or AlphaFold identifier is not valid.",
        "action": "Please check the identifier format and try again.",
        "example": [],
    },
    "SAV2PDB_CUSTOM_STRUCTURE_UNREADABLE": {
        "message": "The uploaded custom structure could not be read.",
        "action": "Please verify that the structure file is complete and in a supported format.",
        "example": [],
    },

    # FEATURE_STAGE = "Feature calculation"
    # PDB / structure preparation
    "PDB_PREP_FAILED": {
        "message": "Failed to prepare structure '{pdbID}'.",
        "action": "Verify structure source or provide a valid custom structure.",
    },
    "PDB_NOT_FOUND": {
        "message": "Prepared structure file not found for '{pdbID}'.",
        "action": "Try rerunning with refresh or verify external structure availability.",
    },
    "PDB_READ_FAILED": {
        "message": "Failed to read structure '{pdbID}' for feature calculation.",
        "action": "Check if the structure file is complete and readable.",
    },
    "FEATURE_NO_STRUCTURE": {
        "message": "No feature calculation for these SAVs:",
        "action": "Please see features.txt and log.txt for details.",
    },
    "MISSING_FEATURE": {
        "message": "Missing {feature_text} features for these SAVs:",
        "action": "Please see features.txt and log.txt for details.",
    },


    # MODEL_STAGE = "Model inferencing/Training"
    "INF_DUMP_SAVS": {
        "message": "No prediction for these SAVs:",
        "action": "",
    },
    "TF_DUMP_SAVS": {
        "message": "No transfer learning for these SAVs:",
        "action": "",
    },
    "MODEL_NO_SAVS_AFTER_FILTERING": {
        "message": "No SAVs remain after filtering.",
        "action": "Please check the mapping and feature-calculation warnings above.",
        "example": [],
    },
    "MODEL_TOO_FEW_SAVS": {
        "message": "Too few SAVs remain for transfer learning.",
        "action": "Please provide more SAVs with usable structure mapping.",
        "example": [],
    },
    "MODEL_SINGLE_CLASS_LABELS": {
        "message": "Transfer learning requires at least two label classes, but only one class is present.",
        "action": "Please provide both benign and pathogenic labels for training.",
        "example": [],
    },

    "MODEL_BACKEND_FAILED": {
        "message": "Model inferencing or training failed after feature calculation.",
        "action": "Please check log.txt for detailed traceback.",
        "example": [],
    },

    # REPORT_STAGE = "Summary"
    "JOB_FAILED": {
        "message": "Job '{job_name}' failed: {error}",
        "action": "Please check log.txt for detailed traceback.",
    },
}

class UserLog:
    """Append-only JSONL logger for UI-friendly job events."""

    def __init__(self, path: str, defaults: dict | None = None):
        self.path = os.path.abspath(path)
        self.defaults = defaults.copy() if defaults else {}
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._lock = Lock()
        self._times = {}
        self.emit_list = []

    def timeit(self, label: str) -> None:
        """Start a named wall-clock timer."""
        self._times[str(label)] = time.time()

    def report(self, label: str, stage: str, file: str) -> float | None:
        """Emit a user-log event for a named timer and return elapsed seconds."""
        timer_label = str(label)
        started_at = self._times.get(timer_label)
        if started_at is None:
            return None

        elapsed_seconds = max(0.0, time.time() - started_at)
        context = {
            "timer_label": timer_label,
            "duration_seconds": elapsed_seconds,
            "duration_text": self.format_time(elapsed_seconds),
            "file": file
        }
        message=f"{timer_label} completed."
        self.emit(level='info', stage=stage, message=message, context=context)

    def format_time(self, seconds: float) -> str:
        """Format seconds into a short human-readable duration string."""
        seconds = max(0.0, float(seconds))
        if seconds >= 3600:
            return f"{seconds / 3600:.1f} h"
        if seconds >= 60:
            return f"{seconds / 60:.1f} min"
        return f"{seconds:.1f} s"

    def emit(self, level, stage, message, action=None, context: dict | None = None):
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": str(level),
            "stage": str(stage),
            "message": str(message),
            "action": action,
            "context": {**self.defaults, **(context or {})},
        }
        with self._lock:
            self.emit_list.append(row)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def format_message(self, stage_events):
        """Format one stage block for report.txt."""
        warning_events = [event for event in stage_events if str(event.get("level", "")).lower() == "warning"]
        error_events = [event for event in stage_events if str(event.get("level", "")).lower() == "error"]
        info_events = [event for event in stage_events if str(event.get("level", "")).lower() == "info"]

        if not warning_events and not error_events:
            return "OK" if info_events else "Not reached"

        lines = []
        for event in warning_events + error_events:
            level = str(event.get("level", "")).upper().strip()
            message = str(event.get("message", "")).strip()
            action = str(event.get("action", "") or "").strip()
            context = event.get("context", {})
            context = context if isinstance(context, dict) else {}
            savs = context.get("savs", [])
            if isinstance(savs, (list, tuple)):
                sav_text = ", ".join(str(sav).strip() for sav in savs if str(sav).strip())
            else:
                sav_text = str(savs or "").strip()

            message_line = f"> {level} {message}" if message else level
            if sav_text:
                message_line += f" {sav_text}"
            lines.append(message_line)
            if action:
                lines.append(action)
        return "\n".join(lines)

    def dump_report(self, report_path):
        """Write a human-readable stage summary based on cached emit events."""
        with self._lock:
            events = list(self.emit_list)

        blocks = []
        for stage_label in STAGE_LABELS:
            stage_events = [event for event in events if str(event.get("stage", "")) == stage_label]
            stage_summary = self.format_message(stage_events)
            blocks.append(
                "\n".join(
                    [
                        "--------------------------",
                        stage_label,
                        "--------------------------",
                        stage_summary,
                    ]
                )
            )

        with open(report_path, "w", encoding="utf-8") as handle:
            handle.write("\n\n".join(blocks) + "\n")
