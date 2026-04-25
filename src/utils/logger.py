"""This module defines class that can be used a package wide logger.

This code was copied from Prody package and modified by Loci Tran
"""

import sys
import math
import time
import json
import os.path
import logging
import datetime
import logging.handlers
import numbers
import warnings
import re
import textwrap
# from datetime import datetime, timezone
from threading import Lock

__all__ = [
    'PackageLogger', 'LOGGING_LEVELS', 'LOGGER',
    'VALIDATING_STAGE', 'MAPPING_STAGE', 'FEATURE_STAGE',
    'MODEL_STAGE', 'REPORT_STAGE', 'STAGE_LABELS',
    'USERLOG_MESSAGES',
]

LOGGING_PROGRESS = logging.INFO + 5

LOGGING_LEVELS = {'debug': logging.DEBUG,
                'info': logging.INFO,
                'progress': LOGGING_PROGRESS,
                'warning': logging.WARNING,
                'error': logging.ERROR,
                'critical': logging.CRITICAL,
                'none': logging.CRITICAL}
LOGGING_INVERSE = {}
for key, value in LOGGING_LEVELS.items(): # PY3K: OK
    LOGGING_INVERSE[value] = key

now = datetime.datetime.now

warnings.filterwarnings("ignore", message=".*failed to parse occupancy.*")
warnings.filterwarnings("ignore", message=".*failed to parse beta-factor.*")


"""
A0FGR8 S638G
O00189 R271H
O00189 R27111H
Q8XXXXI8 S2P
Q8TDI8 I8N
O00255 D177Y
Q9P2D1 Y72C

---
"""

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
    "SAV2PDB_NO_HITS": {
        "message": "Cannot find an experimental structure or AlphaFold2 structure for these SAVs below.\nIf you want to predict the pathogenicity of these SAVs, please upload your own structure.",
        "example": ["Q9P2D1 Y72C", "Q9P2D1 P86R"]
    },
    "SAV2PDB_WT_MISMATCH": {
        "message": "Cannot map these SAVs below due to residue mismatch.\nPlease ensure that the mutation is defined on the UniProt canonical sequence.",
        "example": ["O00255 R176Q", "O00255 D177Y"]
    },
    "SAV2PDB_OUT_RANGE": {
        "message": "Cannot map these SAVs below due to residue index out of UniProt sequence.",
        "example": ["O00189 R27111H"]
    },
    "SAV2PDB_LOW_CONFIDENCE": {
        "message": "These SAVs fall in low-confidence regions (pLDDT < 50):",
        "example": ["Q8TDI8 S2P", "Q8TDI8 K4Q", "Q8TDI8 I8V", "Q8TDI8 I8N"]
    },
    "PFAM_NO_DOMAIN": {
        "message": "Cannot find any Pfam domain for these SAVs below.",
        "example": ["O00189 R271H"]
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
        "example": [],
    },
    "SAV2PDB_CUSTOM_STRUCTURE_UNREADABLE": {
        "message": "The uploaded custom structure could not be read.",
        "example": [],
    },



    
    "PDB_PREP_FAILED": {
        "message": "Failed to prepare structure '{pdbID}'.",
    },
    "PDB_NOT_FOUND": {
        "message": "Prepared structure file not found for '{pdbID}'.",
    },
    "PDB_READ_FAILED": {
        "message": "Failed to read structure '{pdbID}' for feature calculation.",
    },
    "FEATURE_NO_STRUCTURE": {
        "message": "No feature calculation for these SAVs:",
    },
    "MISSING_FEATURE": {
        "message": "Missing {feature_text} features for these SAVs:",
    },
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
    "JOB_FAILED": {
        "message": "Job '{job_name}' failed: {error}",
        "action": "Please check log.txt for detailed traceback.",
    },
}

# Patterns to strip from log files on close.
_LOG_FILTER_PATTERNS = [
    re.compile(r".*failed to parse occupancy.*"),
    re.compile(r".*failed to parse beta-factor.*"),
]

def _filter_logfile(path, patterns):
    try:
        with open(path, "r") as f:
            lines = f.readlines()
        with open(path, "w") as f:
            for line in lines:
                if any(p.search(line) for p in patterns):
                    continue
                f.write(line)
    except Exception:
        # Best-effort filtering; do not break logging on failure.
        return
    
class PackageLogger(object):

    """A class for package wide logging functionality."""

    def __init__(self, name, **kwargs):
        """Start logger for the package. Returns a logger instance.

        :arg prefix: prefix to console log messages, default is ``'@> '``
        :arg console: log level for console (``sys.stderr``) messages,
            default is ``'debug'``
        :arg info: prefix to log messages at *info* level
        :arg warning: prefix to log messages at *warning* level, default is
            ``'WARNING '``
        :arg error: prefix to log messages at *error* level, default is
            ``'ERROR '``
        """

        self._level = logging.DEBUG
        self._logger = logger = logging.getLogger(name)
        logger.setLevel(self._level)


        for handler in logger.handlers:
            handler.close()
        logger.handlers = []

        console = logging.StreamHandler()
        console.setLevel(LOGGING_LEVELS[kwargs.get('console', 'debug')])
        logger.addHandler(console)
        self.prefix = kwargs.get('prefix', '@> ')

        self._info = kwargs.get('info', '')
        self._warning = kwargs.get('warning', 'WARNING ')
        self._error = kwargs.get('error', 'ERROR ')

        self._n = None
        self._barlen = None
        self._line = None
        self._times = {}
        self._info = {}

        self._n_progress = 0

        self._times = {}
        self._reports = {}
        self._report_times = {}
        self._userlog_path = None
        self._userlog_defaults = {}
        self._emit_list = []
        self._emit_lock = Lock()

    # ====================
    # Attributes
    # ====================

    def _getverbosity(self):

        return LOGGING_INVERSE.get(self._logger.handlers[0].level)

    def _setverbosity(self, level):
        lvl = LOGGING_LEVELS.get(str(level).lower(), None)
        if lvl is None:
            self.warn('{0} is not a valid log level.'.format(level))
        else:
            self._logger.handlers[0].level = lvl
            self._level = lvl

    verbosity = property(_getverbosity, _setverbosity, doc=
        """Verbosity *level* of the logger, default level is **debug**.  Log
        messages are written to ``sys.stderr``.  Following logging levers are
        recognized:

        ========  =============================================
        Level     Description
        ========  =============================================
        debug     Everything will be printed to the sys.stderr.
        info      Only brief information will be printed.
        warning   Only warning messages will be printed.
        none      Nothing will be printed.
        ========  =============================================""")

    def _getprefix(self):

        return self._prefix

    def _setprefix(self, prefix):

        self._prefix = str(prefix)
        prefix += '%(message)s'
        self._logger.handlers[0].setFormatter(logging.Formatter(prefix))

    prefix = property(_getprefix, _setprefix, doc='String prepended to console'
                      ' log messages.')

    # ====================
    # Logging methods
    # ====================

    def info(self, msg):
        """Log *msg* with severity 'INFO'."""

        self.clear()
        self._logger.info(str(msg))

    def critical(self, msg):
        """Log *msg* with severity 'CRITICAL'."""

        self.clear()
        self._logger.critical(str(msg))

    def debug(self, msg):
        """Log *msg* with severity 'DEBUG'."""

        self.clear()
        self._logger.debug(str(msg))

    def warning(self, msg):
        """Log *msg* with severity 'WARNING'."""

        self.clear()
        self._logger.warning(self._warning + str(msg))

    warn = warning

    def error(self, msg):
        """Log *msg* with severity 'ERROR' and terminate with status 2."""

        self.clear()
        self._logger.error(self._error + str(msg))
        self.exit(2)

    def _normalize_for_json(self, value):
        if isinstance(value, dict):
            return {str(k): self._normalize_for_json(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._normalize_for_json(v) for v in value]
        if hasattr(value, "item") and callable(getattr(value, "item")):
            try:
                return value.item()
            except Exception:
                pass
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def start_userlog(self, path, defaults=None, mode="w"):
        self._userlog_path = os.path.abspath(path)
        self._userlog_defaults = defaults.copy() if defaults else {}
        self._emit_list = []
        os.makedirs(os.path.dirname(self._userlog_path), exist_ok=True)
        with open(self._userlog_path, mode, encoding="utf-8"):
            pass

    def emit(self, level, stage, message, savs=None, action=None, context=None, exit_on_error=True):
        context = self._normalize_for_json({**self._userlog_defaults, **(context or {})})
        row = {
            "timestamp": now(datetime.timezone.utc).isoformat(),
            "level": str(level),
            "stage": str(stage),
            "message": str(message),
            "savs": savs,
            "action": action,
            "context": context,
        }
        with self._emit_lock:
            self._emit_list.append(row)
            if self._userlog_path:
                with open(self._userlog_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

        level_lower = str(level).lower()
        if level_lower == "debug":
            self.debug(str(message))
        elif level_lower == "info":
            self.info(str(message))
        elif level_lower == "important":
            self.info(str(message))
        elif level_lower == "warning":
            self.warning(str(message))
        elif level_lower == "error":
            self.clear()
            self._logger.error(self._error + str(message))
            if exit_on_error:
                self.exit(2)
        elif level_lower == "critical":
            self.clear()
            self._logger.critical(str(message))
            if exit_on_error:
                self.exit(2)
        else:
            self.info(str(message))

    def format_time(self, seconds):
        seconds = max(0.0, float(seconds))
        if seconds >= 3600:
            return f"{seconds / 3600:.1f} h"
        if seconds >= 60:
            return f"{seconds / 60:.1f} min"
        return f"{seconds:.1f} s"

    def report_userlog(self, label, stage, file):
        timer_label = str(label)
        started_at = self._times.get(timer_label)
        if started_at is None:
            return None

        elapsed_seconds = max(0.0, time.time() - started_at)
        context = {
            "timer_label": timer_label,
            "duration_seconds": elapsed_seconds,
            "duration_text": self.format_time(elapsed_seconds),
            "file": file,
        }
        message = f"{timer_label} completed."
        self.emit(level="info", stage=stage, message=message, context=context)
        return elapsed_seconds

    def format_stage_message(self, stage_events):
        warning_events = [event for event in stage_events if str(event.get("level", "")).lower() == "warning"]
        error_events = [event for event in stage_events if str(event.get("level", "")).lower() == "error"]
        info_events = [event for event in stage_events if str(event.get("level", "")).lower() in {"info", "important"}]

        if not warning_events and not error_events:
            return "OK" if info_events else "Not reached"

        lines = []
        for event in warning_events + error_events:
            level = str(event.get("level", "")).upper().strip()
            message = str(event.get("message", "")).strip()
            savs = event.get("savs", [])
        
            message_line = f"> {level} {message}" if message else level
            
            if isinstance(savs, list):
                sav_text = ", ".join(str(sav).strip() for sav in savs)
            else:
                sav_text = str(savs or "").strip()
            if sav_text:
                message_line += f" {sav_text}"
            lines.extend(
                textwrap.wrap(message_line, width=100,
                    break_long_words=False,
                    break_on_hyphens=False,
                    subsequent_indent="  ",
                ) or [message_line]
            )

        return "\n".join(lines)

    def dump_userlog(self, report_path, stage_labels):
        with self._emit_lock:
            events = list(self._emit_list)

        blocks = []
        for stage_label in stage_labels:
            stage_events = [event for event in events if str(event.get("stage", "")) == stage_label]
            stage_summary = self.format_stage_message(stage_events)
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

    def write(self, line):
        """Write *line* into ``sys.stderr``."""

        self._line = str(line)
        if self._level < logging.WARNING:
            sys.stderr.write(self._line)
            sys.stderr.flush()

    def clear(self):
        """Clear current line in ``sys.stderr``."""

        if self._level != LOGGING_PROGRESS: 
            if self._line and self._level < logging.WARNING:
                sys.stderr.write('\r' + ' ' * (len(self._line)) + '\r')
                self._line = ''

    def exit(self, status=0):
        """Exit the interpreter."""

        sys.exit(status)

    # ====================
    # Handlers & logfiles
    # ====================

    def addHandler(self, hdlr):
        """Add the specified handler to this logger."""

        self._logger.addHandler(hdlr)

    def getHandlers(self):
        """Returns handlers."""

        return self._logger.handlers

    def delHandler(self, index):
        """Remove handler at given *index* from the logger instance."""

        self._logger.handlers.pop(index)

    def start(self, filename, **kwargs):
        """Start a logfile.  If *filename* does not have an extension.
        :file:`.log` will be appended to it.

        :arg filename: name of the logfile
        :arg mode: mode in which logfile will be opened, default is "w"
        :arg backupcount: number of existing *filename.log* files to
            backup, default is 1"""

        filename = str(filename)
        if os.path.splitext(filename)[1] == '':
            filename += '.log'
        rollover = False
        if os.path.isfile(filename) and kwargs.get('mode', None) != 'a':
            rollover = True

        logfile = logging.handlers.RotatingFileHandler(
            filename,
            mode=kwargs.get('mode', 'a'),
            maxBytes=0,
            backupCount=kwargs.get('backupcount', 1)
        )
        logfile.setLevel(LOGGING_LEVELS[kwargs.get('loglevel', 'debug')])
        logfile.setFormatter(logging.Formatter('%(message)s'))

        # Attach to the root logger so all loggers propagate here
        root_logger = logging.getLogger()
        root_logger.addHandler(logfile)
        root_logger.setLevel(logging.INFO)

        self._logger.info(f"Logging into file: {filename}")
        if rollover:
            logfile.doRollover()
        self._logger.info(f"Logging started at {str(now())}")

    def close(self, filename):
        filename = str(filename)
        if os.path.splitext(filename)[1] == '':
            filename += '.log'
        root_logger = logging.getLogger()
        for index, handler in enumerate(root_logger.handlers):
            if isinstance(handler, logging.handlers.RotatingFileHandler):
                if handler.stream.name in (filename, os.path.abspath(filename)):
                    self.info("Logging stopped at {0}".format(str(now())))
                    handler.close()
                    root_logger.removeHandler(handler)
                    _filter_logfile(filename, _LOG_FILTER_PATTERNS)
                    self.info("Closing logfile: {0}".format(filename))
                    return
        self.warning("Logfile '{0}' was not found.".format(filename))

    # ====================
    # Progress and timing
    # ====================

    def progress(self, msg, steps, label=None, **kwargs):
        """Instantiate a labeled process with message and number of steps."""

        if steps is not None:  # if None then no upperlimit
            assert isinstance(steps, numbers.Integral) and steps > 0, \
                'steps must be a positive integer'
        
        self._times[label] = time.time()
        self._info[label] = {}
        self._info[label]['steps'] = steps
        self._info[label]['msg'] = msg
        self._info[label]['last'] = 0

        if not hasattr(self, '_verb'):
            self._verb = self._getverbosity()
            if self._level < logging.WARNING:
                self._setverbosity('progress')
        self._n_progress += 1

    def update(self, step, msg=None, label=None):
        """Update progress status to current line in the console."""

        assert isinstance(step, numbers.Integral), 'step must be a positive integer'
        
        if msg is None:
            msg = self._info[label]['msg']
        else:
            self._info[label]['msg'] = msg
        
        last = self._info[label]['last']
        n = self._info[label]['steps']
        i = step
        if self._level < logging.WARNING:
            start = self._times[label]
            sys.stderr.write('\r' + ' ' * last + '\r')
            if n is None:  # no upperlimit
                line = self._prefix + msg % i
            elif i <= n:
                percent = 100 * i / n
                if percent > 3:
                    seconds = int(math.ceil((time.time()-start) * (n-i)/i))
                    line = self._prefix + msg + ' [%3d%%] %ds' % (percent, seconds)
                else:
                    line = self._prefix + msg + ' [%3d%%]' % percent
            else:
                return
            sys.stderr.write(line)
            sys.stderr.flush()
            self._line = line
            self._info[label]['last'] = len(line)

    def finish(self):
        self._n_progress -= 1
        if self._n_progress < 0:
            self._n_progress = 0
        if self._n_progress == 0:
            if hasattr(self, '_verb'):
                self._setverbosity(self._verb)
                del self._verb
                self.clear()

    def sleep(self, seconds, msg=''):
        """Sleep for seconds while updating screen message every second.
        Message will start with ``'Waiting for Xs '`` followed by *msg*."""

        msg = str(msg)
        for second in range(int(seconds), 0, -1):
            self.write('Waiting for {0}s {1}'.format(second, msg))
            time.sleep(1)
            self.clear()

    def timeit(self, label=None):
        """Start timing a process.  Use :meth:`timing` and :meth:`report` to
        learn and report timing, respectively."""

        self._times[label] = time.time()

    def timing(self, label=None):
        """Returns timing for a labeled or default (**None**) process."""

        return time.time() - self._times.get(label, 0)

    def report(self, msg='Completed in %.2fs.', label=None):
        """Write *msg* with timing information for a labeled or default process
        at *debug* logging level."""

        if label not in self._times:
            self.warning(f"No timing info for label '{label}'")
            return
        elapsed = time.time() - self._times[label]
        self.debug(msg % elapsed)

        if label not in self._reports:
            self._reports[label] = elapsed
            self._report_times[label] = 1
        else:
            self._reports[label] += elapsed
            self._report_times[label] += 1

    def dump_time(self, output_path, extra_data=None):
        """Dump logger timing summaries to a JSON file.

        Inputs:
        - output_path: file path to write JSON
        - extra_data: optional dict to merge into output
        """
        report_data = {}

        for label in sorted(getattr(self, "_reports", {})):
            seconds = getattr(self, "_reports", {}).get(label)
            count = getattr(self, "_report_times", {}).get(label, 1)
            report_data[label] = {
                "seconds": round(float(seconds), 6),
                "count": int(count),
            }

        if isinstance(extra_data, dict):
            report_data.update(extra_data)

        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(report_data, handle, indent=2)


LOGGER = PackageLogger('.prody')
