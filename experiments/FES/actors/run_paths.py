"""Run-scoped output paths and logging, shared by every FES actor.

Two problems this solves.

**Split runs.** Each actor used to call ``time.strftime("%Y%m%d-%H%M")`` in its
own ``setup()``, in its own process. They agreed only because the format is
minute-resolution and setup normally finishes inside one minute. But
``camera_reader`` deliberately sleeps ~2 s for the capture stagger and
``processor`` loads a DLC model, so a run whose setup straddles a minute
boundary scattered its ``.npy`` files across two sibling folders. Now the run id
is computed once -- by the launcher, into ``$IMPROV_RUN_ID`` -- and every actor
reads it back.

**Logs in the CWD.** Each actor also opened ``logging.FileHandler("<name>.log")``
with a bare relative path, at import time, so the logs landed wherever
``improv run`` happened to be invoked from and were appended to across every run
ever. Now they go to ``<run folder>/logs/``.

Both fall back sanely when an actor is run directly, outside the launcher: the
run id becomes the current minute, exactly as before.
"""

import logging
import os
import re
import time
from pathlib import Path

import yaml

#: Environment variable the launcher (scripts/fes-run.sh) uses to hand every
#: actor process the same run id. Unset means "compute one from the clock".
RUN_ID_ENV = "IMPROV_RUN_ID"

#: A run id looks like ``20260817-1432``. The date half doubles as the parent
#: folder name, so it is parsed out rather than re-derived from the clock -- a
#: run started at 23:59 must not file its logs under the next day.
_RUN_ID_RE = re.compile(r"^(\d{8})-\d{4}$")

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

# Module-level caches. Actors are spawned (``method: spawn``), so each process
# re-imports this module and fills its own cache -- that is fine, because the
# value being cached comes from the environment, not from the clock.
_run_id = None
_run_folder = None
_output_root = None


def run_id():
    """The identifier shared by every actor in one improv run.

    Returns ``$IMPROV_RUN_ID`` when the launcher set it, otherwise the current
    minute in the same ``%Y%m%d-%H%M`` format the actors used before.
    """
    global _run_id
    if _run_id is None:
        env = os.environ.get(RUN_ID_ENV, "").strip()
        _run_id = env if env else time.strftime("%Y%m%d-%H%M")
    return _run_id


def output_root():
    """``output_path`` from config.yaml -- the root of the predictions tree."""
    global _output_root
    if _output_root is None:
        try:
            with open(_CONFIG_PATH, "r") as fh:
                config = yaml.safe_load(fh) or {}
            _output_root = Path(config["output_path"])
        except Exception:
            # Never let a config problem stop an actor from starting: a run that
            # writes to the fallback is recoverable, one that dies in setup
            # wastes the session.
            _output_root = Path.home() / "predictions"
    return _output_root


def run_folder():
    """``<output_path>/<YYYYMMDD>/<run id>``, created on first use."""
    global _run_folder
    if _run_folder is None:
        rid = run_id()
        match = _RUN_ID_RE.match(rid)
        # An id that doesn't parse still gets a folder -- just filed under
        # today's date rather than one read out of the id.
        date = match.group(1) if match else time.strftime("%Y%m%d")
        _run_folder = output_root() / date / rid
        _run_folder.mkdir(parents=True, exist_ok=True)
    return _run_folder


def log_folder():
    """``<run folder>/logs``, created on first use."""
    folder = run_folder() / "logs"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def video_session_folder(raw_chunks_path="camera_video"):
    """``~/<raw_chunks_path>/<YYYY-MM-DD>/<HHMMSS>`` for this run's raw buffers.

    Note the layout differs from :func:`run_folder` -- dashes in the date and a
    six-digit time -- because that is what is already on disk and what the
    offline converter's output naming parses. Only the *source* of the
    timestamp changes here.

    Each ``VideoSaver`` used to call ``time.strftime("%H%M%S")`` in its own
    process, so the four savers of one run could straddle a second boundary and
    scatter ``camera_0``..``camera_3`` across sibling folders. That is worse
    than it sounds: the batch converter sizes each session folder to decide
    whether it was a real run, so a split session can fall under the threshold
    and be skipped as a debug run. Deriving from the shared run id makes all
    savers agree. Seconds are always ``00`` -- the run id is minute-resolution.
    """
    rid = run_id()
    match = _RUN_ID_RE.match(rid)
    if match:
        day = match.group(1)
        date = f"{day[0:4]}-{day[4:6]}-{day[6:8]}"
        session = rid.split("-")[1] + "00"
    else:
        date = time.strftime("%Y-%m-%d")
        session = time.strftime("%H%M%S")
    folder = Path.home() / raw_chunks_path / date / session
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def get_logger(name, filename, level=logging.INFO, handler_level=None):
    """A logger writing to ``<run folder>/logs/<filename>``.

    Replaces the 14-line block that was copy-pasted into every actor. Safe to
    call more than once for the same logger: a second call will not stack a
    duplicate handler, which matters because spawned actors re-import their
    module in each child process.

    :param name: logger name, normally ``__name__``
    :param filename: bare log file name, e.g. ``"camera_reader.log"``
    :param level: level for the logger itself
    :param handler_level: level for the file handler; defaults to ``level``.
        Some actors log at DEBUG but only want INFO on disk.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    target = log_folder() / filename
    for existing in logger.handlers:
        if getattr(existing, "baseFilename", None) == str(target):
            return logger

    handler = logging.FileHandler(target)
    handler.setLevel(level if handler_level is None else handler_level)
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(handler)
    return logger
