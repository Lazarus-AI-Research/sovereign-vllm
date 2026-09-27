"""How much the agent says in its log, as an administrator sets it for the
whole appliance: errors only, problems too, what each part did, or every
request as well. Kept beside the agent's configuration, so a restarted agent
logs at the level last set."""

from __future__ import annotations

import logging
import os
from pathlib import Path

LEVELS = {"error": logging.ERROR, "warn": logging.WARNING, "info": logging.INFO, "debug": logging.DEBUG}

KEPT_NAME = "agent-log-level"


def kept_path(config_path: Path | None) -> Path | None:
    return config_path.parent / KEPT_NAME if config_path else None


def apply(name: str) -> bool:
    """Sets the level for every logger the agent and its libraries write to."""
    level = LEVELS.get(name)
    if level is None:
        return False
    logging.getLogger().setLevel(level)
    return True


def restore(config_path: Path | None) -> None:
    path = kept_path(config_path)
    if path is None:
        return
    try:
        apply(path.read_text(encoding="utf-8").strip())
    except OSError:
        pass


def keep(config_path: Path | None, name: str) -> None:
    """Written beside the file and then over it, so a crash mid-write keeps
    the last level."""
    path = kept_path(config_path)
    if path is None:
        return
    partial = path.with_name("." + path.name + ".partial")
    partial.write_text(name + "\n", encoding="utf-8")
    os.replace(partial, path)
