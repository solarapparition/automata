"""Utilities for importing modules."""

from importlib import import_module
import os
from pathlib import Path
from types import ModuleType

def quick_import(location: Path) -> ModuleType:
    """Import a module directly from a Path."""
    return import_module(str(location.with_suffix("")).replace(os.path.sep, "."))
