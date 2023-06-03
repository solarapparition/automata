"""Functionality relating to automaton reflection."""


from pathlib import Path
from typing import Union
from .utilities.importing import quick_import
from .types import AutomatonReflector


def load_reflect(automaton_path: Path, name: str) -> Union[AutomatonReflector, None]:
    """Load the reflection function for an automaton."""
    if name is None:
        return None
    if name.endswith(".py"):
        return quick_import(automaton_path / name).reflect
    if name == "default_action_logger":
        raise NotImplementedError
    raise NotImplementedError
