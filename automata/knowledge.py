"""Functions relating to the knowledge module for automata."""

from pathlib import Path
from typing import Callable, Union
from .utilities.importing import quick_import


def load_knowledge(
    automaton_path: Path,
    name: Union[str, None],
) -> Union[Callable[[str], str], None]:
    """Load the background knowledge for an automaton."""
    if name is None:
        return None
    if name.endswith(".py"):
        return quick_import(automaton_path / name).load
    raise NotImplementedError
