"""Load automaton info from source files."""

from functools import lru_cache
from typing import Dict

import yaml

from automata.config import AUTOMATON_DATA_LOC


@lru_cache
def load_automaton_data(file_name: str) -> Dict:
    """Load an automaton from a YAML file."""
    automaton_path = AUTOMATON_DATA_LOC / file_name
    data = yaml.load(
        (automaton_path / "spec.yml").read_text(encoding="utf-8"),
        Loader=yaml.FullLoader,
    )
    return data


def get_full_name(file_name: str) -> str:
    """Get the full name of an automaton."""
    try:
        data = load_automaton_data(file_name)
    except FileNotFoundError:
        return file_name
    return f"{data['name']} ({data['role']} {data['rank']})"
