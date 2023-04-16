"""Global variables for the project."""

from pathlib import Path
from typing import Dict

import yaml


AUTOMATON_AFFIXES: Dict[str, str] = {
    key: val.strip()
    for key, val in yaml.load(
        Path("src/prompts/automaton.yml").read_text(encoding="utf-8"),
        Loader=yaml.FullLoader,
    ).items()
}
