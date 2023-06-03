"""Global variables for the project."""

from pathlib import Path
from typing import Dict

import yaml

from automata.resource_metadata import ResourceMetadata

AUTOMATON_AFFIXES: Dict[str, str] = {
    key: val.strip()
    for key, val in yaml.load(
        Path("automata/prompts/automaton.yml").read_text(encoding="utf-8"),
        Loader=yaml.FullLoader,
    ).items()
}

AUTOMATON_DATA_LOC = Path("automaton_data")

resource_metadata = ResourceMetadata("db/resource_metadata.db")
