"""Functionality relating to session management."""

import functools
import json
from pathlib import Path
from typing import Callable, Dict, Tuple, Union

from automata.utilities import generate_timestamp_id

def save_event(event: Dict[str, str], automaton_id: str, automata_location: Path, session_id: str):
    """Save an event to the event log of an automaton."""
    log_path = automata_location / f"{automaton_id}/event_log/{session_id}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as file:
        file.write(json.dumps(event) + "\n")


def add_session_handling(
    run: Callable,
    *,
    automaton_id: str,
    automata_location: Path,
    session_id: str,
    full_name: str,
    input_validator: Union[Callable[[str], Tuple[bool, str]], None],
    requester_id: str,
    requester_session_id: str,
) -> Callable:
    """Handle errors and printouts during execution of a query."""
    preprint = f"\n\n---{full_name}: Start---"
    postprint = f"\n\n---{full_name}: End---"

    @functools.wraps(run)
    def wrapper(*args, **kwargs):
        if input_validator:
            valid, error = input_validator(args[0])
        if input_validator and not valid:
            result = error
        else:
            print(preprint)
            try:
                result = run(*args, **kwargs)
            except KeyboardInterrupt:
                # manual interruption should escape back to the requester
                result = f"Sub-automaton `{full_name}` took too long to process and was manually stopped."
            print(postprint)

        event = {
            "requester": requester_id,
            "sub_automaton_name": automaton_id,
            "input": args[0],
            "result": result,
            "timestamp": generate_timestamp_id(),
        }
        save_event(event, automaton_id, automata_location, session_id)
        save_event(event, requester_id, automata_location, requester_session_id)
        return result

    return wrapper
