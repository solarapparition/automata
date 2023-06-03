"""Type definitions for automaton components."""

from typing import (
    Callable,
    NamedTuple,
    Protocol,
    Sequence,
    Tuple,
    Union,
)


class Automaton(Protocol):
    """Protocol for automata. Uses the same interface as the Langchain `Tool` class."""

    name: str
    """Name of the automaton. Viewable to requesters."""
    run: Callable[[str], str]
    """Function that takes in a query and returns a response."""
    description: str
    """Description of the automaton. Viewable to requesters."""


class AutomatonAction(NamedTuple):
    """An action for an automaton."""

    tool: str
    tool_input: str
    log: str
    reflection: Union[str, None]


AutomatonStep = Tuple[AutomatonAction, str]
AutomatonReflector = Callable[[Sequence[AutomatonStep], str], Union[str, None]]
