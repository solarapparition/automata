"""Type definitions for automaton components."""

from typing import (
    Callable,
    NamedTuple,
    Sequence,
    Tuple,
    Union,
)


class AutomatonAction(NamedTuple):
    """An action for an automaton."""

    tool: str
    tool_input: str
    log: str
    reflection: Union[str, None]


AutomatonStep = Tuple[AutomatonAction, str]
AutomatonReflector = Callable[[Sequence[AutomatonStep], str], Union[str, None]]
