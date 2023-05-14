"""Type definitions for automata."""

from functools import lru_cache
from pathlib import Path
import re
from typing import Callable, Dict, Protocol, Union

from langchain.agents import AgentOutputParser
from langchain.schema import AgentAction, AgentFinish
import yaml


class Automaton(Protocol):
    """Protocol for automata. Uses the same interface as the Langchain `Tool` class."""

    name: str
    """Name of the automata. Viewable to requesters."""
    run: Callable[[str], str]
    """Function that takes in a query and returns a response."""
    description: str
    """Description of the automata. Viewable to requesters."""


class AutomatonOutputParser(AgentOutputParser):
    """A modified version of Lanchain's MRKL parser to handle when the agent does not specify the correct action and input format."""

    final_answer_action = "Finalize Reply"

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        """Parse the output of the automaton."""

        # \s matches against tab/newline/whitespace
        action_regex = r"Sub-Automaton\s*\d*\s*:(.*?)\nInput\s*\d*\s*Requirements\s*\d*\s*:(.*?)\nSub-Automaton\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(action_regex, text, re.DOTALL)
        if not match:
            return AgentAction(
                "Think (function 0)",
                "I must determine what Sub-Automaton to delegate to, what its Input Requirements are, and what Sub-Automaton Input to send.",
                text,
            )
        action = match.group(1).strip()
        action_input = match.group(3)
        if self.final_answer_action in action:
            return AgentFinish({"output": action_input}, text)
        return AgentAction(action, action_input.strip(" ").strip('"').strip("."), text)


@lru_cache
def load_automaton_data(file_name: str) -> Dict:
    """Load an automaton from a YAML file."""
    automaton_path = Path(f"automata/{file_name}")
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
