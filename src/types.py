"""Type definitions for automata."""

import re
from typing import Callable, Protocol, Union

from langchain.agents import AgentOutputParser
from langchain.schema import AgentAction, AgentFinish


class Automaton(Protocol):
    """Protocol for automata. Uses the same interface as the Langchain `Tool` class."""

    name: str
    """Name of the automata. Viewable to delegators."""
    run: Callable[[str], str]
    """Function that takes in a query and returns a response."""
    description: str
    """Description of the automata. Viewable to delegators."""


class AutomatonOutputParser(AgentOutputParser):
    """A modified version of Lanchain's MRKL parser to handle when the agent does not specify the correct action and input format."""

    final_answer_action = "Finalize Result"

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        """Parse the output of the automaton."""

        # \s matches against tab/newline/whitespace
        action_regex = r"Sub-Automaton\s*\d*\s*:(.*?)\nInput\s*\d*\s*Requirements\s*\d*\s*:(.*?)\nSub-Automaton\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(action_regex, text, re.DOTALL)
        if not match:
            return AgentAction(
                "Think (function 0)",
                "I must examine the Plan and decide on what Sub-Automaton to delegate to, what its Input Requirements are, and what Sub-Automaton Input to send.",
                text,
            )
        action = match.group(1).strip()
        action_input = match.group(3)
        if self.final_answer_action in action:
            return AgentFinish(
                {"output": action_input}, text
            )
        return AgentAction(action, action_input.strip(" ").strip('"'), text)
