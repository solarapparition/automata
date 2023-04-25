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

    final_answer_action = "Final Result:"

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        if self.final_answer_action in text:
            return AgentFinish(
                {"output": text.split(self.final_answer_action)[-1].strip()}, text
            )
        # \s matches against tab/newline/whitespace
        regex = r"Sub-Automaton\s*\d*\s*:(.*?)\nSub-Automaton\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, text, re.DOTALL)
        if not match:
            return AgentAction(
                "Think",
                "I didn't post my output in the correct format. I must adjust my output to match the format in my prompt.",
                text,
            )
            # raise OutputParserException(f"Could not parse LLM output: `{text}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        return AgentAction(action, action_input.strip(" ").strip('"'), text)
