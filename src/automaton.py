"""Type definitions for automata."""

from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, Protocol, List, Tuple, Union

from langchain.agents import AgentExecutor
from langchain.agents.tools import InvalidTool
from langchain.schema import (
    AgentAction,
    AgentFinish,
)
from langchain.tools.base import BaseTool
import yaml


class Automaton(Protocol):
    """Protocol for automata. Uses the same interface as the Langchain `Tool` class."""

    name: str
    """Name of the automaton. Viewable to requesters."""
    run: Callable[[str], str]
    """Function that takes in a query and returns a response."""
    description: str
    """Description of the automaton. Viewable to requesters."""


class InvalidSubAutomaton(BaseTool):
    """Exception raised when a sub-automaton is invalid."""

    sub_automata_allowed: List[str] = []
    name = "Invalid Sub-Automaton"
    description = "Called when sub-automaton name is invalid."

    def _run(self, tool_input: str) -> str:
        """Use the tool."""
        return f"{tool_input} is not a valid Sub-Automaton, try another one from the Sub-Automata list: {self.sub_automata_allowed}"

    async def _arun(self, tool_input: str) -> str:
        """Use the tool."""
        return self._run(tool_input)


class AutomatonExecutor(AgentExecutor):
    def _take_next_step(
        self,
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        inputs: Dict[str, str],
        intermediate_steps: List[Tuple[AgentAction, str]],
    ) -> Union[AgentFinish, List[Tuple[AgentAction, str]]]:
        """Take a single step in the thought-action-observation loop.

        Override this to take control of how the agent makes and acts on choices.
        """
        # Call the LLM to see what to do.
        output = self.agent.plan(intermediate_steps, **inputs)
        # If the tool chosen is the finishing tool, then we end and return.
        if isinstance(output, AgentFinish):
            return output
        actions: List[AgentAction]
        if isinstance(output, AgentAction):
            actions = [output]
        else:
            actions = output
        result = []
        for agent_action in actions:
            self.callback_manager.on_agent_action(
                agent_action, verbose=self.verbose, color="green"
            )
            # Otherwise we lookup the tool
            if agent_action.tool in name_to_tool_map:
                tool = name_to_tool_map[agent_action.tool]
                return_direct = tool.return_direct
                color = color_mapping[agent_action.tool]
                tool_run_kwargs = self.agent.tool_run_logging_kwargs()
                if return_direct:
                    tool_run_kwargs["llm_prefix"] = ""
                # We then call the tool on the tool input to get an observation
                observation = tool.run(
                    agent_action.tool_input,
                    verbose=self.verbose,
                    color=color,
                    **tool_run_kwargs,
                )
            else:
                tool_run_kwargs = self.agent.tool_run_logging_kwargs()
                observation = InvalidSubAutomaton(
                    sub_automata_allowed=name_to_tool_map.keys()
                ).run(
                    agent_action.tool,
                    verbose=self.verbose,
                    color="red",
                    **tool_run_kwargs,
                )
            result.append((agent_action, observation))
        return result


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
