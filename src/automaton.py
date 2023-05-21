"""Type definitions for automata."""

import re
from typing import (
    Any,
    Callable,
    Dict,
    NamedTuple,
    Optional,
    Protocol,
    List,
    Tuple,
    Union,
)

from langchain.agents import AgentExecutor, AgentOutputParser, ZeroShotAgent
from langchain.input import print_text
from langchain.schema import AgentFinish
from langchain.tools.base import BaseTool

from src.validation import IOValidator


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


class AutomatonOutputParser(AgentOutputParser):
    """A modified version of Lanchain's MRKL parser to handle when the agent does not specify the correct action and input format."""

    final_answer_action = "Finalize Reply"
    validator: Union[IOValidator, None] = None

    def parse(
        self, text: str, reflection: Optional[str] = None
    ) -> Union[AutomatonAction, AgentFinish]:
        """Parse the output of the automaton."""

        # \s matches against tab/newline/whitespace
        action_regex = r"Sub-Automaton\s*\d*\s*:(.*?)\nInput\s*\d*\s*Requirements\s*\d*\s*:(.*?)\nSub-Automaton\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(action_regex, text, re.DOTALL)
        if not match:
            return AutomatonAction(
                "Think (function 0)",
                "I must determine what Sub-Automaton to delegate to, what its Input Requirements are, and what Sub-Automaton Input to send.",
                text,
                reflection,
            )
        action = match.group(1).strip()
        action_input = match.group(3)
        if self.final_answer_action in action:
            return AgentFinish({"output": action_input}, text)
        return AutomatonAction(
            action, action_input.strip(" ").strip('"').strip("."), text, reflection
        )


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


class AutomatonAgent(ZeroShotAgent):
    """Agent for automata."""

    reflect: Union[Callable[[Any], str], None]
    """Reflect on information relevant to the current step."""

    def _construct_scratchpad(
        self, intermediate_steps: List[Tuple[AutomatonAction, str]]
    ) -> str:
        """Construct the scratchpad that lets the agent continue its thought process."""

        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += f"\n{action.reflection}\n\n{self.llm_prefix}\n{action.log}"
            thoughts += f"\n{self.observation_prefix}{observation}"
            thoughts += "\n\n---Thoughtcycle---\n\nReflection:"
        return thoughts

    def plan(
        self, intermediate_steps: List[Tuple[AutomatonAction, str]], **kwargs: Any
    ) -> Union[AutomatonAction, AgentFinish]:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """

        reflection = self.reflect(intermediate_steps) if self.reflect else None
        print_text(f"\nReflection:\n{reflection}", color="yellow", end="\n\n")
        full_inputs = self.get_full_inputs(intermediate_steps, **kwargs)
        full_inputs[
            "agent_scratchpad"
        ] = f'{full_inputs["agent_scratchpad"]}\n{reflection}\n\n{self.llm_prefix}'
        full_output = self.llm_chain.predict(**full_inputs)
        return self.output_parser.parse(full_output, reflection=reflection)


class AutomatonExecutor(AgentExecutor):
    """Executor for automata."""

    def _take_next_step(
        self,
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        inputs: Dict[str, str],
        intermediate_steps: List[Tuple[AutomatonAction, str]],
    ) -> Union[AgentFinish, List[Tuple[AutomatonAction, str]]]:
        """Take a single step in the thought-action-observation loop.

        Override this to take control of how the agent makes and acts on choices.
        """
        # Call the LLM to see what to do.
        output = self.agent.plan(intermediate_steps, **inputs)
        # If the tool chosen is the finishing tool, then we end and return.
        if isinstance(output, AgentFinish):
            return output
        actions: List[AutomatonAction]
        if isinstance(output, AutomatonAction):
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

