import os
import sys
from typing import Callable, List, Protocol, Sequence, Union


sys.path.append("")


from langchain.agents import load_tools, initialize_agent, Tool
from langchain.llms import OpenAI, BaseLLM


from langchain.agents import AgentType

AgentType.ZERO_SHOT_REACT_DESCRIPTION

from langchain.tools import BaseTool


class Automata(Protocol):
    """Protocol for automata. Uses the same interface as the Langchain `Tool` classs."""

    name: str
    """Name of the automata. Viewable to delegators."""
    run: Callable[[str], str]
    """Function that takes in a query and returns a response."""
    description: str
    """Description of the automata. Viewable to delegators."""


def create_automata(
    name: str, description: str, loaded_tools: List[Union[str, BaseTool]], llm: BaseLLM
):
    """Create an automata with a list of tools."""
    loaded_tools: List[BaseTool] = [
        load_tools([tool], llm)[0] if isinstance(tool, str) else tool
        for tool in loaded_tools
    ]
    agent = initialize_agent(
        loaded_tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
    automata = Tool(name, agent.run, description)
    return automata


def main():

    llm = OpenAI(temperature=0)
    calc_bot = Tool(
        "Calculator Bot 1",
        lambda query: "Sorry, I actually have no clue.",
        "I calculate math equations and return their answer.",
    )
    calc_bot_2 = create_automata(
        name="Calculator Bot 2",
        description="I calculate math equations and return their answer.",
        loaded_tools=["llm-math"],
        llm=llm,
    )
    file_keeper_bot = create_automata(
        name="File Keeper Bot",
        description="I read and write files. I need descriptive instructions on what file to read or write to and what contents it needs.",
        loaded_tools=["terminal"],
        llm=llm,
    )
    manager_bot = create_automata(
        name="Boss Robot",
        description="I use other robots to get tasks done.",
        loaded_tools=[calc_bot, calc_bot_2, file_keeper_bot],
        llm=llm,
    )
    manager_bot.run("What's 5 * 5? Write the answer down.")


if __name__ == "__main__":
    main()
