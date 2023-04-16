"""Run a specific automaton and its sub-automata."""

from functools import lru_cache
import functools
from pathlib import Path
import sys
from typing import Callable, Dict, List, Protocol, Union

from langchain import LLMChain, PromptTemplate
from langchain.agents import (
    ZeroShotAgent,
    Tool,
    AgentExecutor,
    load_tools,
    initialize_agent,
    Tool,
    AgentType,
)
from langchain.chat_models import ChatOpenAI
from langchain.llms import BaseLLM, OpenAI
from langchain.tools import BaseTool
import yaml

sys.path.append("")

from src.globals import AUTOMATON_AFFIXES


class Automaton(Protocol):
    """Protocol for automata. Uses the same interface as the Langchain `Tool` class."""

    name: str
    """Name of the automata. Viewable to delegators."""
    run: Callable[[str], str]
    """Function that takes in a query and returns a response."""
    description: str
    """Description of the automata. Viewable to delegators."""


def create_automaton(
    name: str, description: str, loaded_tools: List[Union[str, BaseTool]], llm: BaseLLM
):
    """Create an automaton with a list of tools."""
    loaded_tools: List[BaseTool] = [
        load_tools([tool], llm)[0] if isinstance(tool, str) else tool
        for tool in loaded_tools
    ]
    agent = initialize_agent(
        loaded_tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
        # loaded_tools, llm, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
    automata = Tool(name, agent.run, description)
    return automata


def find_model(reasoning_type: str) -> BaseLLM:
    """Find the model to use for a given reasoning type."""
    if reasoning_type == "fast":
        return ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    if reasoning_type == "careful":
        return ChatOpenAI(temperature=0, model_name="gpt-4")
    return ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")


from langchain.prompts.chat import ChatPromptTemplate


def load_tool_wrapper(name: str, description: str, reasoning_type: str) -> Automaton:
    """Load a base tool. Supports all tools in the langchain library, as well as Rank 0 automata."""
    from langchain.chat_models import ChatOpenAI
    from langchain import PromptTemplate, LLMChain
    from langchain.prompts.chat import (
        ChatPromptTemplate,
        SystemMessagePromptTemplate,
        AIMessagePromptTemplate,
        HumanMessagePromptTemplate,
    )
    from langchain.schema import AIMessage, HumanMessage, SystemMessage

    llm = find_model(reasoning_type)
    supported_lc_tools = ["llm-math", "terminal"]
    custom_tools = ["Writing Generator", "Analysis Assistant", "Coding Assistant"]

    if name == "Terminal":
        return Tool(name, load_tools(["terminal"])[0].run, description=description)
    if name in supported_lc_tools:
        return load_tools([name], llm)[0]

    template = "You are a helpful assistant who can help generate a variety of content."
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    chain = LLMChain(llm=llm, prompt=chat_prompt)
    if name in supported_lc_tools:
        return load_tools([name], llm)[0]
    if name in ["Writing Generator", "Coding Assistant"]:
        return Tool(name, chain.run, description=description)

    raise NotImplementedError(
        f"Unsupported tool name: {name}. Only {supported_lc_tools + custom_tools} are supported for now."
    )


def get_role_info(role: str) -> Dict:
    """Get the role info for a given role."""
    return yaml.load(
        Path(f"src/prompts/roles/{role}.yml").read_text(encoding="utf-8"),
        Loader=yaml.FullLoader,
    )


def create_automaton_prompt(
    name: str,
    description: str,
    rank: int,
    input_requirements: str,
    self_imperatives: List[str],
    role_info: Dict[str, str],
    sub_automata: List[Tool],
) -> PromptTemplate:
    """Put together a prompt for an automaton."""
    # global_imperative = ["- All of your outputs MUST either include `Action:` and `Action Input:`, OR include `Final Answer:`."]
    imperatives = role_info["imperatives"] + self_imperatives
    imperatives = "- " + "\n- ".join(imperatives)
    imperatives = f"You have several heuristic imperatives, all of equal importance:\n{imperatives}"
    prefix = AUTOMATON_AFFIXES["prefix"].format(
        # name=name,
        # description=description,
        # rank=rank,
        input_requirements=input_requirements,
        role_description=role_info["description"],
        role_instruction=role_info["instruction"],
        imperatives=imperatives,
    )
    prompt = ZeroShotAgent.create_prompt(
        sub_automata,
        prefix=prefix,
        suffix=AUTOMATON_AFFIXES["suffix"],  # .replace(
        # "{imperatives}", imperatives),
        input_variables=["input", "agent_scratchpad"],
        format_instructions=role_info["output_format"],
    )
    return prompt


def add_handling(run: Callable, preprint: str, postprint: str) -> Callable:
    """Handle errors during execution of a query."""

    @functools.wraps(run)
    def wrapper(*args, **kwargs):
        print(preprint)
        try:
            result = run(*args, **kwargs)
            print(postprint)
            return result
        except Exception as error:
            # ignore errors since delegators should handle automaton failures
            return str(error).replace("Could not parse LLM output: ", "").strip("`")
        except KeyboardInterrupt:
            # manual interruption should take process to the delegator
            print(postprint)
            return "Sub-automaton was interrupted."

    return wrapper


@lru_cache(maxsize=None)
def load_automaton(name: str) -> Automaton:
    """Load an automaton from a YAML file."""
    data = yaml.load(
        (Path("automata") / f"{name}.yml").read_text(encoding="utf-8"),
        Loader=yaml.FullLoader,
    )
    name = data["name"]
    reasoning_type = data["reasoning_type"]
    description_and_input = (
        data["description"] + f" Input requirements: {data['input_requirements']}"
    )

    if data["rank"] == 0:  # load base tools directly
        return load_tool_wrapper(name, description_and_input, reasoning_type)

    llm = find_model(reasoning_type)
    sub_automata = data["sub_automata"]
    sub_automata = [load_automaton(name) for name in sub_automata]
    prompt = create_automaton_prompt(
        name=name,
        description=description_and_input,
        rank=data["rank"],
        input_requirements=data["input_requirements"],
        self_imperatives=data["imperatives"],
        role_info=get_role_info(data["role"]),
        sub_automata=sub_automata,
    )
    # print(prompt.format(input="blah", agent_scratchpad={}))
    # breakpoint()
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    agent = ZeroShotAgent(
        llm_chain=llm_chain,
        allowed_tools=[sub_automaton.name for sub_automaton in sub_automata],
    )
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=sub_automata,
        verbose=True,
        max_iterations=data["rank"] * 10 + 5,
        max_execution_time=data["rank"] * 200 + 60,
    )
    automaton = Tool(
        name,
        add_handling(agent_executor.run, f"\n\n---{name}---"),
        description_and_input,
    )
    return automaton


def main():
    quiz_creator = load_automaton("quiz_creator")
    quiz_creator.run(
        "Create a math quiz suitable for a freshman college student, with 10 questions, and include the answer key. Write the quiz to a file called `quiz.txt`, and verify that the answers in the quiz are correct, and that all the questions were generated."
    )


if __name__ == "__main__":
    main()
