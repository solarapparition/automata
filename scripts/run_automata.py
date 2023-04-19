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
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage
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


def find_model(engine: str) -> BaseLLM:
    """Find the model to use for a given reasoning type."""
    if engine is None:
        return None
    if engine in ["gpt-3.5-turbo", "gpt-4"]:
        return ChatOpenAI(temperature=0, model_name=engine)
    raise ValueError(f"Engine {engine} not supported yet.")


def load_tool_wrapper(
    file_name: str, full_name: str, description: str, engine: str
) -> Automaton:
    """Load a base tool. Supports all tools in the langchain library, as well as Rank 0 automata."""

    llm = find_model(engine)
    # supported_lc_tools = ["llm-math", "Terminal"]
    supported_tools = ["writing_assistant"]

    # if file_name == "Terminal":
    # return Tool(full_name, load_tools(["terminal"])[0].run, description=description)
    # if file_name in supported_lc_tools:
    # return load_tools([full_name], llm)[0]

    template = "You are a helpful assistant who can help generate a variety of content. However, if anyone asks you to access files, or refers to something from a past interaction, you will immediately inform them that the task is not possible."
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    assistant_chain = LLMChain(llm=llm, prompt=chat_prompt)
    if file_name in ["writing_assistant"]:
        return Tool(full_name, assistant_chain.run, description=description)

    raise NotImplementedError(
        f"Unsupported tool name: {file_name}. Only {supported_tools} are supported for now."
    )


def get_role_info(role: str) -> Dict:
    """Get the role info for a given role."""
    return yaml.load(
        Path(f"src/prompts/roles/{role}.yml").read_text(encoding="utf-8"),
        Loader=yaml.FullLoader,
    )


def create_automaton_prompt(
    input_requirements: List[str],
    instructions: List[str],
    role_info: Dict[str, str],
    sub_automata: List[Tool],
) -> PromptTemplate:
    """Put together a prompt for an automaton."""

    input_requirements = "\n".join([f"- {req}" for req in input_requirements])
    # global_imperative = ["- All of your outputs MUST either include `Action:` and `Action Input:`, OR include `Final Answer:`."]
    imperatives = role_info["imperatives"]
    imperatives = "\n".join([f"- {imperative}" for imperative in imperatives])
    imperatives = f"You have several heuristic imperatives, all of equal importance:\n{imperatives}"
    prefix = AUTOMATON_AFFIXES["prefix"].format(
        input_requirements=input_requirements,
        role_description=role_info["description"],
        imperatives=imperatives,
        role_instruction=role_info["instructions"],
        self_instruction=instructions,
    )
    suffix = AUTOMATON_AFFIXES["suffix"]
    prompt = ZeroShotAgent.create_prompt(
        sub_automata,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input", "agent_scratchpad"],
        format_instructions=role_info["output_format"],
    )
    return prompt


def add_run_handling(run: Callable, name: str) -> Callable:
    """Handle errors during execution of a query."""
    preprint = f"\n\n---{name}: Start---"
    postprint = f"\n\n---{name}: End---"

    @functools.wraps(run)
    def wrapper(*args, **kwargs):
        print(preprint)
        try:
            result = run(*args, **kwargs)
            print(postprint)
            return result
        except Exception as error:
            # ignore all errors since delegators should handle automaton failures
            return str(error).replace("Could not parse LLM output: ", "").strip("`")
        except KeyboardInterrupt:
            # manual interruption should escape back to the delegator
            print(postprint)
            return "Sub-automaton took too long to process and was stopped."

    return wrapper


@lru_cache(maxsize=None)
def load_automaton(file_name: str) -> Automaton:
    """Load an automaton from a YAML file."""
    data = yaml.load(
        (Path("automata") / f"{file_name}.yml").read_text(encoding="utf-8"),
        Loader=yaml.FullLoader,
    )
    full_name = f"{data['name']} ({data['role']} {data['rank']})"
    engine = data["engine"]
    description_and_input = (
        data["description"] + f" Input requirements: {data['input_requirements']}"
    )

    if data["rank"] == 0:  # load base tools directly
        return load_tool_wrapper(file_name, full_name, description_and_input, engine)

    llm = find_model(engine)
    sub_automata = data["sub_automata"]
    sub_automata = [load_automaton(name) for name in sub_automata]
    prompt = create_automaton_prompt(
        input_requirements=data["input_requirements"],
        instructions=data["instructions"],
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
        full_name,
        add_run_handling(agent_executor.run, name=full_name),
        description_and_input,
    )
    return automaton


def main():
    quiz_creator = load_automaton("quiz_creator")
    quiz_creator.run(
        "Create a math quiz suitable for a freshman college student, with 10 questions, then write it to a file."
    )


if __name__ == "__main__":
    main()
