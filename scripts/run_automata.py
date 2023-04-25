"""Run a specific automaton and its sub-automata."""

from functools import lru_cache, partial
import functools
import json
from pathlib import Path
import re
import sys
from typing import Callable, Dict, List, Optional, Protocol, Tuple, Union

from langchain import LLMChain, PromptTemplate
from langchain.agents import (
    ZeroShotAgent,
    Tool,
    AgentExecutor,
    AgentOutputParser,
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
from langchain.schema import AgentAction, AgentFinish
import yaml

sys.path.append("")

from src.globals import AUTOMATON_AFFIXES, resource_metadata


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


def find_model(engine: str) -> BaseLLM:
    """Create the model to use."""
    if engine is None:
        return None
    if engine in ["gpt-3.5-turbo", "gpt-4"]:
        return ChatOpenAI(temperature=0, model_name=engine)
    raise ValueError(f"Engine {engine} not supported yet.")


def save_file(action_input: str, self_name: str, workspace_name: str) -> str:
    """Save a file."""
    try:
        input_json = json.loads(action_input)
        file_name = input_json["file_name"]
        content = input_json["content"]
        description = input_json.get("description", "")
    except (KeyError, json.JSONDecodeError):
        return "Could not parse input. Please provide the input in the following format: {file_name: <file_name>, description: <description>, content: <content>}"
    path: Path = Path("workspace") / workspace_name / file_name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(content), encoding="utf-8")
    resource_metadata.set_description(str(path), description)
    return f"{self_name}: saved file to `{path.relative_to('workspace')}`"


def load_file(action_input: str, self_name: str) -> str:
    """Load a file."""
    try:
        input_json = json.loads(action_input)
        file_name = input_json["file_name"]
        path: Path = Path("workspace") / file_name
    except (KeyError, json.JSONDecodeError):
        return "Could not parse input. Please provide the input in the following format: {file_name: <file_name>}"
    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"{self_name}: file `{file_name}` not found. Please view your workspace to see which files are available, and use the full path given."
    return content


def view_workspace_files(_, self_name: str, workspace_name: str) -> str:
    """View files in a workspace."""
    path: Path = Path("workspace") / workspace_name
    file_info = (
        f"- `{file.relative_to('workspace')}`: {resource_metadata.get_description(str(file))}"
        for file in path.iterdir()
    )
    if not path.exists():
        raise FileNotFoundError(f"Workspace `{workspace_name}` not found.")
    files = "\n".join(file_info)
    return f"{self_name}: files in your workspace:\n{files}"


def load_function(
    file_name: str, data: dict, delegator: Union[str, None] = None
) -> Automaton:
    """Load a function, which uses the same interface as automata but does not make decisions."""

    model = find_model(data["engine"])
    supported_functions = [
        "llm_assistant",
        "think",
        "human",
        "save_file",
        "load_file",
        "view_workspace",
    ]

    full_name = f"{data['name']} ({data['role']} {data['rank']})"
    input_requirements = (
        "\n".join([f"- {req}" for req in data["input_requirements"]])
        if data["input_requirements"]
        else "None"
    )
    description_and_input = (
        data["description"] + f" Input requirements:\n{input_requirements}"
    )

    if file_name == "llm_assistant":
        template = "You are a helpful assistant who can help generate a variety of content. However, if anyone asks you to access files, or refers to something from a past interaction, you will immediately inform them that the task is not possible."
        system_message_prompt = SystemMessagePromptTemplate.from_template(template)
        human_template = "{text}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        chat_prompt = ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt]
        )
        assistant_chain = LLMChain(llm=model, prompt=chat_prompt)
        return Tool(full_name, assistant_chain.run, description=description_and_input)

    if file_name == "save_file":
        return Tool(
            full_name,
            partial(save_file, self_name=full_name, workspace_name=delegator),
            description=description_and_input,
        )

    if file_name == "load_file":
        return Tool(
            full_name,
            partial(load_file, self_name=full_name),
            description=description_and_input,
        )

    if file_name == "view_workspace":
        return Tool(
            full_name,
            partial(
                view_workspace_files, self_name=full_name, workspace_name=delegator
            ),
            description=description_and_input,
        )

    if file_name == "think":
        return Tool(
            full_name,
            lambda thought: f"I haven't done anything yet, and need to carefully consider what to do next. My previous thought was: {thought}",
            description=description_and_input,
        )

    if file_name == "human":
        return Tool(
            full_name, load_tools(["human"])[0].run, description=description_and_input
        )

    raise NotImplementedError(
        f"Unsupported function name: {file_name}. Only {supported_functions} are supported for now."
    )


def get_role_info(role: str) -> Dict:
    """Get the role info for a given role."""
    return yaml.load(
        Path(f"src/prompts/roles/{role}.yml").read_text(encoding="utf-8"),
        Loader=yaml.FullLoader,
    )


def create_automaton_prompt(
    input_requirements: str,
    self_instructions: List[str],
    self_imperatives: List[str],
    role_info: Dict[str, str],
    sub_automata: List[Tool],
) -> PromptTemplate:
    """Put together a prompt for an automaton."""

    imperatives = role_info["imperatives"] + (self_imperatives or [])
    imperatives = "\n".join([f"- {imperative}" for imperative in imperatives])

    instructions = role_info["instructions"] + (self_instructions or [])
    instructions = "\n".join([f"- {instruction}" for instruction in instructions])

    prefix = AUTOMATON_AFFIXES["prefix"].format(
        input_requirements=input_requirements,
        role_description=role_info["description"],
        imperatives=imperatives,
        instructions=instructions,
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


def add_run_handling(
    run: Callable, name: str, suppress_errors: bool = False
) -> Callable:
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
            if not suppress_errors:
                raise error
            # ignore all errors since delegators should handle automaton failures
            return (
                str(error)
                .replace(
                    "Could not parse LLM output: ",
                    "The sub-automaton ran into an error while processing the query. Its last thought was: ",
                )
                .replace("`", "```")
            )
        except KeyboardInterrupt:
            # manual interruption should escape back to the delegator
            print(postprint)
            return "Sub-automaton took too long to process and was stopped."

    return wrapper


@lru_cache(maxsize=None)
def load_automaton(
    file_name: str, delegator: Union[str, None] = None, suppress_errors: bool = False
) -> Automaton:
    """Load an automaton from a YAML file."""

    data = yaml.load(
        (Path("automata") / f"{file_name}.yml").read_text(encoding="utf-8"),
        Loader=yaml.FullLoader,
    )
    full_name = f"{data['name']} ({data['role']} {data['rank']})"
    engine = data["engine"]
    input_requirements = (
        "\n".join([f"- {req}" for req in data["input_requirements"]])
        if data["input_requirements"]
        else "None"
    )
    description_and_input = (
        data["description"] + f" Input requirements:\n{input_requirements}"
    )

    if data["role"] == "function":  # functions are loaded individually
        return load_function(file_name, data, delegator)

    llm = find_model(engine)

    # wrap rest of loader inside a function to delay loading of sub-automata until needed
    def load_and_run(*args, **kwargs) -> str:
        sub_automata = [
            load_automaton(name, delegator=file_name, suppress_errors=True)
            for name in data["sub_automata"]
        ]
        prompt = create_automaton_prompt(
            input_requirements=input_requirements,
            self_instructions=data["instructions"],
            self_imperatives=data["imperatives"],
            role_info=get_role_info(data["role"]),
            sub_automata=sub_automata,
        )
        # print(prompt.format(input="blah", agent_scratchpad={}))
        # breakpoint()
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=ZeroShotAgent(
                llm_chain=llm_chain,
                allowed_tools=[sub_automaton.name for sub_automaton in sub_automata],
                output_parser=AutomatonOutputParser(),
            ),
            tools=sub_automata,
            verbose=True,
            max_iterations=data["rank"] * 10 + 5,
            max_execution_time=data["rank"] * 200 + 60,
        )
        return agent_executor.run(*args, **kwargs)

    automaton = Tool(
        full_name,
        # add_run_handling(agent_executor.run, name=full_name, suppress_errors=suppress_errors),
        add_run_handling(load_and_run, name=full_name, suppress_errors=suppress_errors),
        description_and_input,
    )
    return automaton


def main():
    quiz_creator = load_automaton("quiz_creator")
    quiz_creator.run(
        # "Create a math quiz suitable for a freshman college student, with 10 questions, then write it to a file called `quiz.txt`."
        "Find an existing quiz in your workspace, load it, and figure out how many questions there is in it."
    )


if __name__ == "__main__":
    main()
