"""Run a specific automaton and its sub-automata."""

from datetime import datetime
from functools import partial
import json
from pathlib import Path
import sys
from typing import Callable, Union

from langchain import LLMChain
from langchain.agents import (
    load_tools,
)
from langchain.llms import BaseLLM
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

sys.path.append("")

from src.globals import resource_metadata


def save_text(action_input: str, self_name: str, workspace_name: str) -> str:
    """Save a file."""
    try:
        input_json = json.loads(action_input)
        file_name = input_json["file_name"]
        content = input_json["content"]
        description = input_json.get("description", "")
    except (KeyError, json.JSONDecodeError):
        return "Could not parse input. Please provide the input in the following format: {file_name: <file_name>, description: <description>, content: <content>}"
    path: Path = Path(f"workspace/{workspace_name}/{file_name}")
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


def open_notebook(action_input: str, self_name: str, requester: str) -> str:
    """Open a notebook and perform a read or write action on it."""
    try:
        input_json = json.loads(action_input)
    except json.JSONDecodeError:
        return "Could not parse input. Please provide the input in valid JSON format."
    try:
        mode = input_json["mode"]
    except KeyError:
        return 'Could not parse input. Please include the "mode" value in your input.'
    if mode not in ("read", "write"):
        return 'Could not parse input. Please provide a valid "mode" value (either "read" or "write").'
    if mode == "read" and "topic" not in input_json:
        return 'Could not parse input. Please include the "topic" value in your input.'
    if mode == "read":
        return "Your notebook is empty."  # TODO: implement
    if mode == "write" and not all(key in input_json for key in ("topic", "content")):
        return 'Could not parse input. Please include the "topic" and "content" values in your input.'
    if mode == "write":
        note = {
            "topic": input_json["topic"],
            "content": input_json["content"],
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(
            Path(f"automata/{requester}/notebook.jsonl"), "a", encoding="utf-8"
        ) as file:
            file.write(json.dumps(note) + "\n")


def load_function(
    file_name: str,
    data: dict,
    engine: Union[BaseLLM, None],
    requester: Union[str, None] = None,
) -> Callable[[str], str]:
    """Load a function, which are basically wrappers around external functionality (including other agents)."""

    supported_functions = [
        "llm_assistant",
        "think",
        "human",
        "save_text",
        "load_file",
        "view_workspace",
        "finalize",
        "search",
    ]

    full_name = f"{data['name']} ({data['role']} {data['rank']})"

    if file_name == "llm_assistant":
        template = "You are a helpful assistant who can help generate a variety of content. However, if anyone asks you to access files, or refers to something from a past interaction, you will immediately inform them that the task is not possible, and provide no further information."
        system_message_prompt = SystemMessagePromptTemplate.from_template(template)
        human_template = "{text}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        chat_prompt = ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt]
        )
        assistant_chain = LLMChain(llm=engine, prompt=chat_prompt)
        run = assistant_chain.run

    elif file_name == "save_text":
        run = partial(save_text, self_name=full_name, workspace_name=requester)

    elif file_name == "load_file":
        run = partial(load_file, self_name=full_name)

    elif file_name == "view_workspace":
        run = partial(
            view_workspace_files, self_name=full_name, workspace_name=requester
        )

    elif file_name == "think":
        run = lambda thought: f"I must reflect on my next steps. {thought}"

    elif file_name == "human":
        run = load_tools(["human"])[0].run

    elif file_name == "finalize":
        run = (
            lambda _: None
        )  # not meant to actually be run; the finalize action should be caught by the parser first

    elif file_name == "search":
        run = load_tools(["google-serper"], llm=engine)[0].run

    elif file_name == "notebook":
        run = partial(open_notebook, self_name=full_name, requester=requester)

    else:
        raise NotImplementedError(
            f"Unsupported function name: {file_name}. Only {supported_functions} are supported for now."
        )

    return run
