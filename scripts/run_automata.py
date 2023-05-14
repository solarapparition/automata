"""Run a specific automaton and its sub-automata."""

from datetime import datetime
from functools import lru_cache
import functools
import json
from pathlib import Path
import sys
from typing import Callable, Dict, List, Tuple, Union

from langchain import LLMChain, PromptTemplate
from langchain.agents import (
    ZeroShotAgent,
    Tool,
    AgentExecutor,
)
import yaml

sys.path.append("")

from src.globals import AUTOMATON_AFFIXES
from src.engines import create_engine
from src.function_loading import load_function
from src.validation import load_input_validator
from src.automaton import (
    Automaton,
    AutomatonOutputParser,
    get_full_name,
    load_automaton_data,
)
from src.utilities.importing import quick_import


def load_background_knowledge(
    automaton_path: Path, knowledge_info: str, request: Union[str, None] = None
) -> str:
    """Load the background knowledge for an automaton."""
    if knowledge_info.endswith(".py"):
        return quick_import(automaton_path / knowledge_info).load(request=request)
    raise NotImplementedError


def get_role_info(role: str) -> Dict:
    """Get the role info for a given role."""
    return yaml.load(
        Path(f"src/prompts/roles/{role}.yml").read_text(encoding="utf-8"),
        Loader=yaml.FullLoader,
    )


def create_automaton_prompt(
    objective: str,
    self_instructions: List[str],
    self_imperatives: List[str],
    role_info: Dict[str, str],
    sub_automata: List[Tool],
    requester: str,
    background_knowledge: Union[str, None],
) -> PromptTemplate:
    """Put together a prompt for an automaton."""

    imperatives = role_info["imperatives"] + (self_imperatives or [])
    imperatives = "\n".join([f"- {imperative}" for imperative in imperatives]) or "N/A"

    instructions = (self_instructions or []) + role_info["instructions"]
    instructions = (
        "\n".join([f"- {instruction}" for instruction in instructions]) or "N/A"
    )

    prefix = AUTOMATON_AFFIXES["prefix"].format(
        # input_requirements=input_requirements,
        role_description=role_info["description"],
        imperatives=imperatives,
        background_knowledge=background_knowledge,
        # instructions=instructions,
    )

    suffix = (
        AUTOMATON_AFFIXES["suffix"]
        .replace("{instructions}", instructions)
        .replace("{objective}", objective)
        .replace("{requester}", get_full_name(requester))
    )
    prompt = ZeroShotAgent.create_prompt(
        sub_automata,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input", "agent_scratchpad"],
        format_instructions=role_info["output_format"],
    )
    return prompt


def add_run_handling(
    run: Callable,
    name: str,
    input_validator: Union[Callable[[str], Tuple[bool, str]], None] = None,
    requester: Union[str, None] = None,
) -> Callable:
    """Handle errors and printouts during execution of a query."""
    preprint = f"\n\n---{name}: Start---"
    postprint = f"\n\n---{name}: End---"

    @functools.wraps(run)
    def wrapper(*args, **kwargs):
        if input_validator:
            valid, error = input_validator(args[0])
        if input_validator and not valid:
            result = error
        else:
            print(preprint)
            try:
                result = run(*args, **kwargs)
            except KeyboardInterrupt:
                # manual interruption should escape back to the requester
                result = f"Sub-automaton `{name}` took too long to process and was manually stopped."
            print(postprint)

        event = {
            "requester": get_full_name(requester),
            "sub_automaton_name": name,
            "input": args[0],
            "result": result,
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        }

        with open(
            Path(f"automata/{requester}/event_log.jsonl"), "a", encoding="utf-8"
        ) as file:
            file.write(json.dumps(event) + "\n")

        return result

    return wrapper


@lru_cache(maxsize=None)
def load_automaton(file_name: str, requester: Union[str, None] = None) -> Automaton:
    """Load an automaton from a YAML file."""

    data = load_automaton_data(file_name)
    full_name = f"{data['name']} ({data['role']} {data['rank']})"
    engine = data["engine"]
    input_requirements = data["input_requirements"]
    input_requirements_prompt = (
        "\n".join([f"- {req}" for req in input_requirements])
        if input_requirements
        else "None"
    )
    description_and_input = (
        data["description"] + f" Input requirements:\n{input_requirements_prompt}"
    )

    input_validator = load_input_validator(
        data["input_validator"], input_requirements, file_name
    )
    engine = create_engine(engine)

    def load_and_run_function(*args, **kwargs) -> str:
        run = load_function(
            file_name,
            data,
            engine,
            requester=requester,
        )
        return run(*args, **kwargs)

    # wrap rest of loader inside a function to delay loading of sub-automata until needed
    def load_and_run_automaton(*args, **kwargs) -> str:
        sub_automata = [
            load_automaton(name, requester=file_name) for name in data["sub_automata"]
        ]
        background_knowledge = (
            load_background_knowledge(
                Path(f"automata/{file_name}"),
                data["background_knowledge"],
                request=args[0],
            )
            if "background_knowledge" in data
            else None
        )

        prompt = create_automaton_prompt(
            objective=data["objective"],
            self_instructions=data["instructions"],
            self_imperatives=data["imperatives"],
            role_info=get_role_info(data["role"]),
            background_knowledge=background_knowledge,
            sub_automata=sub_automata,
            requester=requester,
        )
        # print(prompt.format(input="blah", agent_scratchpad={}))
        # breakpoint()
        llm_chain = LLMChain(llm=engine, prompt=prompt)
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=ZeroShotAgent(
                llm_chain=llm_chain,
                allowed_tools=[sub_automaton.name for sub_automaton in sub_automata],
                output_parser=AutomatonOutputParser(),
            ),
            tools=sub_automata,
            verbose=True,
            max_iterations=None,
            max_execution_time=None,
        )
        return agent_executor.run(*args, **kwargs)

    load_and_run = (
        load_and_run_function if data["role"] == "function" else load_and_run_automaton
    )
    automaton = Tool(
        full_name,
        add_run_handling(
            load_and_run,
            name=full_name,
            requester=requester,
            input_validator=input_validator,
        ),
        description_and_input,
    )
    return automaton


def demo():
    automaton = load_automaton("quiz_creator", requester="human_tester")
    automaton.run(
        "Create a quiz with the following attributes: subject matter is mathematics, and difficult is freshman college level. Include 10 questions in the quiz, then write it to a file called `math_quiz.txt`."
    )


def test():
    automaton = load_automaton("quiz_creator", requester="human_tester")
    result = automaton.run(
        "Create a quiz with the following attributes: subject matter is mathematics, and difficult is freshman college level. Include 10 questions in the quiz, then write it to a file called `math_quiz.txt`."
    )

    # automaton = load_automaton("auto_reflector", requester="user")
    # automaton.run("Learn more about the Automata system, which you are a part of.")

    # "Give me some instructions on washing windows."
    # "What are the steps for building a computer?"
    # "Delete all quizzes in your workspace."
    # "Find an existing quiz in your workspace, load it, and figure out how many questions there is in it."

    # assistant = load_automaton("llm_assistant")
    # result = assistant.run(
    #     "generate a math quiz and save it to a file called `math_quiz.txt`."
    # )

    # automaton = load_automaton("notebook", requester="human_tester")
    # automaton.run('{"mode": "write", "topic": "test_1", "content": "test_1_content"}')
    # print(result := automaton.run('{"mode": "write", "topic": "test_2", "content": "Texas is the Lone Star State."}'))
    # print(result := automaton.run('{"mode": "write", "topic": "test_3", "content": "Chroma is an open source embeddings database."}'))
    # print(result := automaton.run('{"mode": "read", "question": "What is Chroma??"}'))

    # automaton = load_automaton("notebook_tester", requester="human_tester")
    # print(result := automaton.run("write the following test note: 'chroma is an open source embeddings database.'"))
    # print(result := automaton.run("find out from the notebook what Chroma is.'"))

    # automaton = load_automaton("search", requester="human_tester")
    # print(result := automaton.run("What is chain-of-thought prompting?"))

    # automaton = load_automaton("src_keeper", requester="human_tester")
    # print(result := automaton.run("What is the purpose of the 'src' package?"))

    breakpoint()


if __name__ == "__main__":
    # test()
    demo()
