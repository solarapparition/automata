"""Run a specific automaton and its sub-automata."""

from functools import lru_cache
from pathlib import Path
import sys
from typing import Callable, Dict, List, Union

from langchain import LLMChain, PromptTemplate
from langchain.agents import Tool
import yaml

sys.path.append("")

from src.globals import AUTOMATON_AFFIXES
from src.engines import create_engine
from src.function_loading import load_automaton_function
from src.validation import (
    load_input_validator,
    load_output_validator,
    IOValidator,
)
from src.automaton import (
    Automaton,
    AutomatonAgent,
    AutomatonExecutor,
    AutomatonOutputParser,
    AutomatonReflector,
)
from src.loaders import (
    get_full_name,
    load_automaton_data,
)
from src.planners import load_planner
from src.sessions import add_session_handling
from src.utilities import generate_timestamp_id
from src.utilities.importing import quick_import


def load_reflect(automaton_path: Path, name: str) -> Union[AutomatonReflector, None]:
    """Load the reflection function for an automaton."""
    if name is None:
        return None
    if name.endswith(".py"):
        return quick_import(automaton_path / name).reflect
    if name == "default_action_logger":
        raise NotImplementedError
    raise NotImplementedError


def load_background_knowledge(
    automaton_path: Path,
    name: Union[str, None],
) -> Union[Callable[[str], str], None]:
    """Load the background knowledge for an automaton."""
    if name is None:
        return None
    if name.endswith(".py"):
        return quick_import(automaton_path / name).load_background_knowledge
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
    prompt = AutomatonAgent.create_prompt(
        sub_automata,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input", "agent_scratchpad"],
        format_instructions=role_info["output_format"],
    )
    return prompt


@lru_cache(maxsize=None)
def load_automaton(
    automaton_id: str, requester_session_id: str, requester_id: str
) -> Automaton:
    """Load an automaton from a YAML file."""

    data = load_automaton_data(automaton_id)
    automaton_location = Path(f"automata/{automaton_id}")
    full_name = f"{data['name']} ({data['role']} {data['rank']})"
    engine = data["engine"]
    engine = create_engine(engine)

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
        data["input_validator"], input_requirements, automaton_id
    )

    def run_function(*args, **kwargs) -> str:
        run = load_automaton_function(
            automaton_id,
            data,
            engine,
            requester_id=requester_id,
        )
        return run(*args, **kwargs)

    self_session_id = generate_timestamp_id()

    # lazy load sub-automata until needed
    def run_core_automaton(*args, **kwargs) -> str:
        request = args[0]
        output_validator: Union[IOValidator, None] = load_output_validator(
            data["output_validator"], request=request, file_name=automaton_id
        )
        reflect: Union[Callable, None] = load_reflect(
            Path(f"automata/{automaton_id}"), data["reflect"]
        )
        planner = load_planner(automaton_location, data["planner"])
        sub_automata = [
            load_automaton(
                sub_automata_id,
                requester_session_id=self_session_id,
                requester_id=automaton_id,
            )
            for sub_automata_id in data["sub_automata"]
        ]
        create_background_knowledge = load_background_knowledge(
            automaton_location,
            data["knowledge"],
        )
        background_knowledge = (
            create_background_knowledge(args[0])
            if create_background_knowledge
            else None
        )
        prompt = create_automaton_prompt(
            objective=data["objective"],
            self_instructions=data["instructions"],
            self_imperatives=data["imperatives"],
            role_info=get_role_info(data["role"]),
            background_knowledge=background_knowledge,
            sub_automata=sub_automata,
            requester=requester_id,
        )
        # print(prompt.format(input="blah", agent_scratchpad={}))
        # breakpoint()
        agent_executor = AutomatonExecutor.from_agent_and_tools(
            agent=AutomatonAgent(
                # llm_chain=llm_chain,
                llm_chain=LLMChain(llm=engine, prompt=prompt),
                allowed_tools=[sub_automaton.name for sub_automaton in sub_automata],
                output_parser=AutomatonOutputParser(validate=output_validator),
                reflect=reflect,
                planner=planner,
            ),
            tools=sub_automata,
            verbose=True,
            max_iterations=None,
            max_execution_time=None,
        )
        return agent_executor.run(*args, **kwargs)

    run_mapping = {
        "default_function_runner": run_function,
        "default_automaton_runner": run_core_automaton,
    }
    run = run_mapping[data["runner"]]

    automaton = Tool(
        full_name,
        add_session_handling(
            run,
            automaton_id=automaton_id,
            session_id=self_session_id,
            full_name=full_name,
            requester_id=requester_id,
            input_validator=input_validator,
            requester_session_id=requester_session_id,
        ),
        description_and_input,
    )
    return automaton


def demo():
    session_id = generate_timestamp_id()
    automaton = load_automaton(
        "quiz_creator", requester_session_id=session_id, requester_id="human_tester"
    )
    automaton.run(
        "Create a quiz having the subject matter of mathematics, and a difficulty at a freshman college level. Include 10 questions in the quiz, then write it to a file called `math_quiz.txt`."
    )


def test():
    session_id = generate_timestamp_id()
    automaton = load_automaton(
        "quiz_creator", requester_session_id=session_id, requester_id="human_tester"
    )
    result = automaton.run(
        "Create a quiz having the subject matter of mathematics, and a difficulty at a freshman college level. Include 10 questions in the quiz, then write it to a file called `math_quiz.txt`."
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
