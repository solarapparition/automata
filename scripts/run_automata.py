"""Run a specific automaton and its sub-automata."""

import ast
from datetime import datetime
from functools import lru_cache, partial
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
from langchain.chat_models import ChatOpenAI
from langchain.llms import BaseLLM
import yaml

sys.path.append("")

from src.automaton_functions import load_function
from src.globals import AUTOMATON_AFFIXES
from src.llm_function import make_llm_function
from src.types import Automaton, AutomatonOutputParser


def create_engine(engine: str) -> BaseLLM:
    """Create the model to use."""
    if engine is None:
        return None
    if engine in ["gpt-3.5-turbo", "gpt-4"]:
        return ChatOpenAI(temperature=0, model_name=engine)
    raise ValueError(f"Engine {engine} not supported yet.")


def get_role_info(role: str) -> Dict:
    """Get the role info for a given role."""
    return yaml.load(
        Path(f"src/prompts/roles/{role}.yml").read_text(encoding="utf-8"),
        Loader=yaml.FullLoader,
    )


def create_automaton_prompt(
    # input_requirements: str,
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
        # input_requirements=input_requirements,
        role_description=role_info["description"],
        imperatives=imperatives,
        # instructions=instructions,
    )
    suffix = AUTOMATON_AFFIXES["suffix"].replace("{instructions}", instructions)
    prompt = ZeroShotAgent.create_prompt(
        sub_automata,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input", "agent_scratchpad"],
        format_instructions=role_info["output_format"],
    )
    return prompt


def inspect_input_specs(input: str, requirements: List[str]) -> Dict[str, str]:
    """
    Validate whether the input for a function adheres to the requirements of the function.

    This function checks the given input string against a list of requirements and returns a dictionary. The dictionary
    contains two keys: "success" and "message". The "success" key has a boolean value indicating whether the input
    meets all the requirements, and the "message" key has a string value with an error message if the input does not
    meet a requirement, or an empty string if the input meets all the requirements.

    :param input: The input string to be validated.
    :type input: str
    :param requirements: A list of requirements that the input string should meet.
    :type requirements: List[str]
    :return: A dictionary containing a boolean value indicating whether the input meets all requirements and an error message.
    :rtype: Dict[str, str]

    Examples:
    >>> inspect_input("1 + 1", ["A math expression"])
    {"success": true, "message": ""}
    >>> inspect_input("x = 5", ["A math expression", "Contains a variable", "Variable is named 'y'"])
    {"success": false, "message": "The input does not have a variable named 'y'."}
    >>> inspect_input("def example_function():", ["A function definition", "Named 'example_function'"])
    {"success": true, "message": ""}
    >>> inspect_input("x + y * z", ["A math expression", "Uses all basic arithmetic operations"])
    {"success": false, "message": "The input does not use all basic arithmetic operations."}
    >>> inspect_input("How are you?", ["A question", "About well-being"])
    {"success": true, "message": ""}
    >>> inspect_input("The quick brown fox jumps over the lazy dog.", ["A sentence", "Contains all English letters"])
    {"success": true, "message": ""}
    >>> inspect_input("Once upon a time...", ["A narrative", "Begins with a common opening phrase"])
    {"success": true, "message": ""}
    >>> inspect_input("How old are you?", ["A question", "About age", "Uses the word 'years'"])
    {"success": false, "message": "The input does not use the word 'years'."}
    >>> inspect_input("The sun sets in the east.", ["A statement", "Describes a natural phenomenon", "Factually accurate"])
    {"success": false, "message": "The input is not factually accurate."}
    >>> inspect_input("Are you going to the party tonight?", ["A question", "About attending an event", "Mentions a specific person"])
    {"success": false, "message": "The input does not mention a specific person."}
    >>> inspect_input("I prefer dogs over cats.", ["A preference", "Involves animals", "Prefers cats"])
    {"success": false, "message": "The input expresses a preference for dogs, not cats."}
    """
    ...


def validate_input(
    run_input: str, input_inspector: Callable[[str], str], full_name: str
) -> Tuple[bool, str]:
    """Validate input against input requirements, using an input inspector. The input inspector is intended to be powered by an LLM."""
    expected_output_keys = ["success", "message"]
    output = input_inspector(run_input)

    try:
        output = json.loads(output)
    except json.JSONDecodeError:
        output = ast.literal_eval(
            output.replace("true", "True").replace("false", "False")
        )
    except Exception as error:
        raise ValueError("Input inspector output is not a valid dictionary.") from error
    try:
        if output["success"]:
            return True, ""
        return (
            output["success"],
            f"{full_name}: {output['message']} Please check the input requirements of this automaton and try again.",
        )
    except KeyError as error:
        raise ValueError(
            f"Input inspector output does not have the correct format. Expected keys: {expected_output_keys}"
        ) from error


def add_run_handling(
    run: Callable,
    name: str,
    input_validator: Union[Callable[[str], Tuple[bool, str]], None] = None,
    delegator: Union[str, None] = None,
) -> Callable:
    """Handle errors and printouts during execution of a query."""
    preprint = f"\n\n---{name}: Start---"
    postprint = f"\n\n---{name}: End---"

    @functools.wraps(run)
    def wrapper(*args, **kwargs):
        if input_validator:
            valid, error = input_validator(args[0])
            if not valid:
                result = error
        print(preprint)
        try:
            result = run(*args, **kwargs)
        except KeyboardInterrupt:
            # manual interruption should escape back to the delegator
            result = f"Sub-automaton `{name}` took too long to process and was manually stopped."
        print(postprint)

        event = {
            "delegator": delegator,
            "sub_automaton_name": name,
            "input": args[0],
            "result": result,
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        }

        with open(
            Path(f"automata/{delegator}/event_log.jsonl"), "a", encoding="utf-8"
        ) as file:
            file.write(json.dumps(event) + "\n")

        return result

    return wrapper


@lru_cache(maxsize=None)
def load_automaton(file_name: str, delegator: Union[str, None] = None) -> Automaton:
    """Load an automaton from a YAML file."""

    data = yaml.load(
        Path(f"automata/{file_name}/spec.yml").read_text(encoding="utf-8"),
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

    # create input validation
    validator_engine = data["input_validator_engine"]
    if validator_engine:
        inspect_input = make_llm_function(
            inspect_input_specs, model=create_engine(validator_engine)
        )
        inspect_input = partial(
            inspect_input, input_requirements=data["input_requirements"]
        )

    input_validator = (
        partial(validate_input, input_inspector=inspect_input, full_name=full_name)
        if validator_engine
        else None
    )

    engine = create_engine(engine)

    def load_and_run_function(*args, **kwargs) -> str:
        run = load_function(
            file_name,
            data,
            engine,
            delegator=delegator,
        )
        return run(*args, **kwargs)

    # wrap rest of loader inside a function to delay loading of sub-automata until needed
    def load_and_run_automaton(*args, **kwargs) -> str:
        sub_automata = [
            load_automaton(name, delegator=file_name) for name in data["sub_automata"]
        ]
        prompt = create_automaton_prompt(
            self_instructions=data["instructions"],
            self_imperatives=data["imperatives"],
            role_info=get_role_info(data["role"]),
            sub_automata=sub_automata,
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
            delegator=delegator,
            input_validator=input_validator,
        ),
        description_and_input,
    )
    return automaton


def main():
    quiz_creator = load_automaton("quiz_creator")
    quiz_creator.run(
        "Create a math quiz suitable for a freshman college student, with 10 questions, then write it to a file called `math_quiz.txt`."
        # "Find an existing quiz in your workspace, load it, and figure out how many questions there is in it."
    )

    # assistant = load_automaton("llm_assistant")
    # result = assistant.run(
    #     "generate a math quiz and save it to a file called `math_quiz.txt`."
    # )
    breakpoint()


if __name__ == "__main__":
    main()
