"""Functionality for validating input for automata."""

import ast
import json
from typing import Callable, Dict, List, Tuple


def inspect_input(input: str, requirements: List[str]) -> Dict[str, str]:
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
