"""Run a specific automaton and its sub-automata."""

from pathlib import Path
import sys

sys.path.append("")

from automata.utilities import generate_timestamp_id
from automata.core import load_automaton


def demo():
    session_id = generate_timestamp_id()
    automaton = load_automaton(
        "quiz_creator",
        requester_session_id=session_id,
        requester_id="human_tester",
        automata_location=Path("automata_data"),
    )
    automaton.run(
        "Create a quiz having the subject matter of mathematics, and a difficulty at a freshman college level. Include 10 questions in the quiz, then write it to a file called `math_quiz.txt`."
    )


def test():
    session_id = generate_timestamp_id()
    automaton = load_automaton(
        "quiz_creator",
        requester_session_id=session_id,
        requester_id="human_tester",
        automata_location=Path("automata_data"),
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

    # automaton = load_automaton("automata_keeper", requester="human_tester")
    # print(result := automaton.run("What is the purpose of the 'automata' package?"))

    breakpoint()


if __name__ == "__main__":
    # test()
    demo()
