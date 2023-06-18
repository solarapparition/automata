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
        automata_location=Path("demo_automata"),
    )
    automaton.run(
        "Create a quiz having the subject matter of mathematics, and a difficulty at a freshman college level. Include 10 questions in the quiz, then write it to a file called `math_quiz.txt`."
    )


if __name__ == "__main__":
    demo()
