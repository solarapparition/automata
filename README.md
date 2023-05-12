# Automata

## Introduction
Automata is an attempt at a cognitive architecture that can compose actions from simple autonomous agents into complex, goal-oriented collective behavior.

The core idea behind this architecture is that instead of having a complex central agent managing many commands and sub-agents, or a fixed set of agents with specific roles in a task loop, we modify Langchain agents to be able to call each other as tools, and then establish a hierarchical, rank-based structure to control the direction of the calls:
```
Agent A (Rank 3):
  - Agent B (Rank 2)
    - Tool 1
    - Tool 2
  - Agent C (Rank 2)
    - Tool 1
    - Tool 3
    - Agent D (Rank 1)
      - Tool 4
      - Tool 5
      - Tool 6
```
Agent A can then potentially be included as a callable sub-agent by another agent of higher rank, and so on.

The hope with this architecture is that it allows each individual sub-agent (called an "automaton" in this system) to be predictable and reliable, while still allowing for complex, flexible behavior to emerge from the interactions between automata.

## Key Targeted Features
The project is currently in the very early stages of core architecture construction, but some key targeted features are:
- Simple, YAML-based Specification: agents, referred to here as automata, can be easily specified using a YAML-based format, specifying each automaton's capabilities, knowledge, and other properties without needing to update code.
- Hierarchical Composition: the system employs a hierarchical structure where higher-ranked automata delegate tasks to lower-ranked ones, essentially converting agents to tools. Each automaton can, in turn, become a callable tool by other automata, with the hope that higher-ranked automata will be able to leverage the capabilities of lower-ranked ones to accomplish more complex tasks.
- Modularity: Automata capabilities are highly modular, both within individual automata and between different automata. This should allow automata to be highly configurable without needing heavy coding.
- Flexible Prompt Control: there are multiple levels granularity available to fine-tune prompts, from individual automata, to role groups, to globally.
- Input and Output Validation: LLM-based validation mechanism to ensure reliability when agents communicate with each other. This helps to correct hallucinations and prevent mistakes from cascading through action sequences and reduces the need for manual human correction.
- Learning Mechanisms: automata are able to accumulate knowledge, both automatically via event logs, and through self-directed notes. This knowledge can then be pre-seeded into and/or explicitly queried by the automata.

## Installation
The core capabilities of the system are still being developed, so the currently implemented automata don't have any capabilities beyond existing Langchain agents. If you want to experiment with what's here anyway, do the following:
1. Clone the repo
2. Change directory to the repo
3. [Optional, recommended] Create a virtual environment for the project
4. Run `pip install -r requirements.txt`
5. Run `python scripts/run_automata.py`

This will run the `Quiz Creator` automaton, which creates and saves a quiz to a file.

## Development Goals

### Short Term:
- Establish the core logical structure of each individual automaton.

### Medium Term:
- Create a set of automata that can collectively perform a more complex learning task (such as figuring out the design principles of a code repository) than what simple agents are capable of right now.
- Add ability to easily plug in external autonomous agents (e.g. AutoGPT, BabyAGI, etc.) in the form of an automaton wrapper, to leverage capabilities of other autonomous AI projects and avoid retreading the same ground.

### Long Term:
- Incorporate [@daveshap](https://github.com/daveshap/)'s Heuristic Imperatives into higher-ranked automata.
- Add ability for automata to create other automata.
- Add ability for automata themselves to contribute to this codebase—docs, issues, feature requests, PRs, etc.
- Extend LLM models to be able to use additional sources besides OpenAI API.
