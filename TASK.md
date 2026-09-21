## Mock Streaming Framework for AI Tutor

The goal of this repo is to create a mock streaming framework to create dataset that simulates the behavior of the
conversation between a student and an AI tutor.

The conversation topics come from the **MathVista** dataset (https://mathvista.github.io/,
https://github.com/lupantech/MathVista). Each simulated conversation corresponds to exactly **one question** from
MathVista — the simulated student asks the AI tutor about that question, and the tutor guides the student through it.

We can start from the "utterance" level, where we will simulate the conversation by generating a series of utterances
that represent the dialogue between the student and the AI tutor.

You can use OPENAI models with the API I will provide to generate text utterances. But this framework
should be modular and flexible enough to allow for the integration of other models or even local models or APIs in the
future.

For the system prompt of the simulated tutor and student, put them in a separate place and make them easily
configurable.

This repo is managed by `uv`, if you want to add any package, you can do so by running `uv add <package-name>`. For
example, if you want to add `openai` package, you can run `uv add openai`.

## Code Requirements

1. The code should be modular and well-structured, with clear separation of concerns between different components (e.g.,
   text generation).
2. The code should be well-documented, with clear comments and docstrings explaining the purpose of each function and
   class.
3. The code should be flexible and extensible, allowing for easy integration of new models or APIs in the future.
4. The code should include error handling to manage potential issues that may arise during the generation process (e.g.,
   API errors, network issues).
5. The code should include a simple interface for configuring the system prompts for the simulated tutor and student,
   allowing for easy customization of the conversation style and content.
6. When Claude Code is iterating the project, there is NO NEED to fallback compatibility with the old code. The codebase
   should be readable easily and maintainable, and don't be over-engineered, it is a research project instead of an
   engineering project. But the code readability and maintainability is absolutely important.
