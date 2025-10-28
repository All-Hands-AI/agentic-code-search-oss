# Agentic Code Search OSS

An open-source implementation of a low-latency agent for code localization.

- **Repository:** `https://github.com/All-Hands-AI/agentic-code-search-oss`
- **Slack:** `#agentic-code-search-oss` (All-Hands-AI workspace)

## 1. Problem Statement

LLM-based coding agents are bottlenecked by context retrieval. They are often slow and inefficient at finding the correct files and code snippets to edit in a large repository. This project builds a small, fast, specialized agent to solve the **code localization** problem.

## 2. Objective

The primary goal is to **minimize the latency of code localization**. The secondary goal is to maintain high precision.

Success will be measured by:

- **Latency:** Time to identify target code locations.
- **Precision:** Percentage of identified locations that are correct.
- **Recall:** Percentage of all correct locations that were identified.

## 3. Technical Plan

The approach is to train a small language model using Reinforcement Learning (RL) on a standardized benchmark.

1. **Benchmark Environment:** `SWE-Gym` will be used for training and evaluation, as it provides realistic software engineering tasks with executable environments.

2. **Reward Signal:** The evaluation logic from the `Agentless` project will be used as the "verifiable reward" mechanism. The agent is rewarded for correctly identifying the files and lines that require edits.

3. **RL Framework:** The agent will be trained using an RL framework. `SkyRL` and `AReaL` are the primary candidates.

4. **Model:** A small, efficient language model (e.g., `Qwen3-0.6B`) will be fine-tuned for the localization task to ensure low inference latency.

5. **Tooling Strategy:** The agent will use a set of tools to navigate the codebase. The focus is on:
    - **Diverse Tool Calls:** Implementing and evaluating tools beyond `grep`, such as Abstract Syntax Tree (AST) parsers for structural code analysis.
    - **Parallel Tool Calling:** Architecting the agent to execute multiple search queries simultaneously to reduce the number of sequential steps.

## 4. Workstreams & Next Steps

The project is broken down into the following workstreams:

- **Workstream 1: Evaluation & RL Environment**

  - **Task:** Set up the core training environment by integrating the `Agentless` validator with `SWE-Gym`. This will provide the foundation for an RL training loop.

- **Workstream 2: Tooling**

  - **Task:** Research, implement, and evaluate different tool calls (e.g., AST-based search, advanced regex, semantic search).
  - **Task:** Design and implement an architecture that supports parallel execution of these tools.

- **Workstream 3: Reinforcement Learning**

  - **Task:** Implement and run training loops using a selected RL framework (e.g., `SkyRL`, `AReaL`).
  - **Task:** Experiment with reward shaping and policy optimization to improve agent performance.

- **Future Considerations:**
  - Investigating question-answering tasks using datasets like `CodeSearchNet`.
  - Analyzing successful agent trajectories to improve learning.

## 5. Contribution

This is a community-driven project.

1. Join the `#agentic-code-search-oss` channel on the All-Hands-AI Slack.
2. Check the GitHub Issues for open tasks.
3. Attend the weekly meetings to sync on progress (details in the Slack channel).

## 6. Resources

- **Primary Inspiration:** [Cognition AI's SWE-grep Blog Post](https://cognition.ai/blog/swe-grep)
- **Core Components:**
  - [SWE-Gym (Environment)](https://github.com/SWE-Gym/SWE-Gym)
  - [Agentless (Reward Validator)](https://github.com/OpenAutoCoder/Agentless)
  - [SkyRL (RL Framework)](https://github.com/NovaSky-AI/SkyRL)
  - [AReaL (RL Framework)](https://github.com/inclusionAI/AReaL)
- **Relevant Research & Projects:**
  - [NVIDIA Nemotron-CORTEXA](https://github.com/NVIDIA/Nemotron-CORTEXA)
  - [LocAgent: Graph-Guided LLM Agents](https://arxiv.org/abs/2503.09089)
  - [SWE-Fixer: Open-Source LLMs for Issue Resolution](https://arxiv.org/abs/2501.05040)
- **Datasets:**
  - [CodeSearchNet](https://github.com/github/CodeSearchNet)
  - [SWE-Fixer-Train-110K](https://huggingface.co/datasets/internlm/SWE-Fixer-Train-110K)
- **Training Parallel Tool Calling:**
  - [SWE-Grep](https://cognition.ai/blog/swe-grep): Forcing parallel tool calling during training (8 tools in parallel per step)
  - [LLMCompiler](https://arxiv.org/abs/2405.17438): Using a "compiler" idea to orchestrate parallel tool calling during training; could be an overall kill for just searching tasks.
  - [Divide-Then-Aggregate](https://aclanthology.org/2025.acl-long.1401.pdf): Another similar training method for parallel tool calling.
  - Overall, this space is relatively unexplored.
