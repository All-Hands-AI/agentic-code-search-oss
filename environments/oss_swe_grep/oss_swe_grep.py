import verifiers as vf


def calculate(expression: str) -> float:
    """Calculate a mathematical expression."""
    return eval(expression)  # Simplified example


def load_environment(**kwargs):
    """Load and configure the environment."""
    # 1. Load dataset
    dataset = vf.load_example_dataset("gsm8k", split="train")

    # 2. Configure parser
    parser = vf.ThinkParser()

    # 3. Define reward functions -- can automatically reference:
    # - parser, prompt, completion, answer, state , task, info
    def correct_answer(parser, completion, answer):
        response = parser.parse_answer(completion) or ""
        return 1.0 if response.strip() == answer.strip() else 0.0

    # 4. Create rubric
    rubric = vf.Rubric(
        funcs=[correct_answer, parser.get_format_reward_func()],
        weights=[1.0, 0.2],
    )

    # 5. Return configured environment
    return vf.SingleTurnEnv(
        dataset=dataset,
        tools=[calculate],
        system_prompt="Think step-by-step, then give your answer.",
        parser=parser,
        rubric=rubric,
        **kwargs  # Pass through additional arguments
    )
