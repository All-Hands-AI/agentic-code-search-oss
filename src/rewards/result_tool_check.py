import verifiers as vf

from src.utils.get_result_tool_call import get_result_tool_call


def result_tool_check(
    prompt, completion: vf.types.Messages, answer, state, task, info
) -> float:
    """
    Check if the result tool call is successful.
    """

    _, success = get_result_tool_call(completion)
    return 1.0 if success else 0.0
