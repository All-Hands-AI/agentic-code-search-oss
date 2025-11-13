import verifiers as vf


def max_tokens_check(
    prompt, completion: vf.types.Messages, answer, state, task, info
) -> float:
    """
    Check if max tokens was exceeded.
    
    Returns 1.0 if max_tokens was NOT exceeded (good), 0.0 if it was exceeded (bad).
    """
    max_tokens_exceeded = state.get("max_tokens_exceeded", False)
    return 0.0 if max_tokens_exceeded else 1.0

