import verifiers as vf

from src.constants import DEFAULT_MAX_TOOL_CALLS


def tool_call_count(
    prompt, completion: vf.types.Messages, answer, state, task, info
) -> float:
    """
    Count the number of tool calls made in the completion.
    
    Returns a float reward from 0.0-max_tool_calls based on tool call count.
    """
    if not isinstance(completion, list):
        return 0.0
    
    count = 0
    for message in completion:
        if isinstance(message, dict) and "tool_calls" in message:
            tool_calls = message.get("tool_calls", [])
            if tool_calls:
                count += len(tool_calls)

    info = state.get("info", {})
    max_tool_calls: int = info.get("max_tool_calls", DEFAULT_MAX_TOOL_CALLS)

    # Map to 0.0-max_tool_calls float scale
    reward = min(max_tool_calls, (float(count) / max_tool_calls) * max_tool_calls)
    
    return reward

