import verifiers as vf


def get_result_tool_call(
    messages: vf.types.Messages,
) -> tuple[vf.types.ChatMessage | None, bool]:
    """
    Get the result tool call from the messages.

    Returns:
        tuple: (result_tool_call, success) where success is True if there's a "Success" response
    """

    # Find the result tool call
    result_tool_call = None
    for message in messages:
        if message.get("role") == "assistant" and message.get("tool_calls"):
            for tool_call in message["tool_calls"]:
                # Check if it's a result tool call
                if (
                    hasattr(tool_call, "function")
                    and tool_call.function.name == "result"
                ):
                    result_tool_call = tool_call
                    break
            if result_tool_call:
                break

    # Check if there's a corresponding tool response with "Success"
    success = False
    if result_tool_call:
        for message in messages:
            if (
                message.get("role") == "tool"
                and message.get("tool_call_id") == result_tool_call.id
            ):
                content = message.get("content", "")
                if content == "Success":
                    success = True
                break

    return result_tool_call, success
