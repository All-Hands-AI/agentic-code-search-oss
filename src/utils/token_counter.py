"""Utility for tokenizing messages using vLLM tokenization endpoint."""

import logging
from typing import Any

import httpx
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from pydantic import BaseModel


logger = logging.getLogger(__name__)


def _serialize_message(msg: Any) -> dict:
    """
    Serialize a message to a JSON-serializable dictionary.
    
    Handles Pydantic models by converting them to dicts recursively.
    """
    if isinstance(msg, BaseModel):
        # Convert Pydantic model to dict, excluding unset fields
        return msg.model_dump(mode="json", exclude_unset=True)
    elif isinstance(msg, dict):
        # Recursively serialize nested structures
        return {
            key: _serialize_message(value) 
            for key, value in msg.items()
        }
    elif isinstance(msg, list):
        return [_serialize_message(item) for item in msg]
    else:
        return msg


def _serialize_messages(messages: list[ChatCompletionMessageParam]) -> list[dict]:
    """Serialize a list of messages to JSON-serializable format."""
    return [_serialize_message(msg) for msg in messages]


async def count_tokens(
    messages: list[ChatCompletionMessageParam],
    model: str = "Qwen/Qwen3-8B",
    base_url: str = "http://localhost:8000",
    timeout: float = 10.0,
) -> dict[str, Any]:
    """
    Count tokens for a list of messages using vLLM's tokenization endpoint.

    Args:
        messages: List of chat messages to tokenize
        model: Model name to use for tokenization
        base_url: Base URL of the vLLM server
        timeout: Request timeout in seconds

    Returns:
        Dictionary containing:
            - count: Total number of tokens
            - max_model_len: Maximum context length for the model
            - tokens: List of token IDs
            - token_strs: List of token strings (if requested)

    Raises:
        httpx.HTTPError: If the request fails
    """
    url = f"{base_url}/tokenize"

    # Serialize messages to handle Pydantic models
    serialized_messages = _serialize_messages(messages)

    payload = {
        "model": model,
        "messages": serialized_messages,
        "add_generation_prompt": True,
        "return_token_strs": False,
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            result = response.json()

            # logger.debug(
            #     f"Tokenized {len(messages)} messages: "
            #     f"{result['count']} tokens (max: {result['max_model_len']})"
            # )

            return result

        except httpx.HTTPError as e:
            logger.error(f"Failed to tokenize messages: {e}")
            raise


async def check_token_limit(
    messages: list[ChatCompletionMessageParam],
    max_tokens: int,
    model: str = "Qwen/Qwen3-8B",
    base_url: str = "http://localhost:8000",
    timeout: float = 10.0,
) -> tuple[bool, int, int]:
    """
    Check if messages exceed the token limit.

    Args:
        messages: List of chat messages to check
        max_tokens: Maximum allowed tokens
        model: Model name to use for tokenization
        base_url: Base URL of the vLLM server
        timeout: Request timeout in seconds

    Returns:
        Tuple of (exceeded, current_count, max_tokens):
            - exceeded: True if token count exceeds max_tokens
            - current_count: Current token count
            - max_tokens: Maximum allowed tokens

    Raises:
        httpx.HTTPError: If the request fails
    """
    result = await count_tokens(
        messages=messages,
        model=model,
        base_url=base_url,
        timeout=timeout,
    )

    current_count = result["count"]
    exceeded = current_count >= max_tokens

    if exceeded:
        logger.warning(
            f"Token limit exceeded: {current_count}/{max_tokens} tokens"
        )

    return exceeded, current_count, max_tokens

