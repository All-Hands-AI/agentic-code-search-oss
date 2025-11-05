import datetime
import json
import logging
import verifiers as vf
from datasets import load_dataset
import traceback
import src.tools as tools
import src.rewards as rewards
from src.constants import DEFAULT_MAX_TOKENS, DEFAULT_MAX_TOOL_CALLS
from src.prompts.system_prompt import SYSTEM_PROMPT
from src.utils.get_instance import get_instance_path
from src.utils.tokenize import check_token_limit
from src.utils.get_result_tool_call import get_result_tool_call


logger = logging.getLogger("swe-grep-oss")


class SWEGrepEnv(vf.StatefulToolEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.add_tool(tools.bash, args_to_skip=["cwd"])
        self.add_tool(tools.result)

    async def is_completed(
        self, messages: vf.types.Messages, state: vf.types.State, **kwargs
    ) -> bool:
        max_turns_reached = await self.max_turns_reached(state)
        prompt_too_long = await self.prompt_too_long(state)
        
        # Check if we've exceeded the token limit
        max_tokens_exceeded = False
        try:
            if not (isinstance(messages, list) and "info" in state and "max_tokens" in state["info"]):
                raise ValueError("Messages or state is not valid")
            
            exceeded, current_count, max_allowed = await check_token_limit(
                messages=messages,
                max_tokens=state["info"]["max_tokens"],
                model=kwargs.get("model", "Qwen/Qwen3-8B"),
                base_url=kwargs.get("base_url", "http://localhost:8000"),
            )
            max_tokens_exceeded = exceeded
        except Exception as e:
            self.logger.warning(f"Failed to check token limit: {e}")

            self.logger.warning(f"\nMessages: {messages}")
            self.logger.warning(f"\nState: {state}")

            # save messages and state to a file
            with open(f"messages-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.json", "w") as f:
                json.dump({"messages": messages, "state": state}, f, indent=2)
            
            raise e
        
        # Store max_tokens_exceeded in state for reward function
        state["max_tokens_exceeded"] = max_tokens_exceeded

        # Check if the result tool call succeeded
        _, result_tool_call_success = get_result_tool_call(messages)
        
        if max_tokens_exceeded or result_tool_call_success or max_turns_reached or prompt_too_long:
            return True
        
        return False

    async def env_response(
        self, messages: vf.types.Messages, state: vf.types.State, **kwargs
    ) -> tuple[vf.types.Messages, vf.types.State]:
        assert isinstance(messages, list)

        tool_calls = messages[-1].get("tool_calls", [])
        tool_messages = []
        for tool_call in tool_calls:
            assert isinstance(tool_call, vf.types.ChatCompletionMessageToolCall)
            tool_name: str = tool_call.function.name
            tool_args: dict = json.loads(tool_call.function.arguments)
            tool_call_id: str = tool_call.id or ""
            tool_args = self.update_tool_args(
                tool_name, tool_args, messages, state, **kwargs
            )
            tool_message: vf.types.Message = await self.call_tool(
                tool_name, tool_args, tool_call_id
            )
            tool_messages.append(tool_message)
        return tool_messages, state


    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict,
        messages: vf.types.Messages,
        state: vf.types.State,
        **kwargs,
    ) -> dict:
        if tool_name == "bash":
            repo_path = get_instance_path(
                {
                    "repo": state["info"]["repo"],
                    "instance_id": state["info"]["instance_id"],
                }
            )
            updated_tool_args = dict(tool_args)
            updated_tool_args["cwd"] = repo_path
            return updated_tool_args

        return tool_args


def load_environment(max_tokens: int = DEFAULT_MAX_TOKENS, max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS, **kwargs):
    """Load and configure the environment."""

    # Load dataset
    dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    dataset = dataset.map(
        lambda row: {
            # we can add metadata related to the dataset row here
            "info": {
                "repo": row["repo"],
                "instance_id": row["instance_id"],
                "max_tokens": max_tokens,
                "max_tool_calls": max_tool_calls,
            },
            "prompt": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": row["problem_statement"]}],
            "answer": row["patch"],
        }
    )

    # Define rubric
    rubric = vf.Rubric(
        funcs=[
            rewards.result_tool_check,
            rewards.result_tool_f1,
            rewards.result_tool_precision,
            rewards.result_tool_recall,
            rewards.tool_call_count,
            rewards.max_tokens_check,
        ],
        weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    )

    # Load environment
    return SWEGrepEnv(
        dataset=dataset,
        rubric=rubric,
        max_turns=8,
        **kwargs,  # Pass through additional arguments
    )
