import logging
import verifiers as vf
from datasets import load_dataset

import src.tools as tools
import src.rewards as rewards
from src.prompts.system_prompt import SYSTEM_PROMPT
from src.utils.get_instance import get_instance_path


logger = logging.getLogger("swe-grep-oss")


class SWEGrepEnv(vf.StatefulToolEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.add_tool(tools.bash, args_to_skip=["cwd"])
        self.add_tool(tools.result)

    # async def is_completed(
    #     self, messages: vf.types.Messages, state: vf.types.State, **kwargs
    # ) -> bool:
    #     max_turns_reached = await self.max_turns_reached(state)
    #     prompt_too_long = await self.prompt_too_long(state)
    #     if max_turns_reached or prompt_too_long:
    #         return True

    #     return False

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


def load_environment(**kwargs):
    """Load and configure the environment."""

    # Load dataset
    dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    dataset = dataset.map(
        lambda row: {
            # we can add metadata related to the dataset row here
            "info": {
                "repo": row["repo"],
                "instance_id": row["instance_id"],
            },
            "prompt": [{"role": "user", "content": row["problem_statement"]}],
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
        ],
        weights=[2.0, 1.0, 1.0, 1.0],
    )

    # Load environment
    return SWEGrepEnv(
        dataset=dataset,
        system_prompt=SYSTEM_PROMPT,
        rubric=rubric,
        max_turns=10,
        **kwargs,  # Pass through additional arguments
    )
