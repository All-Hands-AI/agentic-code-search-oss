import logging
import textwrap
import verifiers as vf
from datasets import load_dataset

import src.tools as tools
import src.rewards as rewards
from src.prompts.system_prompt import SYSTEM_PROMPT
from src.utils.get_instance import get_instance_path
from src.environments.sandbox_env import SandboxEnv


logger = logging.getLogger("swe-grep-oss")


class SWEGrepEnv(SandboxEnv):
    _REPO_URL = "https://github.com/astropy/astropy"
    _REPO_DIR = "/workspace/repo"
    _READY_FLAG = "/tmp/repo_ready"

    _START_COMMAND_TEMPLATE = textwrap.dedent(
        """
        sh -c '
        set -eu

        # install ast-grep
        apk update
        apk add --no-cache ast-grep

        # clone if missing
        if [ ! -d "{repo_dir}/.git" ]; then
          mkdir -p "{repo_dir}"
          git clone --depth 1 "{repo_url}" "{repo_dir}"
        fi

        # signal readiness and keep container alive
        : > "{ready_flag}"
        tail -f /dev/null
        '
        """
    )

    _READY_WAIT_SCRIPT = textwrap.dedent(
        """
        sh -c '
        for i in $(seq 1 240); do
          if [ -f "{ready_flag}" ]; then
            exit 0
          fi
          sleep 0.5
        done
        echo "Sandbox failed to become ready" >&2
        exit 1
        '
        """
    )

    def __init__(self, **kwargs):
        start_command = self._START_COMMAND_TEMPLATE.format(
            repo_url=self._REPO_URL,
            repo_dir=self._REPO_DIR,
            ready_flag=self._READY_FLAG,
        )
        super().__init__(
            sandbox_name="swe-grep-oss-env",
            docker_image="alpine/git",
            start_command=start_command,
            **kwargs,
        )

        # self.add_tool(tools.bash, args_to_skip=["cwd"])
        self.add_tool(tools.result)

    async def setup_state(self, state: vf.State, **kwargs) -> vf.State:
        state = await super().setup_state(state, **kwargs)
        wait_script = self._READY_WAIT_SCRIPT.format(ready_flag=self._READY_FLAG)
        await self.bash(wait_script, sandbox_id=state["sandbox_id"])
        return state

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
        tool_args = super().update_tool_args(tool_name, tool_args, messages, state, **kwargs)

        if tool_name == "bash":
            updated_args = dict(tool_args)
            cmd = tool_args.get("command", "")
            updated_args["command"] = f'cd "{self._REPO_DIR}" && {cmd}'
            return updated_args

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
