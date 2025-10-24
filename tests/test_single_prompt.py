import os
import ast
import json
from pathlib import Path
from pydantic import SecretStr

from openhands.tools.preset.planning import get_planning_tools
from openhands.sdk import (
    LLM,
    Agent,
    Conversation,
    get_logger,
)

from src.prompts.prompt_builder import get_instruction
from src.utils.instance import clone_instance

logger = get_logger(__name__)

def test_single_prompt(instance, working_dir) -> None:
    api_key = os.getenv("LLM_KEY")
    api_url = os.getenv("LLM_URL")
    api_mdl = os.getenv("LLM_MODEL")
    assert api_key is not None, "LLM_KEY environment variable is not set."
    assert api_url is not None, "LLM_URL environment variable is not set."
    assert api_mdl is not None, "LLM_MODEL environment variable is not set."
    llm = LLM(
        service_id="agent",
        model=api_mdl,
        base_url=api_url,
        api_key=SecretStr(api_key),
    )

    agent = Agent(
        llm=llm,
        tools=get_planning_tools()[:-1],
        cli_mode=False,
        system_prompt_filename="search.j2",
    )

    conversation = Conversation(
        agent=agent,
        max_iteration_per_run=10,
        visualize=True,
        workspace=str(working_dir),
    )

    input_message = get_instruction(instance, None, str(working_dir))
    conversation.send_message(input_message)
    print("Starting conversation...")
    try:
        logger.info("Conversation Starting")
        conversation.run()
    except Exception as e:
        logger.error(f"Error is sending conversation: {e}", exc_info=True)
    finally:
        conversation.close()
        logger.info("Conversation Finished")

    messages = list(map(lambda event: event.model_dump(), conversation.state.events))
    print("=" * 100)
    print("Conversation finished. Got the following LLM messages:")
    for i, message in enumerate(messages):
        print(f"Message {i}: {str(message)[:200]}")

    with open(f"train_traj_{instance_id}.jsonl", "w") as f:
        f.writelines(json.dumps(msg) + "\n" for msg in messages)

    return 0

if __name__ == "__main__":
    import pandas as pd

    # Build with `uv run src/build_dataset.py --output data/`
    dataset = pd.read_parquet("data/SWE-Gym__SWE-Gym_train/train.parquet")
    instance = dataset.iloc[0].to_dict()
    target_files = ast.literal_eval(instance["target"])
    print("#" * 100)
    print("Target files:", target_files)
    workspace = "testbed/"
    instance_id = instance["instance_id"]
    repo_name = instance["repo"]
    commit_id = instance["base_commit"]

    status, working_dir = clone_instance(repo_name, commit_id, instance_id, Path("testbed/"))
    print("working_dir:", working_dir)
    try:
        status = test_single_prompt(instance, working_dir)
    except Exception as e:
        print(f"Error occurred: {e}")