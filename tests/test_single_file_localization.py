import os
import ast
import json
from pathlib import Path
from pydantic import SecretStr

from openhands.tools.preset.planning import get_planning_tools
from openhands.tools.preset.default import get_default_agent
from openhands.sdk import (
    LLM,
    Conversation,
    get_logger,
)

from src.prompts.prompt_builder import get_instruction
from src.utils.instance import clone_instance

logger = get_logger(__name__)


def f1_reward_function(predicted_files, true_files):
    pred, true = set(predicted_files), set(true_files)
    tp = len(pred & true)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(true) if true else 0.0
    if not pred and not true:
        return 1.0
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

def reward_function(final_message, instance):
    predicted_files = set(ast.literal_eval(final_message.split("<file-list>")[1].split("</file-list>")[0]))
    print("Predicted files:", predicted_files)
    true_files = set(x[0] for x in ast.literal_eval(instance["target"]))
    print("True files:", true_files)
    return f1_reward_function(predicted_files, true_files)

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

    agent = get_default_agent(llm, cli_mode=True)

    conversation = Conversation(
        agent=agent,
        max_iteration_per_run=30,
        visualize=True,
        workspace=str(working_dir),
    )
    prompt_path = os.path.join(os.path.dirname(__file__), "..", "src", "prompts", "templates", "file_localization.j2")
    input_message = get_instruction(instance, prompt_path, str(working_dir))
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

    final_message = conversation.agent_final_response()
    print("Final Reward:", reward_function(final_message, instance))

    return 0

if __name__ == "__main__":
    import pandas as pd

    # Build with `uv run src/build_dataset.py --output data/`
    dataset = pd.read_parquet("data/SWE-Gym__SWE-Gym_train/train.parquet")
    instance = dataset.iloc[1000].to_dict()
    target_files = ast.literal_eval(instance["target"])
    print("#" * 100)
    print("Target files:", target_files)
    workspace = Path("testbed/")
    instance_id = instance["instance_id"]
    repo_name = instance["repo"]
    commit_id = instance["base_commit"]

    status, working_dir = clone_instance(repo_name, commit_id, instance_id, workspace)
    print("working_dir:", working_dir)
    working_dir = Path.cwd() / working_dir
    try:
        status = test_single_prompt(instance, working_dir)
    except Exception as e:
        print(f"Error occurred: {e}")
