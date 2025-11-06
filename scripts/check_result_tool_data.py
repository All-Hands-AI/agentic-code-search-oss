import json
from datasets import load_dataset
import pandas as pd
from pathlib import Path

from src.utils.parse_patch import parse_patch


def log_example_result_files():
    results_paths = [
        # {
        #     "path": Path("outputs/evals/swe-grep-oss-env--Qwen--Qwen3-8B/78b0c169/results.jsonl"),
        #     "label": "Sequential Tool Calls",
        #     "color": "#1f77b4"
        # },
        {
            "path": Path("outputs/evals/swe-grep-oss-env--Qwen--Qwen3-8B/d297f7fa/results.jsonl"),
            "label": "Parallel Tool Calls",
            "color": "#ff7f0e"
        },
    ]

    all_dfs = []
    for result_config in results_paths:
        data = []
        with open(result_config["path"], "r") as f:
            for line in f:
                data.append(json.loads(line))
        
        df_temp = pd.DataFrame(data)
        all_dfs.append(df_temp)
        print(f"Loaded {len(data)} evaluation results from {result_config['label']}")

    # Combine all dataframes
    df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal: {len(df)} evaluation results across {len(results_paths)} run(s)")

    failed_cases = df[df["result_tool_f1"] == 0.0]
    for case in failed_cases.iterrows():
        completion = case[1]['completion']
        for message in completion:
            if not message.get('tool_calls'):
                continue

            for tool_call in message['tool_calls']:
                parsed_tool_call = json.loads(tool_call)
                if parsed_tool_call['function']['name'] == 'result':
                    print(parsed_tool_call)


def log_example_answer_files():
    dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    instance = dataset[0]
    print(instance['patch'])
    parsed_patch = parse_patch(instance['patch'])
    print(parsed_patch)


if __name__ == "__main__":
    # log_example_answer_files()
    log_example_result_files()