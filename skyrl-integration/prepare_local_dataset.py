import argparse
import os
import sys
from functools import partial
from typing import Any, Dict

# Add local environment to path and activate its venv
local_env_path = "/home/ubuntu/agentic-code-search-oss"
local_venv_path = os.path.join(local_env_path, ".venv", "lib", "python3.12", "site-packages")

# Insert both paths at the beginning
sys.path.insert(0, local_venv_path)
sys.path.insert(0, local_env_path)

from swe_grep_oss_env import load_environment


def extract_env_name(env_id: str) -> str:
    """Return only the environment name from strings like 'org/name@version' or 'name@version'."""
    base = env_id.split("/")[-1]
    return base.split("@")[0]


def build_row(sample: Dict[str, Any], data_source: str, env_name: str) -> Dict[str, Any]:
    if "prompt" not in sample:
        raise ValueError("Example must contain a 'prompt' field")
    prompt = sample["prompt"]  # Already formatted by the environment as chat messages

    answer = sample.get("answer", "")
    info = sample.get("info", None)
    task = sample.get("task", "default")

    full_sample = {
        "data_source": data_source,
        "prompt": prompt,
        "verifiers": {
            "answer": answer,
            "task": task,
            "environment": env_name,
        },
    }

    if info:
        full_sample["verifiers"]["info"] = info

    return full_sample


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Parquet dataset from local SWE-Grep environment.")
    parser.add_argument(
        "--output_dir", default=None, help="Output directory for Parquet files. Defaults to ~/data/swe-grep-local"
    )
    parser.add_argument(
        "--num_train", type=int, default=100, help="Number of training examples to generate. -1 for no limit."
    )
    parser.add_argument(
        "--num_eval", type=int, default=20, help="Number of evaluation examples to generate. -1 for no limit."
    )

    args = parser.parse_args()

    # Resolve output directory
    output_dir_name = args.output_dir if args.output_dir else "~/data/swe-grep-local"
    output_dir = os.path.expanduser(output_dir_name)
    os.makedirs(output_dir, exist_ok=True)

    # Load local environment
    env_name = "swe_grep_oss"
    print(f"Loading local environment: {env_name}")
    vf_env = load_environment()
    
    # Access the dataset directly from the environment
    # The environment stores the dataset in self.dataset
    dataset = vf_env.dataset
    
    data_source = f"swe-grep-local/{env_name}"
    map_fn = partial(build_row, data_source=data_source, env_name=env_name)

    # Split dataset into train and validation
    # If num_train is -1, use all but num_eval samples for training
    total_samples = len(dataset)
    print(f"Total samples in dataset: {total_samples}")
    
    if args.num_train == -1:
        if args.num_eval == -1:
            # Use 80/20 split
            num_train = int(total_samples * 0.8)
            num_eval = total_samples - num_train
        else:
            num_train = total_samples - args.num_eval
            num_eval = args.num_eval
    else:
        num_train = min(args.num_train, total_samples)
        if args.num_eval == -1:
            num_eval = total_samples - num_train
        else:
            num_eval = min(args.num_eval, total_samples - num_train)
    
    print(f"Using {num_train} training samples and {num_eval} validation samples")
    
    # Create train dataset
    train_ds = dataset.select(range(num_train))
    train_ds = train_ds.map(map_fn, num_proc=16)
    # Drop top-level 'info' column if it exists
    if "info" in train_ds.column_names:
        train_ds = train_ds.remove_columns("info")
    train_path = os.path.join(output_dir, "train.parquet")
    train_ds.to_parquet(train_path)
    print(f"Saved training dataset to {train_path}")

    # Create validation dataset
    eval_ds = dataset.select(range(num_train, num_train + num_eval))
    eval_ds = eval_ds.map(map_fn, num_proc=16)
    # Drop top-level 'info' column if it exists
    if "info" in eval_ds.column_names:
        eval_ds = eval_ds.remove_columns("info")
    val_path = os.path.join(output_dir, "validation.parquet")
    eval_ds.to_parquet(val_path)
    print(f"Saved validation dataset to {val_path}")
    
    print(f"\nDataset preparation complete!")
    print(f"Train samples: {len(train_ds)}")
    print(f"Validation samples: {len(eval_ds)}")

