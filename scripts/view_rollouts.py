#!/usr/bin/env python3
"""
Script to view rollouts by decoding token IDs to text.

Supports two backends:
1. vLLM server detokenize endpoint (default, if server is running)
2. Transformers tokenizer (fallback)

Usage:
    # View latest step with vLLM (server at localhost:8000)
    python scripts/view_rollouts.py

    # View specific step
    python scripts/view_rollouts.py --step 5

    # Use transformers instead of vLLM
    python scripts/view_rollouts.py --backend transformers

    # View specific sample index
    python scripts/view_rollouts.py --sample 0

    # Custom vLLM server URL
    python scripts/view_rollouts.py --vllm-url http://localhost:8000

    # View all samples (careful with output length)
    python scripts/view_rollouts.py --all
"""

import argparse
import json
import sys
from pathlib import Path

import requests
import torch


def get_latest_step(rollout_dir: Path) -> int:
    """Get the latest step number from rollout directory."""
    steps = [int(d.name.split("_")[1]) for d in rollout_dir.iterdir() if d.name.startswith("step_")]
    return max(steps) if steps else 0


def load_rollout(rollout_dir: Path, step: int, rank: int = 0) -> list[dict]:
    """Load rollout data from a specific step and rank."""
    rollout_path = rollout_dir / f"step_{step}" / f"rank_{rank}.pt"
    if not rollout_path.exists():
        raise FileNotFoundError(f"Rollout not found: {rollout_path}")
    return torch.load(rollout_path, weights_only=False)


class VLLMTokenizer:
    """Tokenizer using vLLM server's detokenize endpoint."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.model_name = self._get_model_name()

    def _get_model_name(self) -> str:
        """Get model name from vLLM server."""
        response = requests.get(f"{self.base_url}/v1/models")
        response.raise_for_status()
        models = response.json()["data"]
        if not models:
            raise RuntimeError("No models available on vLLM server")
        return models[0]["id"]

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs to text using vLLM detokenize endpoint."""
        response = requests.post(
            f"{self.base_url}/detokenize",
            json={"model": self.model_name, "tokens": token_ids},
        )
        response.raise_for_status()
        return response.json()["prompt"]


class TransformersTokenizer:
    """Tokenizer using transformers library."""

    def __init__(self, model_name: str = "willcb/Qwen3-4B"):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)


def check_vllm_server(url: str) -> bool:
    """Check if vLLM server is available."""
    try:
        response = requests.get(f"{url}/v1/models", timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def format_sample(idx: int, text: str, micro_batch: dict, show_metadata: bool = True) -> str:
    """Format a single sample for display."""
    lines = [f"\n{'='*80}", f"SAMPLE {idx}", "=" * 80]

    if show_metadata:
        # Get sequence length and other metadata
        input_ids = micro_batch.get("input_ids")
        if input_ids is not None:
            seq_len = input_ids.shape[-1] if len(input_ids.shape) > 1 else len(input_ids)
            lines.append(f"Sequence length: {seq_len} tokens")

        # Show advantages summary if available
        advantages = micro_batch.get("advantages")
        if advantages is not None:
            adv_tensor = advantages.flatten()
            nonzero_adv = adv_tensor[adv_tensor != 0]
            if len(nonzero_adv) > 0:
                lines.append(f"Advantage: mean={nonzero_adv.mean():.4f}, std={nonzero_adv.std():.4f}")

        # Show loss mask info
        loss_mask = micro_batch.get("loss_mask")
        if loss_mask is not None:
            num_loss_tokens = loss_mask.sum().item()
            lines.append(f"Loss tokens: {num_loss_tokens}")

        lines.append("-" * 80)

    lines.append(text)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="View decoded rollouts from RL training")
    parser.add_argument(
        "--rollout-dir",
        type=Path,
        default=Path("prime-rl/outputs/rollouts"),
        help="Path to rollouts directory",
    )
    parser.add_argument("--step", type=int, default=None, help="Step number to view (default: latest)")
    parser.add_argument("--rank", type=int, default=0, help="Rank to load (default: 0)")
    parser.add_argument(
        "--sample", type=int, default=None, help="Specific sample index to view (default: first 3)"
    )
    parser.add_argument("--all", action="store_true", help="View all samples")
    parser.add_argument(
        "--backend",
        choices=["vllm", "transformers", "auto"],
        default="auto",
        help="Tokenizer backend (default: auto - tries vllm first)",
    )
    parser.add_argument("--vllm-url", default="http://localhost:8000", help="vLLM server URL")
    parser.add_argument(
        "--model", default="willcb/Qwen3-4B", help="Model name for transformers backend"
    )
    parser.add_argument("--no-metadata", action="store_true", help="Don't show metadata")
    parser.add_argument("--list-steps", action="store_true", help="List available steps and exit")

    args = parser.parse_args()

    # Resolve rollout directory
    if not args.rollout_dir.is_absolute():
        args.rollout_dir = Path.cwd() / args.rollout_dir

    if not args.rollout_dir.exists():
        print(f"Error: Rollout directory not found: {args.rollout_dir}", file=sys.stderr)
        sys.exit(1)

    # List steps if requested
    if args.list_steps:
        steps = sorted([int(d.name.split("_")[1]) for d in args.rollout_dir.iterdir() if d.name.startswith("step_")])
        print(f"Available steps: {steps}")
        sys.exit(0)

    # Determine step to load
    step = args.step if args.step is not None else get_latest_step(args.rollout_dir)
    print(f"Loading step {step} from {args.rollout_dir}")

    # Load rollout data
    try:
        batches = load_rollout(args.rollout_dir, step, args.rank)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(batches)} micro-batches")

    # Initialize tokenizer
    if args.backend == "auto":
        if check_vllm_server(args.vllm_url):
            print(f"Using vLLM server at {args.vllm_url}")
            tokenizer = VLLMTokenizer(args.vllm_url)
        else:
            print(f"vLLM server not available, using transformers with {args.model}")
            tokenizer = TransformersTokenizer(args.model)
    elif args.backend == "vllm":
        if not check_vllm_server(args.vllm_url):
            print(f"Error: vLLM server not available at {args.vllm_url}", file=sys.stderr)
            sys.exit(1)
        tokenizer = VLLMTokenizer(args.vllm_url)
    else:
        tokenizer = TransformersTokenizer(args.model)

    # Determine which samples to show
    if args.sample is not None:
        sample_indices = [args.sample]
    elif args.all:
        sample_indices = list(range(len(batches)))
    else:
        sample_indices = list(range(min(3, len(batches))))

    # Decode and display samples
    for idx in sample_indices:
        if idx >= len(batches):
            print(f"Warning: Sample {idx} out of range (max: {len(batches) - 1})", file=sys.stderr)
            continue

        micro_batch = batches[idx]
        input_ids = micro_batch["input_ids"]

        # Handle batched vs unbatched input_ids
        if len(input_ids.shape) > 1:
            # Batched: shape is [batch, seq_len] - typically [1, seq_len] for packed sequences
            token_ids = input_ids[0].tolist()
        else:
            token_ids = input_ids.tolist()

        # Filter out padding tokens (usually 0 or tokenizer's pad_token_id)
        # For packed sequences, we might want to keep all tokens
        text = tokenizer.decode(token_ids)

        print(format_sample(idx, text, micro_batch, show_metadata=not args.no_metadata))


if __name__ == "__main__":
    main()

