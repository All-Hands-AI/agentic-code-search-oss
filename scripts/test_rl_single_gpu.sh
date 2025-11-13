#!/bin/bash
# Test RL training on single GPU with SWE-Grep environment
# Based on successful reverse-text example

set -e

echo "Starting SWE-Grep RL test run on single GPU..."
echo "================================================"
echo ""
echo "Configuration:"
echo "  - Model: willcb/Qwen3-8B (modified chat template for multi-turn)"
echo "  - Training: LoRA (rank=4, alpha=8) + gradient checkpointing"
echo "  - Max steps: 5"
echo "  - Batch size: 4 (minimal for single GPU)"
echo "  - Rollouts per example: 4"
echo "  - Sequence length: 1024 (minimal for single GPU)"
echo "  - GPU memory utilization: 0.5 (inference)"
echo "  - Max tokens: 1024"
echo ""

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Change to prime-rl directory to run the rl command
cd /home/ubuntu/agentic-code-search-oss/prime-rl

# Run RL training with configs from parent directory
uv run rl \
  --trainer @ ../configs/swe-grep-oss/rl/train.toml \
  --orchestrator @ ../configs/swe-grep-oss/rl/orch.toml \
  --inference @ ../configs/swe-grep-oss/rl/infer.toml \
  --model.name willcb/Qwen3-8B \
  --wandb.offline true \
  --inference-gpu-ids [0] \
  --trainer-gpu-ids [0] \
  --inference.gpu-memory-utilization 0.5 \
  --inference.model.enforce-eager \
  --orchestrator.batch-size 4 \
  --orchestrator.seq-len 1024 \
  --orchestrator.rollouts-per-example 4

echo ""
echo "================================================"
echo "Test run completed!"
echo "Check outputs/weights/ for checkpoints"
echo "Check W&B for training metrics"

