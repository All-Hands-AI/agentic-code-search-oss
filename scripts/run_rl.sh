#!/bin/bash

# Set PyTorch CUDA allocator config to reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Navigate to prime-rl directory
cd $HOME/agentic-code-search-oss/prime-rl

# Run RL training
uv run rl \
  --trainer @ ../configs/swe-grep-oss/rl/train.toml \
  --orchestrator @ ../configs/swe-grep-oss/rl/orch.toml \
  --inference @ ../configs/swe-grep-oss/rl/infer.toml

