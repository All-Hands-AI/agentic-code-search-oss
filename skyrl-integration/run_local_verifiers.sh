#!/bin/bash
# Launches SkyRL training on the local SWE-Grep environment.
#
# Example:
#   bash integrations/verifiers/run_local_verifiers.sh
#
set -x

ENV_NAME="swe_grep_oss"
DATA_DIR="$HOME/data/swe-grep-local"
NUM_GPUS=1
LOGGER="wandb"  # change to "wandb" if you want to log to W&B

# Set the path to your cloned SWE-bench repos for tool calls
export SWEBENCH_REPOS_DIR="$HOME/agentic-code-search-oss/swebench_repos"

# Unset expandable segments (incompatible with vLLM memory pool)
unset PYTORCH_CUDA_ALLOC_CONF

uv run --isolated --extra vllm -m integrations.verifiers.entrypoints.main_local_verifiers \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="Qwen/Qwen3-4B" \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.num_inference_engines=$NUM_GPUS \
  generator.inference_engine_tensor_parallel_size=1 \
  generator.n_samples_per_prompt=2 \
  trainer.epochs=1 \
  trainer.eval_before_train=false \
  trainer.eval_interval=50 \
  trainer.train_batch_size=16 \
  trainer.policy_mini_batch_size=16 \
  trainer.micro_forward_batch_size_per_gpu=4 \
  trainer.micro_train_batch_size_per_gpu=4 \
  trainer.max_prompt_length=2000 \
  generator.max_input_length=2000 \
  generator.sampling_params.max_generate_length=27000 \
  generator.enable_http_endpoint=true \
  generator.gpu_memory_utilization=0.35 \
  trainer.logger="$LOGGER" \
  environment.env_class="$ENV_NAME" \
  trainer.project_name="swe-grep-local" \
  trainer.run_name="swe_grep_test" \
  trainer.ckpt_interval=-1 \
  trainer.ckpt_path="$HOME/ckpts/swe_grep_ckpt" \
  $@

