#!/bin/bash
#SBATCH --partition=general
#SBATCH --mem=150Gb
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:8
#SBATCH -t 2-00:00:00
#SBATCH --job-name=rl_qwen3_8b
#SBATCH --error=/home/sanidhyv/agentic-code-search-oss/logs/%x__%j.err
#SBATCH --output=/home/sanidhyv/agentic-code-search-oss/logs/%x__%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# Load environment variables if needed
if [ -f .env ]; then
    . .env
fi

# Configuration
MODEL="Qwen/Qwen3-8B"
MODEL_ALIAS="Qwen-Qwen3-8B"
DATA_PATH="${DATA_PATH:-data/SWE-Gym__SWE-Gym_train}"
CKPT_PATH="/data/user_data/sanidhyv/grep/train"
N_ROLLOUTS="${N_ROLLOUTS:-4}"
MAX_LENGTH=16384
export WANDB_API_KEY=bd054e89bc6dc33ce731d090da4a87bffa973032
export WANDB_PROJECT="grep"


# Get number of GPUs
NUM_GPUS=8
NNODES=1
NUM_INFERENCE_ENGINES=4
TP_SIZE=2  # Tensor parallel size for 8B model (use 2 GPUs per engine)
LOGGER=wandb
RUN_NAME="code_search_${MODEL_ALIAS}"

# Create checkpoint directory
mkdir -p $CKPT_PATH
mkdir -p $CKPT_PATH/trajectories

# Create logs directory if it doesn't exist
mkdir -p logs

echo "======================================"
echo "Starting RL Training with Qwen3-8B"
echo "======================================"
echo "Model: $MODEL"
echo "GPUs: $NUM_GPUS"
echo "Data Path: $DATA_PATH"
echo "Checkpoint Path: $CKPT_PATH"
echo "Rollouts per prompt: $N_ROLLOUTS"
echo "Tensor Parallel Size: $TP_SIZE"
echo "======================================"

set -x

# Launch training with semantic search enabled
CUDA_LAUNCH_BLOCKING=1 uv run --isolated -m src.train \
  data.train_data="['$DATA_PATH/train.parquet']" \
  data.val_data="['$DATA_PATH/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path=${MODEL} \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.policy_num_nodes=$NNODES \
  trainer.placement.ref_num_nodes=$NNODES \
  trainer.policy.sequence_parallel_size=$NUM_GPUS \
  generator.num_inference_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine_tensor_parallel_size=$TP_SIZE \
  +generator.traj_dir=$CKPT_PATH/trajectories/ \
  +generator.engine_init_kwargs="{enable_auto_tool_choice:true,tool_call_parser:hermes}" \
  trainer.epochs=20 \
  trainer.eval_batch_size=100 \
  trainer.eval_before_train=false \
  trainer.eval_interval=100 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=4 \
  trainer.policy_mini_batch_size=4 \
  trainer.micro_forward_batch_size_per_gpu=2 \
  trainer.micro_train_batch_size_per_gpu=2 \
  trainer.dump_data_batch=true \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=4096 \
  generator.sampling_params.max_generate_length=${MAX_LENGTH} \
  generator.max_input_length=24000 \
  generator.max_num_batched_tokens=48000 \
  generator.max_turns=20 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=False \
  generator.backend=vllm \
  generator.run_engines_locally=True \
  generator.enable_http_endpoint=True \
  generator.http_endpoint_host='0.0.0.0' \
  generator.http_endpoint_port=8080 \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=true \
  generator.n_samples_per_prompt=${N_ROLLOUTS} \
  generator.gpu_memory_utilization=0.6 \
  trainer.logger="$LOGGER" \
  trainer.project_name="code_search" \
  trainer.run_name=${RUN_NAME} \
  trainer.resume_mode=null \
  trainer.ckpt_path="$CKPT_PATH" \
  generator.use_semantic_search=true \
  embedding_device=cpu \
  max_indices=50 \
  max_cache_size_gb=25.0

echo "======================================"
echo "Training completed!"
echo "======================================"