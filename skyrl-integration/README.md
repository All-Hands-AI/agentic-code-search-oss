# SkyRL Integration Files

This directory (`skyrl-integration/`) contains integration files for training with SkyRL on the local SWE-Grep environment.

## Files in this Directory

1. **`prepare_local_dataset.py`** - Script to generate Parquet datasets from your environment
2. **`local_verifiers_generator.py`** - Custom generator for SkyRL training
3. **`entrypoints/main_local_verifiers.py`** - Custom entrypoint for SkyRL training
4. **`run_local_verifiers.sh`** - Shell script to launch training
5. **`README.md`** - This file (complete setup and usage guide)

## Related Files (in parent directory)

- **`../swe_grep_oss_env.py`** - Your local Verifiers environment definition
- **`../swebench_repos/`** - Cloned SWE-bench repositories

## Quick Start

### Prerequisites

- SkyRL installed at `/home/ubuntu/SkyRL/skyrl-train`
- SWE-bench repos cloned at `./swebench_repos`
- Ray cluster running
- WandB API key set (optional)

### Step 1: Copy Files to SkyRL

```bash
# From the skyrl-integration directory
cd /home/ubuntu/agentic-code-search-oss/skyrl-integration

# Copy integration files to SkyRL
cp prepare_local_dataset.py /home/ubuntu/SkyRL/skyrl-train/integrations/verifiers/
cp local_verifiers_generator.py /home/ubuntu/SkyRL/skyrl-train/integrations/verifiers/
cp entrypoints/main_local_verifiers.py /home/ubuntu/SkyRL/skyrl-train/integrations/verifiers/entrypoints/
cp run_local_verifiers.sh /home/ubuntu/SkyRL/skyrl-train/integrations/verifiers/
```

### Step 2: Prepare Dataset

```bash
cd /home/ubuntu/SkyRL/skyrl-train

uv run --isolated --extra vllm python integrations/verifiers/prepare_local_dataset.py \
  --output_dir ~/data/swe-grep-local \
  --num_train 100 \
  --num_eval 20
```

### Step 3: Launch Training

```bash
cd /home/ubuntu/SkyRL/skyrl-train
bash integrations/verifiers/run_local_verifiers.sh
```

## Configuration

The training is configured for a small test run in `run_local_verifiers.sh`:

- **Model**: Qwen/Qwen3-4B (or Qwen/Qwen3-8B)
- **Batch size**: 16 (train), 4 (micro)
- **Samples per prompt**: 2
- **Max prompt length**: 2000 tokens
- **Max generation length**: 27000 tokens
- **GPU memory utilization**: 35% (for vLLM)
- **Epochs**: 1

## Environment Variables

The integration sets these environment variables:

- `SWEBENCH_REPOS_DIR`: Path to cloned SWE-bench repositories
- `PYTORCH_CUDA_ALLOC_CONF`: Unset to avoid conflicts with vLLM

## Architecture

### Path-Based Imports

The integration uses path-based imports to avoid dependency conflicts:

1. `sys.path.insert(0, "/home/ubuntu/agentic-code-search-oss")` - Adds your environment to Python path
2. `sys.path.insert(0, "/home/ubuntu/agentic-code-search-oss/.venv/lib/python3.12/site-packages")` - Adds dependencies

This allows importing `swe_grep_oss_env` without installing it as a package.

### Custom Generator

`local_verifiers_generator.py` extends the standard Verifiers generator to:

1. Import your local environment directly
2. Call `load_environment()` from your module
3. Handle the Verifiers environment's rollout and reward processing

### Custom Entrypoint

`main_local_verifiers.py` extends the standard Verifiers entrypoint to:

1. Use the custom `LocalVerifiersGenerator`
2. Set the environment class to `swe_grep_oss`

## Troubleshooting

### Ray Version Mismatch

If you see a Ray version mismatch error, restart the Ray cluster:

```bash
ray stop
ray start --head
```

### Import Errors

If you see import errors for `verifiers` or other packages, check that:

1. The local venv path is correct in `prepare_local_dataset.py` and `local_verifiers_generator.py`
2. Your local environment has all dependencies installed

### GPU Out of Memory

If you hit OOM errors, reduce:

- `generator.gpu_memory_utilization` (currently 0.35)
- `trainer.train_batch_size` (currently 16)
- `generator.sampling_params.max_generate_length` (currently 27000)

### HTTP 500 Errors

If you see HTTP 500 errors about missing 'choices' field, this is a known limitation of SkyRL's HTTP endpoint. The standard Verifiers integration has the same issue. Potential workarounds:

1. Run a separate vLLM server (requires more GPU memory)
2. Fix SkyRL's HTTP endpoint to return proper OpenAI format (requires modifying SkyRL)

## Notes

- The integration is designed to work with a single 80GB GPU
- CPU offloading for the reference model can be enabled with `trainer.ref.fsdp_config.cpu_offload=true`
- The dataset format matches the standard Verifiers format with an additional `environment` field

## See Also

- `LOCAL_SETUP.md` - Detailed setup guide
- `README_verifiers.md` - Your original Verifiers environment documentation

