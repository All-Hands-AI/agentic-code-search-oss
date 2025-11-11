# Prime RL Setup for SWE-Grep Environment

Complete guide for running reinforcement learning training on the SWE-Grep code localization task using Prime RL.

## Quick Start

### Run Training (Single GPU)

```bash
cd /home/ubuntu/agentic-code-search-oss
bash scripts/test_rl_single_gpu.sh
```

This will run 5 training steps with the optimized single-GPU configuration.

## Configuration Files

All configuration files are in `configs/swe-grep-oss/rl/`:

### `train.toml` - Trainer Configuration

```toml
max_steps = 5

[model]
name = "willcb/Qwen3-8B"

[model.ac]
freq = 1  # Full gradient checkpointing (every layer)

[model.experimental.lora]
rank = 4
alpha = 8
dropout = 0.0
target_modules = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj"
]

[optim]
lr = 1e-5
weight_decay = 0.0
```

**Key Settings:**
- **LoRA rank=4**: Minimal adapter size for memory efficiency
- **Gradient checkpointing**: Reduces activation memory by ~50%
- **willcb/Qwen3-8B**: Modified chat template for multi-turn training

### `infer.toml` - Inference Server Configuration

```toml
gpu_memory_utilization = 0.5

[model]
name = "willcb/Qwen3-8B"
enforce_eager = true
enable_auto_tool_choice = true
tool_call_parser = "hermes"
```

**Key Settings:**
- **gpu_memory_utilization=0.5**: Allocates 50% of GPU for inference
- **enforce_eager=true**: Skips torch.compile to reduce memory spikes
- **Tool calling enabled**: For bash and result tools

### `orch.toml` - Orchestrator Configuration

```toml
batch_size = 4
rollouts_per_example = 4
seq_len = 1024
max_steps = 5
mask_truncated_completions = false

[model]
name = "willcb/Qwen3-8B"

[sampling]
max_tokens = 1024

[[env]]
id = "swe-grep-oss-env"
args = { max_tokens = 1024, max_tool_calls = 20 }
```

**Key Settings:**
- **batch_size=4**: Minimal batch for single GPU
- **seq_len=1024**: Reduced sequence length for memory
- **max_tokens=1024**: Matches seq_len for consistency

## Memory Optimization Strategy

Running an 8B model for RL on a single GPU requires aggressive memory optimization:

### Trainer Memory Reduction
1. **LoRA (rank=4)**: ~11M trainable params vs 8B full model
2. **Gradient Checkpointing**: Trades compute for memory
3. **Small Batch (4)**: Reduces activation memory

### Inference Memory Reduction
1. **gpu_memory_utilization=0.5**: Caps KV cache size
2. **enforce_eager=true**: Avoids torch.compile overhead
3. **Short sequences (1024)**: Reduces KV cache needs

### Total Memory Budget (80GB A100)
- Inference server: ~40 GB (50% utilization)
- Trainer: ~30 GB (LoRA + gradient checkpointing)
- Headroom: ~10 GB (for peaks and fragmentation)

## Known Issues & Solutions

### Issue 1: Empty Loss Mask (Current)

**Problem:** All positions in `loss_mask` are `False`, causing empty tensors during training.

**Root Cause:** The environment's rollouts don't have any trainable tokens marked.

**Status:** Under investigation - need to verify how `verifiers` marks assistant vs tool messages.

**Temporary Workaround:** This is a verifiers/environment integration issue, not a memory issue. The infrastructure is working correctly.

### Issue 2: OOM Errors

**Solution Applied:**
- Reduced LoRA rank from 8→4
- Added gradient checkpointing
- Reduced batch size from 16→4
- Reduced sequence length from 4096→1024

**Result:** ✅ Both trainer and inference server load successfully

### Issue 3: Wandb API Key

**Solution:** Use `--wandb.offline true` for local-only logging

## Model Notes: willcb/Qwen3-8B

We use `willcb/Qwen3-8B` instead of the official `Qwen/Qwen3-8B` because:

**Problem with Official Model:**
```python
# Official Qwen3 chat template
{% for message in messages %}
    {% if loop.first and message['role'] != 'system' %}
        <|im_start|>system\nYou are Qwen...<|im_end|>\n
    {% endif %}
    ...
{% endfor %}
```

The `loop.first` check causes issues in multi-turn RL training because the chat template changes depending on conversation history, leading to non-increasing position IDs.

**Solution:**
The `willcb/Qwen3-8B` model has a modified chat template that works correctly for multi-turn training.

**Source:** [Verifiers Training Docs](https://verifiers.readthedocs.io/en/latest/training.html)

## Environment Structure

Your `swe_grep_oss_env.py` environment:

```python
class SWEGrepEnv(vf.StatefulToolEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_tool(tools.bash, args_to_skip=["cwd"])
        self.add_tool(tools.result)
    
    async def is_completed(self, messages, state, **kwargs) -> bool:
        # Returns True when episode should end
        
    async def env_response(self, messages, state, **kwargs):
        # Executes tools and returns responses
```

**Reward Functions (6 metrics):**
1. `correct_files_reward`: Files correctly identified
2. `incorrect_files_penalty`: Penalty for wrong files
3. `missed_files_penalty`: Penalty for missing files
4. `max_tokens_penalty`: Penalty for exceeding token limit
5. `max_tool_calls_penalty`: Penalty for too many tool calls
6. `completion_bonus`: Bonus for successful completion

## Verification Commands

### Check Environment Loading
```bash
uv run python scripts/verify_env.py
```

### Inspect Rollout Data
```bash
cd prime-rl
uv run python ../scripts/inspect_rollout_detailed.py
```

### Check Loss Mask
```bash
cd prime-rl
uv run python ../scripts/check_loss_mask.py
```

### Monitor Training
```bash
cd prime-rl
tail -f outputs/logs/orchestrator.log
tail -f outputs/logs/trainer.stdout
tail -f outputs/logs/inference.stdout
```

## Directory Structure

```
agentic-code-search-oss/
├── swe_grep_oss_env.py              # Your custom environment
├── pyproject.toml                    # Environment registration
├── configs/swe-grep-oss/rl/
│   ├── train.toml                    # Trainer config
│   ├── infer.toml                    # Inference config
│   └── orch.toml                     # Orchestrator config
├── scripts/
│   ├── test_rl_single_gpu.sh        # Main training script
│   ├── verify_env.py                # Environment verification
│   ├── inspect_rollout_detailed.py  # Rollout inspection
│   └── check_loss_mask.py           # Loss mask debugging
└── prime-rl/
    └── outputs/
        ├── logs/                     # Training logs
        ├── rollouts/                 # Generated rollouts
        ├── weights/                  # Model checkpoints
        └── wandb/                    # Wandb logs (offline)
```

## Next Steps

1. **Fix Loss Mask Issue**: Investigate why all positions are masked
   - Check verifiers documentation for trainable message marking
   - Verify environment message format
   - May need to explicitly set trainable regions

2. **Scale Up**: Once working, increase:
   - `max_steps` from 5 to full training
   - `batch_size` from 4 to 8-16 (if using multiple GPUs)
   - `seq_len` from 1024 to 2048-4096
   - `LoRA rank` from 4 to 8-16 for better capacity

3. **Multi-GPU**: For faster training:
   - Use separate GPUs for inference and trainer
   - Increase batch size and sequence length
   - Remove memory constraints

## References

- [Prime RL GitHub](https://github.com/PrimeIntellect-ai/prime-rl)
- [Verifiers GitHub](https://github.com/PrimeIntellect-ai/verifiers)
- [Verifiers Training Docs](https://verifiers.readthedocs.io/en/latest/training.html)
- [SWE-bench Lite Dataset](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite)
