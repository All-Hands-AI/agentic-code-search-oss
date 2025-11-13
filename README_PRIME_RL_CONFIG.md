# Prime RL Configuration Reference

This document provides a comprehensive reference for all configuration variables available in Prime RL's `orch.toml`, `train.toml`, and `infer.toml` files.

## Table of Contents
- [Configuration System](#configuration-system)
- [orch.toml - Orchestrator Configuration](#orchtom---orchestrator-configuration)
- [train.toml - Trainer Configuration](#traintoml---trainer-configuration)
- [infer.toml - Inference Configuration](#infertoml---inference-configuration)

---

## Configuration System

Prime RL uses `pydantic-settings` for configuration with the following precedence order:
1. **Command-line arguments**: `--key.subkey value`
2. **Config files**: `@ path/to/config.toml`
3. **Environment variables**: `PRIME_KEY__SUBKEY=value`
4. **Defaults**: Built-in default values

---

## orch.toml - Orchestrator Configuration

The orchestrator manages the RL training loop, environment interactions, and rollout generation.

### Top-Level Settings

#### `batch_size` (int, default: 128, min: 1)
Number of samples to train on per step.

#### `rollouts_per_example` (int, default: 1, min: 1)
Number of output sequences to return per example during training.

#### `seq_len` (int, default: 2048)
Sequence length to use for training. Samples shorter than this will be padded; longer samples will be truncated.

#### `max_steps` (int | null, default: null)
Maximum number of training steps to run. If `null`, runs indefinitely.

#### `async_level` (int, default: 1, min: 0)
Maximum number of async levels to use. If 0, does synchronous RL. Otherwise, allows going `async_level` steps ahead of training.

#### `num_train_workers` (int, default: 1, min: 1)
Number of training workers to use for training.

#### `max_concurrent` (int | null, default: null)
Maximum number of concurrent rollouts to generate and score. Creates a global semaphore passed to verifiers Environment. If `null`, does not limit concurrency.

#### `mask_env_responses` (bool, default: true)
Whether to mask environment responses from the loss.

#### `mask_truncated_completions` (bool, default: false)
Whether to mask truncated completions from the loss.

#### `zero_truncated_completions` (bool, default: false)
Whether to override reward scores with 0 for truncated completions.

#### `output_dir` (Path, default: "outputs")
Directory to write outputs to. Populated with checkpoints, weights, rollouts and logs as subdirectories. Should be set to a persistent directory with enough disk space. Must be distinct across experiments on a single node.

#### `bench` (bool, default: false)
Whether to run in benchmark mode. Automatically sets `max_steps` to 5, `async_level` to ~infinity, and disables W&B.

#### `seed` (int | null, default: 42)
Random seed for the orchestrator.

---

### `[client]` - OAI Client Configuration

#### `client.timeout` (int, default: 1200)
Timeout in seconds for API requests.

#### `client.base_url` (list[str], default: ["http://localhost:8000/v1"])
Base URLs to use for the OpenAI API. If multiple URLs are specified, the client will round-robin completion requests across all servers.

#### `client.api_key_var` (str, default: "OPENAI_API_KEY")
Name of environment variable containing the API key. Can be set to an arbitrary string if the inference server is not protected by an API key.

#### `client.server_type` (str: "vllm" | "openai", default: "vllm")
Type of inference server that the client is connected to.

---

### `[model]` - Model Configuration (Orchestrator)

#### `model.name` (str, default: "Qwen/Qwen3-0.6B")
Name or path of the HuggingFace model to use.

#### `model.trust_remote_code` (bool, default: false)
Whether to trust remote code for tokenizer initialization.

---

### `[sampling]` - Sampling Configuration

#### `sampling.temperature` (float, default: 1.0, min: 0)
Scales the output probability distribution. Lower values → more deterministic, higher values → more random. If 0, samples greedily.

#### `sampling.repetition_penalty` (float, default: 1.0, min: 0)
Penalty for repeating tokens. Values > 1.0 discourage repetition, < 1.0 encourage repetition, 1.0 means no penalty.

#### `sampling.max_tokens` (int | null, default: null)
Maximum number of output tokens to generate per turn. If `null`, generates until maximum context length or EOS token.

#### `sampling.min_tokens` (int, default: 0, min: 0)
Minimum number of output tokens to generate per sequence.

#### `sampling.seed` (int | null, default: null)
Random seed to use for sampling. If `null`, no seeding is used.

---

### `[[env]]` - Environment Configuration

Multiple environment configurations can be defined using `[[env]]` (double brackets for arrays).

#### `env.id` (str, default: "reverse-text")
ID of the environment to use.

#### `env.args` (dict, default: {})
Arguments to pass to the environment.

#### `env.name` (str | null, default: null)
Name of the environment to use.

---

### `[env_mix]` - Environment Mixing Configuration

#### `env_mix.strategy` (str: "interleave" | "concatenate", default: "interleave")
Strategy to use for mixing environments.

#### `env_mix.probabilities` (list[float] | null, default: null)
Probabilities to use for each environment.

#### `env_mix.stopping_strategy` (str: "first_exhausted" | "all_exhausted", default: "all_exhausted")
Stopping strategy to use for interleaving environment datasets.

#### `env_mix.seed` (int | null, default: null)
Random seed to use for environment mixing.

---

### `[buffer]` - Data Buffer Configuration

The buffer configuration uses a discriminated union based on the `type` field.

#### Simple Buffer

```toml
[buffer]
type = "simple"
from_scratch = true
seed = null
```

- `buffer.type` (str, default: "simple")
- `buffer.from_scratch` (bool, default: true) - Whether to initialize metadata and rollout buffer from scratch.
- `buffer.seed` (int | null, default: null) - Random seed for the buffer.

#### Difficulty Pool Buffer

```toml
[buffer]
type = "difficulty-pool"
easy_border = 0.8
hard_border = 0.2
easy_fraction = 0.1
hard_fraction = 0.1
```

- `buffer.type` (str: "difficulty-pool")
- `buffer.easy_border` (float, default: 0.8, range: 0-1) - Problems with > this average reward move to easy pool.
- `buffer.hard_border` (float, default: 0.2, range: 0-1) - Problems with < this average reward move to hard pool.
- `buffer.easy_fraction` (float, default: 0.1, range: 0-1) - Fraction of batch from easy samples.
- `buffer.hard_fraction` (float, default: 0.1, range: 0-1) - Fraction of batch from hard samples.

#### Online Difficulty Buffer

```toml
[buffer]
type = "online-difficulty"
min_reward = 0.01
max_reward = 0.99
oversampling_factor = 1.0
```

- `buffer.type` (str: "online-difficulty")
- `buffer.min_reward` (float | null, default: 0.01, range: 0-1) - Minimum reward to include sample in batch.
- `buffer.max_reward` (float | null, default: 0.99, range: 0-1) - Maximum reward to include sample in batch.
- `buffer.oversampling_factor` (float, default: 1.0, min: 0) - Factor by which to oversample during filtering.

---

### `[advantage]` - Advantage Configuration

#### `advantage.std_norm` (str: "local" | "global" | null, default: null)
Standard normalization strategy for advantages.

#### `advantage.length_weighted_mean` (bool, default: false)
Whether to use length-weighted mean for advantage calculation.

#### `advantage.leave_one_out` (bool, default: false)
Whether to use leave-one-out estimation.

#### `advantage.neg_clipped` (bool, default: false)
Whether to clip negative advantages.

---

### `[weight_broadcast]` - Weight Broadcast Configuration

#### Filesystem (Default)

```toml
[weight_broadcast]
type = "filesystem"
```

#### NCCL

```toml
[weight_broadcast]
type = "nccl"
host = "localhost"
port = 29501
timeout = 1200
```

- `weight_broadcast.type` (str: "filesystem" | "nccl", default: "filesystem")
- `weight_broadcast.host` (str, default: "localhost") - Host for NCCL broadcast.
- `weight_broadcast.port` (int, default: 29501) - Port for NCCL broadcast.
- `weight_broadcast.timeout` (int, default: 1200) - Timeout in seconds for NCCL broadcast.

---

### `[ckpt]` - Checkpoint Configuration

#### `ckpt.interval` (int | null, default: null, min: 1)
Interval at which to save the checkpoint.

#### `ckpt.resume_step` (int | null, default: null, min: -1)
Step to resume orchestrator from. If `null`, starts from scratch. If -1, restarts from latest checkpoint available.

#### `ckpt.keep` (int | null, default: null, min: 1)
Keep at most this many recent step checkpoints on disk. If `null`, never cleans old checkpoints.

#### `ckpt.skip_progress` (bool, default: false)
Whether to skip loading the progress from checkpoint.

#### `ckpt.skip_buffer` (bool, default: false)
Whether to skip loading the buffer from checkpoint.

---

### `[val]` - Validation Configuration

#### `val.num_examples` (int, default: 16, min: 1)
Number of examples to use for validation. If -1, uses all examples.

#### `val.rollouts_per_example` (int, default: 1, min: 1)
Number of samples to generate per example for validation.

#### `val.interval` (int, default: 10)
Interval at which to validate the model.

---

### `[eval]` - Online Evaluation Configuration

#### `eval.num_examples` (int, default: -1)
Number of examples to evaluate per environment.

#### `eval.rollouts_per_example` (int, default: 1, min: 1)
Number of samples to generate per example for each environment.

#### `eval.interval` (int, default: 100, min: 1)
Interval at which to evaluate the model.

#### `eval.eval_base_model` (bool, default: true)
Whether to evaluate the base model we are training on.

---

### `[[eval.env]]` - Evaluation Environment Configuration

Multiple evaluation environments can be defined using `[[eval.env]]`.

#### `eval.env.id` (str, default: "reverse-text")
ID of the environment to use.

#### `eval.env.args` (dict, default: {})
Arguments to pass to the environment.

#### `eval.env.name` (str | null, default: null)
Name of the environment.

#### `eval.env.num_examples` (int | null, default: null)
Number of examples to evaluate. If not set, uses `eval.num_examples`.

#### `eval.env.rollouts_per_example` (int | null, default: null)
Number of samples per example. If not set, uses `eval.rollouts_per_example`.

---

### `[eval.sampling]` - Evaluation Sampling Configuration

#### `eval.sampling.temperature` (float | null, default: null, min: 0)
Temperature for evaluation. If `null`, falls back to inference server default.

#### `eval.sampling.repetition_penalty` (float | null, default: null, min: 0)
Repetition penalty for evaluation. If `null`, falls back to server default.

#### `eval.sampling.top_p` (float | null, default: null)
Cumulative probability of top tokens to consider. If `null`, falls back to server default.

#### `eval.sampling.top_k` (int | null, default: null)
Number of top tokens to consider. If -1, all tokens considered. If `null`, falls back to server default.

#### `eval.sampling.min_p` (float | null, default: null)
Minimum probability for a token to be considered. If `null`, falls back to server default.

#### `eval.sampling.max_tokens` (int | null, default: null)
Maximum output tokens for evaluation.

#### `eval.sampling.min_tokens` (int | null, default: null)
Minimum output tokens for evaluation.

#### `eval.sampling.reasoning_effort` (str: "minimal" | "low" | "medium" | "high" | null, default: null)
Constrains reasoning effort for reasoning models.

#### `eval.sampling.seed` (int | null, default: null)
Random seed for evaluation sampling.

---

### `[eval.save]` - Evaluation Save Configuration

#### `eval.save.env_hub` (bool, default: false)
Whether to push eval results to Prime Environment Hub. Requires PRIME_API_KEY and authorization.

---

### `[eval.save.disk]` - Evaluation Disk Save Configuration

#### `eval.save.disk.path` (Path | null, default: null)
Path to save eval results. If `null`, defaults to `<output_dir>/evals/<step_path>/<env_id>`.

---

### `[eval.save.hf]` - Evaluation HuggingFace Save Configuration

#### `eval.save.hf.dataset_name` (str | null, default: null)
Name of HF dataset to save eval results. If `null`, auto-generates a name.

#### `eval.save.hf.dataset_subset` (str | null, default: null)
Subset name of HF dataset. If `null`, defaults to environment ID.

#### `eval.save.hf.dataset_split` (str | null, default: null)
Split name of HF dataset. If `null`, defaults to 'evals'.

#### `eval.save.hf.private` (bool, default: false)
Whether to save to a private HF dataset.

---

### `[log]` - Logging Configuration

#### `log.level` (str, default: "info")
Logging level for the process.

#### `log.vf_level` (str, default: "warn")
Logging level for the verifiers package.

#### `log.file` (bool, default: true)
Whether to log to a file in the output directory.

#### `log.log_data` (bool, default: false)
Whether to log the first data sample to the logger.

---

### `[wandb]` - Weights & Biases Configuration

#### `wandb.project` (str, default: "prime-rl")
The W&B project to log to.

#### `wandb.name` (str | null, default: null)
The W&B run name.

#### `wandb.id` (str | null, default: null)
The W&B run ID. If `null`, a random ID is generated. Set to resume a run.

#### `wandb.offline` (bool, default: false)
Whether to run W&B in offline mode.

---

### `[wandb.log_extras]` - W&B Extra Logging Configuration

#### `wandb.log_extras.samples` (bool, default: true)
Whether to log prompt/response samples to W&B tables.

#### `wandb.log_extras.distributions` (bool, default: true)
Whether to log distributions (rewards, advantages, etc.) to W&B tables.

#### `wandb.log_extras.interval` (int, default: 10, min: 1)
Step interval at which to log extras to W&B table.

---

## train.toml - Trainer Configuration

The trainer handles the actual model training using the data generated by the orchestrator.

### Top-Level Settings

#### `max_steps` (int | null, default: null)
Maximum number of steps to run training for. If `null`, runs indefinitely.

#### `async_level` (int, default: 1, min: 0)
Maximum number of steps that inference can be ahead of training. Higher values yield better throughput but may reduce performance. If 0, fully synchronous.

#### `output_dir` (Path, default: "outputs")
Directory to write outputs to. Must match the orchestrator's `output_dir`.

#### `bench` (bool, default: false)
Whether to run in benchmark mode. Automatically sets `max_steps` to 5 and uses fake data.

#### `memory_profiler_path` (Path | null, default: null)
Path to write memory profile to.

#### `trace_path` (Path | null, default: null)
Path to write PyTorch profiler trace to.

#### `dist_timeout_seconds` (int, default: 600)
Timeout in seconds for torch distributed ops.

---

### `[model]` - Model Configuration (Trainer)

#### `model.name` (str, default: "Qwen/Qwen3-0.6B")
Name or path of the HuggingFace model to use.

#### `model.attn` (str: "sdpa" | "flash_attention_2", default: "flash_attention_2")
The attention implementation to use.

#### `model.reshard_after_forward` (bool, default: true)
Whether to reshard the model after each forward pass.

#### `model.trust_remote_code` (bool, default: false)
Whether to trust remote code for model and tokenizer initialization.

#### `model.dp_replicate` (int, default: 1)
The data parallel dim where model weights are replicated.

#### `model.ep` (int, default: 1)
The expert parallelism to use if the model has MoE layers. If 1, no EP is used.

#### `model.tp` (int, default: 1)
The tensor parallelism size to use. If 1, no TP is used.

#### `model.cp` (int, default: 1)
The context parallelism size to use. If 1, no CP is used.

#### `model.impl` (str: "hf" | "liger_kernel" | "custom", default: "hf")
Whether to use Liger Kernel or custom implementation.

#### `model.load_using_meta` (bool, default: false)
Whether to load the model using meta device then load from HF checkpoint.

#### `model.optimization_dtype` (str: "bfloat16" | "float32", default: "float32")
The dtype to use for model optimization.

#### `model.reduce_dtype` (str: "bfloat16" | "float32", default: "float32")
The dtype to use for model reduce.

#### `model.moe_use_grouped_mm` (bool, default: true)
Whether to use grouped mm for MoE layers. Requires compute capability >= 9.0.

---

### `[model.compile]` - Model Compilation Configuration

#### `model.compile.fullgraph` (bool, default: false)
Whether to compile transformer blocks with fullgraph.

---

### `[model.ac]` - Activation Checkpointing Configuration

#### `model.ac.freq` (int, default: 1, min: 1)
Applies activation checkpointing to every `freq` layers. 1 = full activation checkpointing.

---

### `[model.ac_offloading]` - Activation Offloading Configuration

#### `model.ac_offloading.pin_memory` (bool, default: true)
Whether to pin the offloaded activations to CPU memory.

#### `model.ac_offloading.max_inflight_activations` (int, default: 5, min: 1)
Maximum number of activations to keep while offloading further. More activations = smoother overlap but more GPU memory usage.

---

### `[model.debug]` - Debug Model Configuration

#### `model.debug.num_layers` (int | null, default: null)
The number of layers in the model.

#### `model.debug.random_init` (bool, default: false)
Whether to randomly initialize the model.

---

### `[model.experimental.lora]` - LoRA Configuration

#### `model.experimental.lora.rank` (int, default: 16, min: 1)
Rank of the low-rank decomposition matrices.

#### `model.experimental.lora.alpha` (float, default: 16.0, min: 0)
LoRA scaling parameter.

#### `model.experimental.lora.dropout` (float, default: 0.0, range: 0-1)
LoRA dropout rate.

#### `model.experimental.lora.target_modules` (list[str], default: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
Module names or regex patterns for modules to apply LoRA to.

#### `model.experimental.lora.modules_to_save` (list[str], default: [])
Module names or regex patterns for modules to keep fully trainable.

---

### `[data]` - Data Loader Configuration

#### `data.fake` (FakeDataLoaderConfig | null, default: null)
Whether to use a fake data loader for debugging.

#### `data.fake.batch_size` (int, default: 2, min: 1)
Batch size for fake data loader.

#### `data.fake.seq_len` (int, default: 128, min: 1)
Sequence length for fake data loader.

---

### `[loss]` - Loss Configuration

#### `loss.ratio_type` (str: "token" | "sequence", default: "token")
Type of importance ratio to use.

#### `loss.ratio_length_norm` (bool, default: false)
Whether to normalize the importance ratio by sequence length.

#### `loss.mask_ratio_high` (float, default: 8.0, min: 0)
High threshold for masking importance ratios.

#### `loss.mask_ratio_low` (float, default: 0.125, min: 0)
Low threshold for masking importance ratios.

#### `loss.sequence_mask_ratio_low` (float, default: 0.0, min: 0)
Masks entire sequences when any generated token has an importance ratio below this value.

---

### `[optim]` - Optimizer Configuration

The optimizer configuration uses a discriminated union based on the `type` field.

#### AdamW (Default)

```toml
[optim]
type = "adamw"
lr = 1e-6
weight_decay = 0.01
max_norm = 1.0
betas1 = 0.9
betas2 = 0.999
```

- `optim.type` (str: "adamw", default: "adamw")
- `optim.lr` (float, default: 1e-6, min: 0) - Learning rate.
- `optim.weight_decay` (float, default: 0.01, min: 0) - Weight decay.
- `optim.max_norm` (float, default: 1.0, min: 0) - Maximum gradient norm to clip.
- `optim.betas1` (float, default: 0.9, min: 0) - Beta1 parameter.
- `optim.betas2` (float, default: 0.999, min: 0) - Beta2 parameter.

#### SGD

```toml
[optim]
type = "sgd"
lr = 1e-6
weight_decay = 0.01
max_norm = 1.0
nesterov = true
momentum = 0.9
```

- `optim.type` (str: "sgd")
- `optim.nesterov` (bool, default: true) - Whether to use Nesterov momentum.
- `optim.momentum` (float, default: 0.9) - Momentum factor.

#### Muon

```toml
[optim]
type = "muon"
lr = 1e-6
weight_decay = 0.01
max_norm = 1.0
betas1 = 0.9
betas2 = 0.999
```

- `optim.type` (str: "muon")
- Parameters same as AdamW.

---

### `[scheduler]` - Learning Rate Scheduler Configuration

The scheduler configuration uses a discriminated union based on the `type` field.

#### Constant (Default)

```toml
[scheduler]
type = "constant"
```

#### Linear

```toml
[scheduler]
type = "linear"
warmup_steps = 10
decay_steps = 10
min_lr = 0.0
```

- `scheduler.type` (str: "linear")
- `scheduler.warmup_steps` (int, default: 10, min: 0) - Number of warmup steps.
- `scheduler.decay_steps` (int, default: 10, min: 0) - Number of decay steps during final portion of training.
- `scheduler.min_lr` (float, default: 0.0, min: 0) - Minimum learning rate to converge to.

#### Cosine

```toml
[scheduler]
type = "cosine"
warmup_steps = 10
min_lr = 0.0
```

- `scheduler.type` (str: "cosine")
- `scheduler.warmup_steps` (int, default: 10, min: 0) - Number of warmup steps.
- `scheduler.min_lr` (float, default: 0.0, min: 0) - Minimum learning rate to converge to.

---

### `[ckpt]` - Checkpoint Configuration (Trainer)

#### `ckpt.interval` (int | null, default: null, min: 1)
Interval at which to save the training checkpoint. If `null`, only checkpoints at the end of training.

#### `ckpt.resume_step` (int | null, default: null, min: -1)
Step to resume training from. If `null`, starts from scratch. If -1, restarts from latest checkpoint.

#### `ckpt.keep` (int | null, default: null, min: 1)
Keep at most this many recent step checkpoints on disk. If `null`, never cleans old checkpoints.

#### `ckpt.skip_progress` (bool, default: false)
Whether to skip loading the progress from checkpoint.

#### `ckpt.skip_scheduler` (bool, default: false)
Whether to skip loading the scheduler from checkpoint.

#### `ckpt.skip_dataloader` (bool, default: false)
Whether to skip loading the dataloader from checkpoint.

---

### `[weights]` - Weight Checkpoint Configuration

#### `weights.interval` (int | null, default: null, min: 1)
Interval at which to save weight checkpoint. If `null`, saves all necessary weight checkpoints on RL trainer and only final weight checkpoint on SFT trainer.

#### `weights.save_sharded` (bool, default: false)
Whether to save the weight checkpoint in sharded format.

#### `weights.save_format` (str: "safetensors" | "torch", default: "torch")
The format to save the weight checkpoint in.

#### `weights.save_async` (bool, default: true)
Whether to save the weight checkpoint asynchronously.

#### `weights.save_adapter_separately` (bool, default: false)
Whether to save LoRA adapters separately before merging into full model weights.

---

### `[weight_broadcast]` - Weight Broadcast Configuration (Trainer)

#### Filesystem (Default)

```toml
[weight_broadcast]
type = "filesystem"
```

#### NCCL

```toml
[weight_broadcast]
type = "nccl"
host = "localhost"
port = 29501
timeout = 1200
inference_world_size = 1
```

- `weight_broadcast.type` (str: "filesystem" | "nccl", default: "filesystem")
- `weight_broadcast.host` (str, default: "localhost") - Host for NCCL broadcast.
- `weight_broadcast.port` (int, default: 29501) - Port for NCCL broadcast.
- `weight_broadcast.timeout` (int, default: 1200) - Timeout in seconds.
- `weight_broadcast.inference_world_size` (int, default: 1) - World size for NCCL broadcast.

---

### `[log]` - Logging Configuration (Trainer)

Same as orchestrator logging configuration.

---

### `[wandb]` - Weights & Biases Configuration (Trainer)

Same as orchestrator W&B configuration.

---

## infer.toml - Inference Configuration

The inference configuration sets up the vLLM inference server.

### Top-Level Settings

#### `gpu_memory_utilization` (float, default: 0.9)
The GPU memory utilization to use. Passed to vLLM as `--gpu-memory-utilization`.

#### `seed` (int | null, default: null)
Seed the inference components. If `null`, no seeding is used. Passed to vLLM as `--seed`.

---

### `[server]` - Server Configuration

#### `server.host` (str | null, default: null)
The host to bind to.

#### `server.port` (int, default: 8000)
The port to bind to.

---

### `[model]` - Model Configuration (Inference)

#### `model.name` (str, default: "Qwen/Qwen3-0.6B")
Name or path of the HuggingFace model to use.

#### `model.dtype` (str: "auto" | "float16" | "bfloat16" | "float32", default: "auto")
Data type for model weights and activations. If 'auto', uses FP16 for FP32/FP16 models, BF16 for BF16 models. Passed to vLLM as `--dtype`.

#### `model.max_model_len` (int | null, default: null)
Maximum model context length. If `null`, uses the maximum context length from model config. Passed to vLLM as `--max-model-len`.

#### `model.enforce_eager` (bool, default: false)
Whether to enforce eager mode. If false, uses PyTorch eager and CUDA graphs in hybrid for maximal performance. Passed to vLLM as `--enforce-eager`.

#### `model.trust_remote_code` (bool, default: false)
Whether to trust remote code. Passed to vLLM engine init.

#### `model.enable_auto_tool_choice` (bool, default: false)
Whether to enable auto tool choice. Passed to vLLM as `--enable-auto-tool-choice`.

#### `model.tool_call_parser` (str, default: "hermes")
The tool call parser to use. Passed to vLLM as `--tool-call-parser`.

---

### `[parallel]` - Parallel Configuration

#### `parallel.tp` (int, default: 1)
The tensor parallel size. Passed to vLLM as `--tensor-parallel-size`.

#### `parallel.dp` (int, default: 1, min: 1)
The data parallel size. Passed to vLLM as `--data-parallel-size`.

---

### `[weight_broadcast]` - Weight Broadcast Configuration (Inference)

#### `weight_broadcast.type` (str: "nccl" | "filesystem", default: "filesystem")
The type of weight broadcast to use.

---

## Examples

### Minimal orch.toml

```toml
max_steps = 100
batch_size = 128
seq_len = 2048

[model]
name = "Qwen/Qwen3-0.6B"

[[env]]
id = "reverse-text"
```

### Minimal train.toml

```toml
max_steps = 100

[model]
name = "Qwen/Qwen3-0.6B"

[optim]
lr = 1e-6
```

### Minimal infer.toml

```toml
[model]
name = "Qwen/Qwen3-0.6B"
```

### Advanced Configuration with LoRA and Evaluation

```toml
# orch.toml
max_steps = 1000
batch_size = 2048
seq_len = 8192
rollouts_per_example = 16

[model]
name = "meta-llama/Llama-3.3-70B-Instruct"

[sampling]
temperature = 0.7
max_tokens = 4096

[[env]]
id = "math-problem-solving"
args = { difficulty = "hard" }

[eval]
interval = 50

[[eval.env]]
id = "math500"
rollouts_per_example = 1

[[eval.env]]
id = "aime2024"
rollouts_per_example = 16

[wandb]
project = "my-rl-project"
name = "llama-70b-math"

[ckpt]
interval = 100
keep = 5
```

```toml
# train.toml
max_steps = 1000

[model]
name = "meta-llama/Llama-3.3-70B-Instruct"

[model.experimental.lora]
rank = 16
alpha = 16.0
target_modules = ["q_proj", "v_proj"]

[optim]
type = "adamw"
lr = 1e-6

[scheduler]
type = "cosine"
warmup_steps = 50
min_lr = 1e-7

[ckpt]
interval = 100
```

```toml
# infer.toml
[model]
name = "meta-llama/Llama-3.3-70B-Instruct"

[parallel]
tp = 4
dp = 2

gpu_memory_utilization = 0.95
```

