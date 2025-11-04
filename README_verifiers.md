# Instructions for using the verifiers environment

1. Install dependencies

```bash
uv sync
```

2. Clone some repos from the SWE-bench dataset

```bash
python scripts/clone_repos.py --output-dir ./swebench_repos \
  --dataset princeton-nlp/SWE-bench_Lite \
  --max-instances 1
```

3. Run `vllm` and serve `Qwen3-8B`
```bash
vllm serve Qwen/Qwen3-8B --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser deepseek_r1
```

4. Run the verifiers eval with your model of choice

```bash
uv run vf-eval swe-grep-oss-env --api-base-url http://localhost:8000/v1 --model "Qwen/Qwen3-8B" --num-examples 1 --rollouts-per-example 1
```
