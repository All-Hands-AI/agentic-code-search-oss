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

3. Run the verifiers eval with your model of choice

```bash
uv run vf-eval swe-grep-oss-env \  --api-base-url https://generativelanguage.googleapis.com/v1beta/openai/ \
  --header 'Content-Type: application/json' \
  --api-key-var GEMINI_API_KEY \
  --model "gemini-2.5-flash" \
  --num-examples 1 \
  --rollouts-per-example 1
```
