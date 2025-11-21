#!/bin/bash
# Wrapper script for running SWE-bench evaluation with local vLLM
# This sets required environment variables before Python imports litellm

# Set dummy API key for litellm (required even for local endpoints)
export OPENAI_API_KEY="sk-dummy-key"

# Run the evaluation with all arguments passed through
uv run python eval_swebench.py "$@"