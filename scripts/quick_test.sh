#!/bin/bash
# Quick test on 5 instances to verify everything works

set -e

MODEL="${1:-deepseek-ai/deepseek-coder-33b-instruct}"
BASE_URL="${2:-http://localhost:8000}"

echo "=============================================================================="
echo "Quick Test (5 instances)"
echo "=============================================================================="
echo "Model:    $MODEL"
echo "URL:      $BASE_URL"
echo "=============================================================================="
echo ""

# Check vLLM
echo "Checking vLLM server..."
if ! curl -s "$BASE_URL/v1/models" > /dev/null; then
    echo "❌ ERROR: vLLM server not responding at $BASE_URL"
    exit 1
fi
echo "✓ vLLM server is running"
echo ""

# Test with semantic search (faster to test just one mode)
echo "Running quick test with semantic search..."
uv run python eval.py \
    --semantic \
    --num-instances 5 \
    --model "$MODEL" \
    --base-url "$BASE_URL" \
    --repos-dir "./test_repos"

echo ""
echo "✓ Quick test complete!"
echo ""
echo "If this worked, run full evaluation with:"
echo "  ./scripts/run_eval_comparison.sh -n 50"
echo ""
