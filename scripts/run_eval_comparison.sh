#!/bin/bash
# Convenience script to run both evaluations and compare results

set -e

# Default values
NUM_INSTANCES=50
MODEL="deepseek-ai/deepseek-coder-33b-instruct"
BASE_URL="http://localhost:8000"
REPOS_DIR="./swebench_repos"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--num-instances)
            NUM_INSTANCES="$2"
            shift 2
            ;;
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -u|--url)
            BASE_URL="$2"
            shift 2
            ;;
        --repos-dir)
            REPOS_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -n, --num-instances NUM   Number of instances (default: 50)"
            echo "  -m, --model MODEL        Model name (default: deepseek-ai/deepseek-coder-33b-instruct)"
            echo "  -u, --url URL            vLLM base URL (default: http://localhost:8000)"
            echo "  --repos-dir DIR          Repos directory (default: ./swebench_repos)"
            echo "  -h, --help               Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

echo "=============================================================================="
echo "SWE-bench Evaluation Comparison"
echo "=============================================================================="
echo "Model:        $MODEL"
echo "Instances:    $NUM_INSTANCES"
echo "Base URL:     $BASE_URL"
echo "Repos Dir:    $REPOS_DIR"
echo "=============================================================================="
echo ""

# Check if vLLM is running
echo "Checking vLLM server..."
if ! curl -s "$BASE_URL/v1/models" > /dev/null; then
    echo "❌ ERROR: vLLM server not responding at $BASE_URL"
    echo "Please start vLLM with:"
    echo "  vllm serve $MODEL --host 0.0.0.0 --port 8000"
    exit 1
fi
echo "✓ vLLM server is running"
echo ""

# Run baseline evaluation
echo "=============================================================================="
echo "Running BASELINE evaluation (no semantic search)..."
echo "=============================================================================="
uv run python eval.py \
    --num-instances "$NUM_INSTANCES" \
    --model "$MODEL" \
    --base-url "$BASE_URL" \
    --repos-dir "$REPOS_DIR"

echo ""
echo "✓ Baseline evaluation complete"
echo ""

# Run semantic search evaluation
echo "=============================================================================="
echo "Running SEMANTIC SEARCH evaluation..."
echo "=============================================================================="
uv run python eval.py \
    --semantic \
    --num-instances "$NUM_INSTANCES" \
    --model "$MODEL" \
    --base-url "$BASE_URL" \
    --repos-dir "$REPOS_DIR"

echo ""
echo "✓ Semantic search evaluation complete"
echo ""

# Compare results
echo "=============================================================================="
echo "Comparing results..."
echo "=============================================================================="
uv run python scripts/compare_results.py \
    eval_results_without_semantic.jsonl \
    eval_results_with_semantic.jsonl \
    --verbose \
    --output comparison_results.json

echo ""
echo "✓ Comparison complete!"
echo ""
echo "Results saved to:"
echo "  - eval_results_without_semantic.jsonl"
echo "  - eval_results_with_semantic.jsonl"
echo "  - comparison_results.json"
echo ""
