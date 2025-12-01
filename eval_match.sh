#!/bin/bash
#SBATCH --partition=general
#SBATCH --mem=100Gb
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH -t 12:00:00
#SBATCH --job-name=eval_base
#SBATCH --error=/home/sanidhyv/agentic-code-search-oss/logs/%x__%j.err
#SBATCH --output=/home/sanidhyv/agentic-code-search-oss/logs/%x__%j.out

# Cache Configuration
export UV_CACHE_DIR="/data/user_data/sanidhyv/.cache/uv"
export HF_HOME="/data/user_data/sanidhyv/.cache/huggingface"
export TRANSFORMERS_CACHE="/data/user_data/sanidhyv/.cache/transformers"
export TORCH_HOME="/data/user_data/sanidhyv/.cache/torch"
export XDG_CACHE_HOME="/data/user_data/sanidhyv/.cache"
export TMPDIR="/data/user_data/sanidhyv/tmp"
export RAY_TMPDIR="/data/user_data/sanidhyv/ray_temp_eval"

mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$TRANSFORMERS_CACHE" "$TORCH_HOME" "$TMPDIR" "$RAY_TMPDIR"

# Environment
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="/home/sanidhyv/agentic-code-search-oss:$PYTHONPATH"

# Configuration
MODEL="openai/Qwen/Qwen3-4B"
NUM_INSTANCES="${NUM_INSTANCES:-100}"
USE_SEMANTIC="${USE_SEMANTIC:-true}"
PORT=8080
VLLM_LOG="/tmp/vllm_server_${SLURM_JOB_ID:-$$}.log"

echo "Base Model Evaluation"
echo "Model: $MODEL"
echo "Instances: $NUM_INSTANCES"
echo "Semantic Search: $USE_SEMANTIC"
echo "vLLM Log: $VLLM_LOG"

# Cleanup function
cleanup() {
    echo ""
    echo "Cleaning up..."
    python3 -c "import ray; ray.shutdown() if ray.is_initialized() else None" 2>/dev/null
    
    # Kill vLLM server
    if [ ! -z "$VLLM_PID" ]; then
        echo "Stopping vLLM server (PID: $VLLM_PID)..."
        kill $VLLM_PID 2>/dev/null || true
        sleep 2
        kill -9 $VLLM_PID 2>/dev/null || true
    fi
    
    # Fallback: kill any vllm processes
    pkill -f "vllm.entrypoints" 2>/dev/null || true
    pkill -f "vllm serve" 2>/dev/null || true
    
    rm -rf "$TMPDIR"/* 2>/dev/null || true
    echo "Cleanup complete"
}
trap cleanup EXIT INT TERM

# Start vLLM server in background
echo "Starting vLLM server on port $PORT..."
echo "(This may take 2-5 minutes to load the model...)"
echo ""

uv run vllm serve "Qwen/Qwen3-4B" \
  --host 0.0.0.0 \
  --port $PORT \
  --gpu-memory-utilization 0.8 \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --dtype auto \
  > "$VLLM_LOG" 2>&1 &

VLLM_PID=$!
echo "vLLM server started with PID: $VLLM_PID"
echo "Monitoring log: $VLLM_LOG"
echo ""

# Wait for server to be ready with verbose progress
echo "Waiting for vLLM server to initialize..."
MAX_WAIT=600  # 10 minutes (model loading can take a while)
WAITED=0
READY=false

# Function to check if server is ready from logs
check_logs_for_ready() {
    if [ -f "$VLLM_LOG" ]; then
        # Look for indicators that server is ready
        if grep -q "Uvicorn running on" "$VLLM_LOG" 2>/dev/null; then
            return 0
        fi
        if grep -q "Application startup complete" "$VLLM_LOG" 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}

while [ $WAITED -lt $MAX_WAIT ]; do
    # Check if process is still running
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo ""
        echo "ERROR: vLLM process died unexpectedly!"
        echo ""
        echo "Last 200 lines of log:"
        tail -200 "$VLLM_LOG"
        exit 1
    fi
    
    # Check server availability via HTTP
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:$PORT/health 2>/dev/null | grep -q "200"; then
        echo ""
        echo "✓ Server health check passed!"
        READY=true
        break
    fi
    
    # check /v1/models endpoint
    if curl -s http://localhost:$PORT/v1/models 2>/dev/null | grep -q "data"; then
        echo ""
        echo "✓ Server models endpoint responding!"
        READY=true
        break
    fi
    
    # Check logs for ready state
    if check_logs_for_ready; then
        # Give it a few more seconds after log message
        sleep 5
        if curl -s http://localhost:$PORT/v1/models > /dev/null 2>&1; then
            echo ""
            echo "✓ Server ready (detected from logs)!"
            READY=true
            break
        fi
    fi
    
    # Show progress every 10 seconds
    if [ $((WAITED % 10)) -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] Waiting... (${WAITED}s / ${MAX_WAIT}s)"
        
        # Show last interesting log line
        if [ -f "$VLLM_LOG" ]; then
            LAST_LINE=$(grep -E "INFO|WARNING|ERROR" "$VLLM_LOG" | tail -1)
            if [ ! -z "$LAST_LINE" ]; then
                echo "  Latest: ${LAST_LINE:0:100}..."
            fi
        fi
    fi
    
    sleep 5
    WAITED=$((WAITED + 5))
done

# Final check
if [ "$READY" != "true" ]; then
    echo ""
    echo "ERROR: vLLM server failed to become ready within $MAX_WAIT seconds"
    echo ""
    echo "Full vLLM log:"
    cat "$VLLM_LOG"
    exit 1
fi

# Test with a simple completion
echo ""
echo "Testing with sample completion..."
TEST_RESPONSE=$(curl -s -X POST http://localhost:$PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"$MODEL"'",
    "messages": [{"role": "user", "content": "Say hello"}],
    "max_tokens": 10
  }' 2>&1)

if echo "$TEST_RESPONSE" | grep -q "choices"; then
    echo "✓ Server test successful!"
    FIRST_RESPONSE=$(echo "$TEST_RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null || echo "")
    if [ ! -z "$FIRST_RESPONSE" ]; then
        echo "  Response: $FIRST_RESPONSE"
    fi
else
    echo "WARNING: Test completion returned unexpected response:"
    echo "$TEST_RESPONSE" | head -20
    echo ""
    echo "Proceeding anyway, evaluation will show if there are issues..."
fi

# Run evaluation
echo "Starting Evaluation"

cd /home/sanidhyv/agentic-code-search-oss

if [ "$USE_SEMANTIC" = "true" ]; then
    uv run python matching.py \
        --num-instances $NUM_INSTANCES \
        --model $MODEL \
        --semantic
else
    uv run python matching.py \
        --num-instances $NUM_INSTANCES \
        --model $MODEL
fi

EXIT_CODE=$?

echo "Evaluation Complete"
echo "Exit code: $EXIT_CODE"
echo "vLLM log: $VLLM_LOG"

exit $EXIT_CODE