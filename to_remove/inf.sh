#!/bin/bash
#SBATCH --partition=general
#SBATCH --mem=150Gb
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH -t 2-00:00
#SBATCH --job-name=inf
#SBATCH --error=/home/sanidhyv/agentic-code-search-oss/logs2/%x__%j.err
#SBATCH --output=/home/sanidhyv/agentic-code-search-oss/logs2/%x__%j.out

echo "============================================================"
echo "vLLM Server Access Information"
echo "============================================================"
echo ""

# Get server info
PORT=8000

echo "1. LOCAL ACCESS (same machine):"
echo "   http://localhost:$PORT/v1"
echo "   http://127.0.0.1:$PORT/v1"
echo ""

echo "2. LOCAL NETWORK ACCESS:"
echo ""

# Get all IP addresses
echo "   Available IP addresses on this machine:"
if command -v ip &> /dev/null; then
    # Linux - get all non-loopback IPv4 addresses
    ip -4 addr show | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | grep -v '^127\.' | while read ip; do
        echo "   ✅ http://$ip:$PORT/v1"
    done
elif command -v ifconfig &> /dev/null; then
    # macOS/BSD - get all non-loopback IPv4 addresses
    ifconfig | grep "inet " | grep -v 127.0.0.1 | awk '{print $2}' | while read ip; do
        echo "   ✅ http://$ip:$PORT/v1"
    done
else
    echo "   (Unable to detect IP addresses automatically)"
    echo "   Run: hostname -I    (Linux)"
    echo "   Run: ifconfig       (macOS)"
fi

echo ""
echo "3. HOSTNAME ACCESS:"
if command -v hostname &> /dev/null; then
    HOSTNAME=$(hostname)
    echo "   http://$HOSTNAME:$PORT/v1"
    echo "   (May require DNS/hosts file configuration)"
fi

echo ""
echo "4. DOCKER CONTAINER ACCESS:"
echo "   From containers on same host: http://host.docker.internal:$PORT/v1"
echo ""

echo "5. PUBLIC/INTERNET ACCESS:"
echo "   To access from the internet, you need:"
echo "   - Port forwarding on your router (forward port $PORT)"
echo "   - Your public IP address (see below)"
echo ""

# Try to get public IP
echo "   Checking your public IP address..."
PUBLIC_IP=""

# Try multiple services
if command -v curl &> /dev/null; then
    PUBLIC_IP=$(curl -s -m 5 ifconfig.me 2>/dev/null)
    if [ -z "$PUBLIC_IP" ]; then
        PUBLIC_IP=$(curl -s -m 5 icanhazip.com 2>/dev/null)
    fi
    if [ -z "$PUBLIC_IP" ]; then
        PUBLIC_IP=$(curl -s -m 5 api.ipify.org 2>/dev/null)
    fi
fi

if [ -n "$PUBLIC_IP" ]; then
    echo "   ✅ Your public IP: $PUBLIC_IP"
    echo "   After port forwarding: http://$PUBLIC_IP:$PORT/v1"
    echo "   (Requires router configuration - see below)"
else
    echo "   ⚠️  Could not detect public IP"
    echo "   Visit https://whatismyip.com to find it manually"
fi

echo ""
echo "============================================================"
echo "TESTING YOUR SERVER"
echo "============================================================"
echo ""
echo "Test from same machine:"
echo "  curl http://localhost:$PORT/health"
echo ""
echo "Test from another machine on your network:"
echo "  curl http://YOUR_LOCAL_IP:$PORT/health"
echo ""
echo "Test with Python:"
echo "  python test_vllm_server.py"
echo ""

echo "============================================================"
echo "FIREWALL CONFIGURATION"
echo "============================================================"
echo ""

# Check OS and provide firewall commands
if [ -f /etc/debian_version ]; then
    echo "Ubuntu/Debian - Allow port $PORT:"
    echo "  sudo ufw allow $PORT/tcp"
    echo "  sudo ufw status"
elif [ -f /etc/redhat-release ]; then
    echo "CentOS/RHEL - Allow port $PORT:"
    echo "  sudo firewall-cmd --permanent --add-port=$PORT/tcp"
    echo "  sudo firewall-cmd --reload"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "macOS - Firewall should allow by default"
    echo "Check: System Preferences > Security & Privacy > Firewall"
else
    echo "Allow incoming connections on port $PORT"
fi

echo ""
echo "============================================================"
echo "ROUTER PORT FORWARDING (for internet access)"
echo "============================================================"
echo ""
echo "To access from the internet:"
echo "1. Log into your router admin panel (usually 192.168.1.1 or 192.168.0.1)"
echo "2. Find 'Port Forwarding' or 'Virtual Server' settings"
echo "3. Add a new rule:"
echo "   - External Port: $PORT"
echo "   - Internal IP: YOUR_LOCAL_IP (from section 2 above)"
echo "   - Internal Port: $PORT"
echo "   - Protocol: TCP"
echo "4. Save and apply"
echo ""
echo "Security Warning: Opening ports to the internet has security risks."
echo "Consider using SSH tunneling or VPN for remote access instead."
echo ""

echo "============================================================"
echo "OPENHANDS CONFIGURATION"
echo "============================================================"
echo ""

# Get first non-loopback IP
FIRST_IP=""
if command -v ip &> /dev/null; then
    FIRST_IP=$(ip -4 addr show | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | grep -v '^127\.' | head -n1)
elif command -v ifconfig &> /dev/null; then
    FIRST_IP=$(ifconfig | grep "inet " | grep -v 127.0.0.1 | awk '{print $2}' | head -n1)
fi

if [ -n "$FIRST_IP" ]; then
    echo "For local OpenHands (same machine):"
    echo "  [llm]"
    echo "  model = \"openai/gpt-oss-20b\""
    echo "  base_url = \"http://localhost:$PORT/v1\""
    echo "  api_key = \"dummy-key\""
    echo ""
    echo "For OpenHands on another machine:"
    echo "  [llm]"
    echo "  model = \"openai/gpt-oss-20b\""
    echo "  base_url = \"http://$FIRST_IP:$PORT/v1\""
    echo "  api_key = \"dummy-key\""
fi

echo ""
echo "============================================================"

# vLLM Server Launch Script for GPT-OSS 20B
# Simple wrapper to start the OpenAI-compatible inference server

set -e

# Configuration
MODEL="openai/gpt-oss-20b"
HOST="0.0.0.0"  # 0.0.0.0 allows external access
PORT=8000
TENSOR_PARALLEL_SIZE=2  # Number of GPUs
GPU_MEMORY_UTIL=0.9     # Use 90% of GPU memory
MAX_MODEL_LEN=2048      # Context length
DTYPE="float16"         # Use FP16 for efficiency

echo "============================================================"
echo "Starting vLLM OpenAI-Compatible Server"
echo "============================================================"
echo "Model: $MODEL"
echo "Host: $HOST (accessible from external machines)"
echo "Port: $PORT"
echo "GPUs: $TENSOR_PARALLEL_SIZE (tensor parallelism)"
echo "GPU Memory: ${GPU_MEMORY_UTIL}%"
echo "Max Context: $MAX_MODEL_LEN tokens"
echo "Data Type: $DTYPE"
echo "============================================================"
echo ""
echo "API Endpoints will be available at:"
echo "  - Chat: http://$HOST:$PORT/v1/chat/completions"
echo "  - Completions: http://$HOST:$PORT/v1/completions"
echo "  - Models: http://$HOST:$PORT/v1/models"
echo "  - Health: http://$HOST:$PORT/health"
echo ""
echo "For external access, use: http://$(hostname -I | awk '{print $1}'):$PORT"
echo "============================================================"
echo ""
echo "Starting server (this will download the model on first run)..."
echo ""

# Start vLLM server
uv run vllm serve "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
  --trust-remote-code
  > logs2/vllm.log 2>&1 &

VLLM_PID=$!
echo "vLLM server started with PID: $VLLM_PID"
echo "Monitoring log: $VLLM_LOG"
