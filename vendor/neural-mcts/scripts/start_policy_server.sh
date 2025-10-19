#!/bin/bash
# Start the Neural Policy Server

set -e

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default values
MODEL_NAME="${MODEL_NAME:-Minikazam}"
DEVICE="${DEVICE:-cpu}"
PORT="${PORT:-5000}"
CACHE_SIZE="${CACHE_SIZE:-1000}"

echo "Starting Neural Policy Server"
echo "  Model: $MODEL_NAME"
echo "  Device: $DEVICE"
echo "  Port: $PORT"
echo "  Cache Size: $CACHE_SIZE"
echo ""

# Set Python path
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/../metamon:$PROJECT_ROOT/../foul-play:$PYTHONPATH"
export METAMON_CACHE_DIR="${METAMON_CACHE_DIR:-$PROJECT_ROOT/../../.metamon_cache}"

# Run the policy server
python -m neural_mcts.policy_server \
    --model "$MODEL_NAME" \
    --device "$DEVICE" \
    --port "$PORT" \
    --cache-size "$CACHE_SIZE" \
    --host "127.0.0.1"
