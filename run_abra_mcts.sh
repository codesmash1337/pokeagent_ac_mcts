#!/bin/bash
# Run Abra-boosted MCTS agent (with neural guidance)
# This agent will search for battles on the ladder

cd /Users/yadureddy/Documents/GitHub/clean/pokeagent_ac_mcts/vendor/foul-play

source ../../venv/bin/activate

# Set METAMON_CACHE_DIR for model loading
export METAMON_CACHE_DIR="/Users/yadureddy/Documents/GitHub/clean/pokeagent_ac_mcts/.metamon_cache"

echo "=========================================="
echo "Starting Abra Neural MCTS Agent"
echo "Username: AbraMCTS"
echo "Format: gen9randombattle"
echo "Battles: 5"
echo "Neural Guidance: ENABLED (Abra model)"
echo "C_PUCT: 1.0"
echo "=========================================="

python run.py \
    --websocket-uri ws://localhost:8000/showdown/websocket \
    --ps-username AbraMCTS \
    --ps-password "" \
    --bot-mode search_ladder \
    --pokemon-format gen9randombattle \
    --search-time-ms 500 \
    --run-count 5 \
    --use-neural-mcts \
    --neural-c-puct 1.0 \
    --log-level INFO

echo "=========================================="
echo "Abra Neural MCTS battles complete!"
echo "=========================================="
