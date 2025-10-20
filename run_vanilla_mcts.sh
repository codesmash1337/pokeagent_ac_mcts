#!/bin/bash
# Run vanilla MCTS agent (no neural guidance)
# This agent will search for battles on the ladder

cd /Users/yadureddy/Documents/GitHub/clean/pokeagent_ac_mcts/vendor/foul-play

source ../../venv/bin/activate

echo "=========================================="
echo "Starting Vanilla MCTS Agent"
echo "Username: VanillaMCTS"
echo "Format: gen9randombattle"
echo "Battles: 5"
echo "Neural Guidance: DISABLED"
echo "=========================================="

python run.py \
    --websocket-uri ws://localhost:8000/showdown/websocket \
    --ps-username VanillaMCTS \
    --ps-password "" \
    --bot-mode search_ladder \
    --pokemon-format gen9randombattle \
    --search-time-ms 500 \
    --run-count 5 \
    --log-level INFO

echo "=========================================="
echo "Vanilla MCTS battles complete!"
echo "=========================================="
