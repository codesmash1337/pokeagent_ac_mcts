#!/bin/bash
# Run 5 battles: Vanilla MCTS vs Neural MCTS (with per-node priors)

# Activate virtual environment
source venv/bin/activate

# Set environment
export METAMON_CACHE_DIR=$(pwd)/.metamon_cache

cd vendor/foul-play

echo "======================================================================"
echo "Starting 5-Battle Test: Vanilla MCTS vs Neural MCTS (Per-Node Priors)"
echo "======================================================================"
echo ""
echo "Battle Format: gen9randombattle"
echo "Server: localhost:8000"
echo ""

# Run Player 1: Vanilla MCTS (no neural priors)
echo "Starting Player 1: VanillaMCTS (pure MCTS, no neural priors)..."
python run.py \
    --websocket-uri ws://localhost:8000/showdown/websocket \
    --ps-username VanillaMCTS \
    --ps-password "" \
    --bot-mode search_ladder \
    --pokemon-format gen9randombattle \
    --search-time-ms 500 \
    --run-count 5 \
    --log-level INFO \
    > ../../vanilla_mcts.log 2>&1 &

PLAYER1_PID=$!
echo "VanillaMCTS started (PID: $PLAYER1_PID)"
echo "Log: vanilla_mcts.log"
sleep 3

# Run Player 2: Neural MCTS (with per-node Abra priors)
echo ""
echo "Starting Player 2: NeuralMCTS (MCTS with per-node Abra priors)..."
python run.py \
    --websocket-uri ws://localhost:8000/showdown/websocket \
    --ps-username NeuralMCTS \
    --ps-password "" \
    --bot-mode search_ladder \
    --pokemon-format gen9randombattle \
    --search-time-ms 500 \
    --use-neural-mcts \
    --neural-c-puct 1.5 \
    --run-count 5 \
    --log-level INFO \
    > ../../neural_mcts.log 2>&1 &

PLAYER2_PID=$!
echo "NeuralMCTS started (PID: $PLAYER2_PID)"
echo "Log: neural_mcts.log"

echo ""
echo "======================================================================"
echo "Both players started! Battles will begin shortly..."
echo "======================================================================"
echo ""
echo "Player 1 (VanillaMCTS): PID $PLAYER1_PID (vanilla_mcts.log)"
echo "Player 2 (NeuralMCTS):  PID $PLAYER2_PID (neural_mcts.log)"
echo ""
echo "Waiting for battles to complete..."
echo "You can monitor progress with:"
echo "  tail -f vanilla_mcts.log"
echo "  tail -f neural_mcts.log"
echo ""

# Wait for both processes
wait $PLAYER1_PID
PLAYER1_EXIT=$?

wait $PLAYER2_PID
PLAYER2_EXIT=$?

echo ""
echo "======================================================================"
echo "All 5 battles completed!"
echo "======================================================================"
echo ""

# Extract win/loss records from logs
cd ../..
echo "Results:"
echo ""
echo "VanillaMCTS (Pure MCTS):"
grep -E "W:|Won|Lost" vanilla_mcts.log | tail -5 || echo "  No results found in log"
echo ""
echo "NeuralMCTS (Per-Node Abra Priors):"
grep -E "W:|Won|Lost" neural_mcts.log | tail -5 || echo "  No results found in log"
echo ""
echo "======================================================================"
echo "Full logs available in:"
echo "  - vanilla_mcts.log"
echo "  - neural_mcts.log"
echo "======================================================================"
