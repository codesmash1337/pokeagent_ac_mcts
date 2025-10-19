#!/bin/bash

# Example script to run Neural-Guided MCTS on Pokemon Showdown
# This combines MCTS search with Minikazam Actor-Critic policy guidance

# Configuration
WEBSOCKET_URI="ws://localhost:8000/showdown/websocket"
USERNAME="NeuralMCTSBot"
PASSWORD="your_password"
FORMAT="gen9randombattle"
SEARCH_TIME_MS=500
PARALLELISM=2
RUN_COUNT=5

# Neural MCTS parameters
USE_NEURAL_MCTS="--use-neural-mcts"
NEURAL_C_PUCT="1.0"

# Navigate to foul-play directory
cd vendor/foul-play

# Run the bot with neural guidance
python -m fp.run \
    --websocket-uri "$WEBSOCKET_URI" \
    --ps-username "$USERNAME" \
    --ps-password "$PASSWORD" \
    --bot-mode search_ladder \
    --pokemon-format "$FORMAT" \
    --search-time-ms $SEARCH_TIME_MS \
    --search-parallelism $PARALLELISM \
    --run-count $RUN_COUNT \
    $USE_NEURAL_MCTS \
    --neural-c-puct $NEURAL_C_PUCT \
    --log-level INFO \
    --log-to-file

echo "Neural-guided MCTS bot completed!"
echo "Check logs/ directory for battle logs"
