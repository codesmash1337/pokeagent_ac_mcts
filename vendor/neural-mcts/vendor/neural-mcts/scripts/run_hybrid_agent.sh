#!/bin/bash
# Run a Hybrid Neural-MCTS Agent on Pokemon Showdown

set -e

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default values
AGENT_NAME="${AGENT_NAME:-NeuralMCTSAgent1}"
POLICY_SERVER="${POLICY_SERVER:-http://127.0.0.1:5000}"
WEBSOCKET_URI="${WEBSOCKET_URI:-ws://localhost:8000/showdown/websocket}"
POKEMON_FORMAT="${POKEMON_FORMAT:-gen9randombattle}"
SEARCH_TIME_MS="${SEARCH_TIME_MS:-500}"
PARALLELISM="${PARALLELISM:-2}"
RUN_COUNT="${RUN_COUNT:-10}"

echo "Starting Hybrid Neural-MCTS Agent"
echo "  Agent Name: $AGENT_NAME"
echo "  Policy Server: $POLICY_SERVER"
echo "  Websocket: $WEBSOCKET_URI"
echo "  Format: $POKEMON_FORMAT"
echo "  Search Time: ${SEARCH_TIME_MS}ms"
echo "  Parallelism: $PARALLELISM"
echo "  Battles: $RUN_COUNT"
echo ""

# Set Python path
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/../foul-play:$PROJECT_ROOT/../metamon:$PYTHONPATH"
export METAMON_CACHE_DIR="${METAMON_CACHE_DIR:-$PROJECT_ROOT/../../.metamon_cache}"

# Run the hybrid agent (using neural_search.py with foul-play integration)
python -m neural_mcts.neural_search \
    --agent-name "$AGENT_NAME" \
    --policy-server "$POLICY_SERVER" \
    --websocket-uri "$WEBSOCKET_URI" \
    --pokemon-format "$POKEMON_FORMAT" \
    --search-time-ms "$SEARCH_TIME_MS" \
    --parallelism "$PARALLELISM" \
    --run-count "$RUN_COUNT"
