#!/bin/bash
# Script to run MCTS agent (Foul-Play)

echo "================================================================================"
echo "Starting MCTS Agent (Foul-Play)"
echo "================================================================================"
echo ""
echo "This agent uses poke-engine's MCTS search with hand-crafted heuristics"
USERNAME="MCTS$(date +%s | tail -c 6)"
echo "Username: $USERNAME"
echo "Format: gen9ou"
echo "Team: minikazam_team (same as Minikazam)"
echo "Search time: 500ms per move"
echo "Searching for random opponents on ladder"
echo ""

cd vendor/foul-play

../../venv/bin/python run.py \
  --websocket-uri ws://localhost:8000/showdown/websocket \
  --ps-username "$USERNAME" \
  --ps-password "" \
  --bot-mode search_ladder \
  --pokemon-format gen9ou \
  --team-name gen9/ou/minikazam_team \
  --search-time-ms 500 \
  --search-parallelism 4 \
  --run-count 10 \
  --log-level INFO
