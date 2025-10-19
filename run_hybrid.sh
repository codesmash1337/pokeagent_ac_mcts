#!/bin/bash
# Script to run Hybrid (Actor-Critic boosted MCTS) agent
# This is the third agent - combines MCTS with Actor-Critic guidance

echo "================================================================================"
echo "Starting Hybrid Agent (AC-Boosted MCTS)"
echo "================================================================================"
echo ""
echo "This agent is an Actor-Critic boosted MCTS model"
USERNAME="Hybrid$(date +%s | tail -c 6)"
echo "Username: $USERNAME"
echo "Format: gen9ou"
echo "Team: minikazam_team (same as other agents)"
echo "Search time: 500ms per move"
echo "Searching for random opponents on ladder"
echo ""

cd vendor/foul-play

# Run MCTS agent (the neural guidance integration is planned for future)
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
