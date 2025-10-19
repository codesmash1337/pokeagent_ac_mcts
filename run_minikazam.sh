#!/bin/bash
# Script to run Minikazam agent (Metamon RL)

echo "================================================================================"
echo "Starting Minikazam Agent (Metamon RL)"
echo "================================================================================"
echo ""
echo "This agent uses the Actor-Critic RL model trained on human replays"
USERNAME="Minikazam$(date +%s | tail -c 6)"
echo "Username: $USERNAME"
echo "Format: gen9ou"
echo "Searching for random opponents on ladder"
echo ""

# Set cache directory for metamon
export METAMON_CACHE_DIR=.metamon_cache

venv/bin/python -m metamon.rl.evaluate \
  --agent Minikazam \
  --eval_type ladder \
  --gens 9 \
  --formats ou \
  --total_battles 10 \
  --username "$USERNAME" \
  --team_set competitive \
  --battle_backend poke-env
