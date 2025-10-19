#!/usr/bin/env python3
"""Queue Hybrid Neural-MCTS agent on local ladder for 10 battles."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Setup paths
REPO_ROOT = Path(__file__).resolve().parent
FP_ROOT = REPO_ROOT / "vendor" / "foul-play"
if str(FP_ROOT) not in sys.path:
    sys.path.insert(0, str(FP_ROOT))

# Use the existing foul-play MCTS implementation
# In the future, this would query the neural policy server for priors

def main() -> None:
    """Run Hybrid Neural-MCTS agent on local ladder."""
    print("="*60)
    print("Hybrid Neural-MCTS - Local Ladder Queue")
    print("="*60)
    print("Model: MCTS with Neural Policy Priors (Framework)")
    print("Battles: 10")
    print("Format: gen9randombattle")
    print("Search Time: 500ms per move")
    print("Parallelism: 2")
    print("Note: Currently using vanilla MCTS (neural priors pending)")
    print("="*60)

    # Use the foul-play run script directly
    import subprocess

    cmd = [
        "venv/bin/python", "vendor/foul-play/run.py",
        "--websocket-uri", "ws://localhost:8000/showdown/websocket",
        "--ps-username", "HybridMCTS_Eval",
        "--ps-password", "",
        "--bot-mode", "search_ladder",
        "--pokemon-format", "gen9randombattle",
        "--search-time-ms", "500",
        "--search-parallelism", "2",
        "--run-count", "10",
        "--log-level", "INFO"
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(FP_ROOT) + ":" + env.get("PYTHONPATH", "")

    print("\nStarting Hybrid MCTS agent...")
    print("Command:", " ".join(cmd))
    print()

    try:
        subprocess.run(cmd, env=env, check=True)
        print("\n" + "="*60)
        print("EVALUATION COMPLETE")
        print("="*60)
    except subprocess.CalledProcessError as e:
        print(f"\nError running agent: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user!")
        sys.exit(0)


if __name__ == "__main__":
    main()
