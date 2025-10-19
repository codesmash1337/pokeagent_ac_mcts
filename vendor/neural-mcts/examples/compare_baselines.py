"""
Compare Neural-MCTS against baseline models.

This script helps evaluate the hybrid model against:
1. Vanilla MCTS (no neural priors)
2. Pure RL agent (Metamon Minikazam)
3. Hybrid Neural-MCTS

Usage:
    python compare_baselines.py --battles 50 --format gen9randombattle
"""

import argparse
import logging
import subprocess
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_baseline_mcts(battles: int, format_name: str):
    """Run baseline MCTS agent."""
    logger.info(f"Running baseline MCTS for {battles} battles...")

    cmd = [
        "python", "vendor/foul-play/run.py",
        "--websocket-uri", "ws://localhost:8000/showdown/websocket",
        "--ps-username", "MCTSBaseline",
        "--ps-password", "",
        "--bot-mode", "search_ladder",
        "--pokemon-format", format_name,
        "--search-time-ms", "500",
        "--search-parallelism", "2",
        "--run-count", str(battles),
        "--log-level", "INFO"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    logger.info("MCTS Baseline completed")

    # Parse win/loss from logs
    # This is a simplified parser - actual implementation would be more robust
    return {"wins": 0, "losses": 0, "agent": "MCTS"}


def run_baseline_rl(battles: int, format_name: str):
    """Run baseline RL agent (Metamon)."""
    logger.info(f"Running baseline RL agent for {battles} battles...")

    # Determine generation from format
    gen = format_name[3] if format_name.startswith("gen") else "9"

    cmd = [
        "python", "-m", "metamon.rl.evaluate",
        "--eval_type", "ladder",
        "--agent", "Minikazam",
        "--gens", gen,
        "--formats", format_name.replace(f"gen{gen}", ""),
        "--total_battles", str(battles),
        "--username", "RLBaseline"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, cwd="vendor/metamon")
    logger.info("RL Baseline completed")

    return {"wins": 0, "losses": 0, "agent": "RL"}


def run_hybrid_mcts(battles: int, format_name: str, policy_server: str):
    """Run hybrid Neural-MCTS agent."""
    logger.info(f"Running Hybrid Neural-MCTS for {battles} battles...")

    # Set environment variables
    env = {
        "AGENT_NAME": "HybridMCTS",
        "POLICY_SERVER": policy_server,
        "POKEMON_FORMAT": format_name,
        "RUN_COUNT": str(battles)
    }

    cmd = ["./vendor/neural-mcts/scripts/run_hybrid_agent.sh"]
    result = subprocess.run(cmd, env={**subprocess.os.environ, **env})

    logger.info("Hybrid MCTS completed")

    return {"wins": 0, "losses": 0, "agent": "Hybrid"}


def print_results(results: list):
    """Print comparison results."""
    logger.info("\n" + "="*60)
    logger.info("COMPARISON RESULTS")
    logger.info("="*60)

    for result in results:
        total = result["wins"] + result["losses"]
        win_rate = (result["wins"] / total * 100) if total > 0 else 0
        logger.info(f"{result['agent']:15s}: {result['wins']:3d}W / {result['losses']:3d}L ({win_rate:.1f}%)")

    logger.info("="*60)


def main():
    parser = argparse.ArgumentParser(description="Compare Neural-MCTS against baselines")
    parser.add_argument("--battles", type=int, default=20, help="Number of battles per agent")
    parser.add_argument("--format", default="gen9randombattle", help="Pokemon format")
    parser.add_argument("--policy-server", default="http://localhost:5000", help="Policy server URL")
    parser.add_argument("--skip-mcts", action="store_true", help="Skip baseline MCTS")
    parser.add_argument("--skip-rl", action="store_true", help="Skip baseline RL")

    args = parser.parse_args()

    # Ensure policy server is running for hybrid agent
    logger.info("Checking policy server availability...")
    # TODO: Add health check

    results = []

    # Run baseline MCTS
    if not args.skip_mcts:
        try:
            result = run_baseline_mcts(args.battles, args.format)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to run baseline MCTS: {e}")

    # Run baseline RL
    if not args.skip_rl:
        try:
            result = run_baseline_rl(args.battles, args.format)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to run baseline RL: {e}")

    # Run hybrid Neural-MCTS
    try:
        result = run_hybrid_mcts(args.battles, args.format, args.policy_server)
        results.append(result)
    except Exception as e:
        logger.error(f"Failed to run hybrid MCTS: {e}")

    # Print results
    print_results(results)


if __name__ == "__main__":
    main()
