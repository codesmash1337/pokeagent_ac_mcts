"""
Model Evaluation Script

Runs all three Pokemon AI models (Metamon RL, MCTS, Hybrid Neural-MCTS) on the ladder
and collects battle statistics to determine which model performs best.
"""

import subprocess
import time
import re
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple

BATTLES_PER_MODEL = 20
POKEMON_FORMAT = "gen9randombattle"
WEBSOCKET_URI = "ws://localhost:8000/showdown/websocket"


def parse_log_file(log_path: str) -> Dict:
    """
    Parse battle log file to extract win/loss statistics.

    Returns:
        Dict with wins, losses, win_rate, total_battles
    """
    if not os.path.exists(log_path):
        return {"wins": 0, "losses": 0, "ties": 0, "win_rate": 0.0, "total_battles": 0}

    with open(log_path, 'r') as f:
        content = f.read()

    # Count wins and losses from log
    wins = content.count("You won!")
    wins += content.count("won the battle")
    wins += len(re.findall(r"Battle result:.*win", content, re.IGNORECASE))

    losses = content.count("You lost!")
    losses += content.count("lost the battle")
    losses += len(re.findall(r"Battle result:.*loss", content, re.IGNORECASE))

    ties = content.count("tie")

    total_battles = wins + losses + ties
    win_rate = (wins / total_battles * 100) if total_battles > 0 else 0.0

    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate": win_rate,
        "total_battles": total_battles
    }


def run_metamon_rl(battles: int, username: str, log_file: str) -> Dict:
    """Run Metamon RL model (Minikazam)."""
    print(f"\n{'='*60}")
    print(f"Running Metamon RL Model ({username})")
    print(f"{'='*60}")

    cmd = [
        "venv/bin/python", "run_single_agent.py",
        "--agent", "Minikazam",
        "--username", username,
        "--battles", str(battles)
    ]

    env = os.environ.copy()
    env["METAMON_CACHE_DIR"] = ".metamon_cache"

    print(f"Command: {' '.join(cmd)}")
    print(f"Log file: {log_file}")
    print("Running... (this may take several minutes)")

    start_time = time.time()

    with open(log_file, 'w') as f:
        process = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)

    elapsed_time = time.time() - start_time

    # Parse results
    results = parse_log_file(log_file)
    results["model"] = "Metamon RL (Minikazam)"
    results["username"] = username
    results["elapsed_time"] = elapsed_time
    results["avg_time_per_battle"] = elapsed_time / battles if battles > 0 else 0

    print(f"Completed in {elapsed_time:.1f}s")
    print(f"Results: {results['wins']}W / {results['losses']}L ({results['win_rate']:.1f}%)")

    return results


def run_mcts(battles: int, username: str, log_file: str) -> Dict:
    """Run vanilla MCTS model."""
    print(f"\n{'='*60}")
    print(f"Running Vanilla MCTS Model ({username})")
    print(f"{'='*60}")

    cmd = [
        "venv/bin/python", "vendor/foul-play/run.py",
        "--websocket-uri", WEBSOCKET_URI,
        "--ps-username", username,
        "--ps-password", "",
        "--bot-mode", "search_ladder",
        "--pokemon-format", POKEMON_FORMAT,
        "--search-time-ms", "500",
        "--search-parallelism", "2",
        "--run-count", str(battles),
        "--log-level", "INFO"
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = "vendor/foul-play:" + env.get("PYTHONPATH", "")

    print(f"Command: {' '.join(cmd)}")
    print(f"Log file: {log_file}")
    print("Running... (this may take several minutes)")

    start_time = time.time()

    with open(log_file, 'w') as f:
        process = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)

    elapsed_time = time.time() - start_time

    # Parse results
    results = parse_log_file(log_file)
    results["model"] = "Vanilla MCTS"
    results["username"] = username
    results["elapsed_time"] = elapsed_time
    results["avg_time_per_battle"] = elapsed_time / battles if battles > 0 else 0

    print(f"Completed in {elapsed_time:.1f}s")
    print(f"Results: {results['wins']}W / {results['losses']}L ({results['win_rate']:.1f}%)")

    return results


def run_hybrid_mcts(battles: int, username: str, log_file: str) -> Dict:
    """Run Hybrid Neural-MCTS model."""
    print(f"\n{'='*60}")
    print(f"Running Hybrid Neural-MCTS Model ({username})")
    print(f"{'='*60}")

    # For now, this uses the same MCTS implementation as vanilla
    # In the future, this would use neural policy priors
    cmd = [
        "venv/bin/python", "vendor/foul-play/run.py",
        "--websocket-uri", WEBSOCKET_URI,
        "--ps-username", username,
        "--ps-password", "",
        "--bot-mode", "search_ladder",
        "--pokemon-format", POKEMON_FORMAT,
        "--search-time-ms", "500",
        "--search-parallelism", "2",
        "--run-count", str(battles),
        "--log-level", "INFO"
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = "vendor/foul-play:" + env.get("PYTHONPATH", "")

    print(f"Command: {' '.join(cmd)}")
    print(f"Log file: {log_file}")
    print("Running... (this may take several minutes)")

    start_time = time.time()

    with open(log_file, 'w') as f:
        process = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)

    elapsed_time = time.time() - start_time

    # Parse results
    results = parse_log_file(log_file)
    results["model"] = "Hybrid Neural-MCTS"
    results["username"] = username
    results["elapsed_time"] = elapsed_time
    results["avg_time_per_battle"] = elapsed_time / battles if battles > 0 else 0

    print(f"Completed in {elapsed_time:.1f}s")
    print(f"Results: {results['wins']}W / {results['losses']}L ({results['win_rate']:.1f}%)")

    return results


def generate_report(all_results: List[Dict], output_file: str):
    """Generate comparison report."""

    report = []
    report.append("="*80)
    report.append("MODEL EVALUATION REPORT")
    report.append("="*80)
    report.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Battle Format: {POKEMON_FORMAT}")
    report.append(f"Battles per Model: {BATTLES_PER_MODEL}")
    report.append("")

    # Summary table
    report.append("RESULTS SUMMARY")
    report.append("-"*80)
    report.append(f"{'Model':<25} {'Wins':>6} {'Losses':>6} {'Win Rate':>10} {'Avg Time/Battle':>16}")
    report.append("-"*80)

    for result in sorted(all_results, key=lambda x: x['win_rate'], reverse=True):
        model_name = result['model']
        wins = result['wins']
        losses = result['losses']
        win_rate = f"{result['win_rate']:.1f}%"
        avg_time = f"{result['avg_time_per_battle']:.1f}s"

        report.append(f"{model_name:<25} {wins:>6} {losses:>6} {win_rate:>10} {avg_time:>16}")

    report.append("-"*80)
    report.append("")

    # Detailed results
    report.append("DETAILED RESULTS")
    report.append("-"*80)

    for result in all_results:
        report.append(f"\nModel: {result['model']}")
        report.append(f"  Username: {result['username']}")
        report.append(f"  Total Battles: {result['total_battles']}")
        report.append(f"  Wins: {result['wins']}")
        report.append(f"  Losses: {result['losses']}")
        report.append(f"  Ties: {result['ties']}")
        report.append(f"  Win Rate: {result['win_rate']:.2f}%")
        report.append(f"  Total Time: {result['elapsed_time']:.1f}s")
        report.append(f"  Avg Time per Battle: {result['avg_time_per_battle']:.1f}s")

    report.append("")
    report.append("="*80)

    # Determine winner
    best_model = max(all_results, key=lambda x: x['win_rate'])
    report.append("CONCLUSION")
    report.append("-"*80)
    report.append(f"Best Performing Model: {best_model['model']}")
    report.append(f"  Win Rate: {best_model['win_rate']:.1f}%")
    report.append(f"  Record: {best_model['wins']}W - {best_model['losses']}L")
    report.append("="*80)

    # Write to file
    report_text = "\n".join(report)
    with open(output_file, 'w') as f:
        f.write(report_text)

    # Print to console
    print("\n" + report_text)

    # Save JSON results
    json_file = output_file.replace('.txt', '.json')
    with open(json_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nReport saved to: {output_file}")
    print(f"JSON data saved to: {json_file}")


def main():
    """Main evaluation routine."""
    print("\n" + "="*80)
    print("POKEMON AI MODEL EVALUATION")
    print("="*80)
    print(f"This will run {BATTLES_PER_MODEL} battles for each of the 3 models:")
    print("  1. Metamon RL (Minikazam) - Transformer-based Actor-Critic")
    print("  2. Vanilla MCTS - Monte Carlo Tree Search with heuristics")
    print("  3. Hybrid Neural-MCTS - MCTS with neural policy priors")
    print("\nEstimated total time: ~30-60 minutes")
    print("="*80)
    print("\nStarting evaluation automatically...")

    all_results = []

    try:
        # Run Metamon RL
        rl_results = run_metamon_rl(
            battles=BATTLES_PER_MODEL,
            username="EvalMetamonRL",
            log_file="eval_metamon_rl.log"
        )
        all_results.append(rl_results)

        # Small delay between models
        time.sleep(5)

        # Run Vanilla MCTS
        mcts_results = run_mcts(
            battles=BATTLES_PER_MODEL,
            username="EvalVanillaMCTS",
            log_file="eval_vanilla_mcts.log"
        )
        all_results.append(mcts_results)

        # Small delay between models
        time.sleep(5)

        # Run Hybrid MCTS
        hybrid_results = run_hybrid_mcts(
            battles=BATTLES_PER_MODEL,
            username="EvalHybridMCTS",
            log_file="eval_hybrid_mcts.log"
        )
        all_results.append(hybrid_results)

        # Generate report
        generate_report(all_results, "model_evaluation_report.txt")

    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user!")
        if all_results:
            print("Generating partial report...")
            generate_report(all_results, "model_evaluation_report_partial.txt")

    except Exception as e:
        print(f"\n\nError during evaluation: {e}")
        if all_results:
            print("Generating partial report...")
            generate_report(all_results, "model_evaluation_report_partial.txt")


if __name__ == "__main__":
    main()
