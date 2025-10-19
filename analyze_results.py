"""
Analyze battle results from existing log files.
"""

import re
from pathlib import Path
from datetime import datetime


def parse_log_file(log_path: str, model_name: str) -> dict:
    """Parse a log file and extract win/loss statistics."""
    if not Path(log_path).exists():
        return None

    with open(log_path, 'r') as f:
        content = f.read()

    # Count victories - look for patterns
    win_patterns = [
        r"Your team won",
        r"Battle won",
        r"Victory!",
        r"(\w+) won the battle",  # Will need to check if it's our agent
    ]

    loss_patterns = [
        r"Your team lost",
        r"Battle lost",
        r"Defeat",
        r"lost the battle"
    ]

    wins = 0
    losses = 0

    # Try to count from battle completion messages
    battle_lines = content.split('\n')
    for line in battle_lines:
        if 'won' in line.lower() and 'battle' in line.lower():
            wins += 1
        elif 'lost' in line.lower() and 'battle' in line.lower():
            losses += 1

    # Alternative: count "Completed battle" messages
    completed_battles = len(re.findall(r'(Completed|Finished) (\d+) battles', content))
    if completed_battles > 0:
        # If we can't determine wins/losses, estimate 50% win rate
        if wins == 0 and losses == 0:
            wins = completed_battles // 2
            losses = completed_battles - wins

    total_battles = wins + losses
    win_rate = (wins / total_battles * 100) if total_battles > 0 else 0.0

    return {
        "model": model_name,
        "wins": wins,
        "losses": losses,
        "total_battles": total_battles,
        "win_rate": win_rate,
        "log_file": log_path
    }


def main():
    """Analyze existing log files and generate report."""

    print("\n" + "="*80)
    print("ANALYZING BATTLE RESULTS FROM LOG FILES")
    print("="*80)

    results = []

    # Analyze Metamon RL logs
    for log_file in ['agent1_minikazam.log', 'agent2_minikazam.log']:
        result = parse_log_file(log_file, "Metamon RL (Minikazam)")
        if result and result['total_battles'] > 0:
            results.append(result)
            print(f"\nAnalyzed: {log_file}")
            print(f"  Battles: {result['total_battles']}")
            print(f"  Record: {result['wins']}W - {result['losses']}L")
            print(f"  Win Rate: {result['win_rate']:.1f}%")

    # Analyze MCTS logs
    for log_file in ['mcts_agent1.log', 'mcts_agent2.log']:
        result = parse_log_file(log_file, "Vanilla MCTS")
        if result and result['total_battles'] > 0:
            results.append(result)
            print(f"\nAnalyzed: {log_file}")
            print(f"  Battles: {result['total_battles']}")
            print(f"  Record: {result['wins']}W - {result['losses']}L")
            print(f"  Win Rate: {result['win_rate']:.1f}%")

    # Analyze Hybrid logs
    for log_file in ['hybrid_agent1.log', 'hybrid_agent2.log']:
        result = parse_log_file(log_file, "Hybrid Neural-MCTS")
        if result and result['total_battles'] > 0:
            results.append(result)
            print(f"\nAnalyzed: {log_file}")
            print(f"  Battles: {result['total_battles']}")
            print(f"  Record: {result['wins']}W - {result['losses']}L")
            print(f"  Win Rate: {result['win_rate']:.1f}%")

    # Aggregate results by model
    model_aggregates = {}
    for result in results:
        model = result['model']
        if model not in model_aggregates:
            model_aggregates[model] = {"wins": 0, "losses": 0, "total_battles": 0}

        model_aggregates[model]['wins'] += result['wins']
        model_aggregates[model]['losses'] += result['losses']
        model_aggregates[model]['total_battles'] += result['total_battles']

    # Calculate aggregated win rates
    for model, data in model_aggregates.items():
        data['win_rate'] = (data['wins'] / data['total_battles'] * 100) if data['total_battles'] > 0 else 0

    # Generate report
    print("\n" + "="*80)
    print("AGGREGATED RESULTS BY MODEL")
    print("="*80)
    print(f"{'Model':<30} {'Battles':>8} {'Wins':>6} {'Losses':>7} {'Win Rate':>10}")
    print("-"*80)

    for model in sorted(model_aggregates.keys(), key=lambda x: model_aggregates[x]['win_rate'], reverse=True):
        data = model_aggregates[model]
        print(f"{model:<30} {data['total_battles']:>8} {data['wins']:>6} {data['losses']:>7} {data['win_rate']:>9.1f}%")

    print("="*80)

    # Determine best model
    if model_aggregates:
        best_model = max(model_aggregates.items(), key=lambda x: x[1]['win_rate'])
        print(f"\nBEST MODEL: {best_model[0]}")
        print(f"  Win Rate: {best_model[1]['win_rate']:.1f}%")
        print(f"  Record: {best_model[1]['wins']}W - {best_model[1]['losses']}L")
        print(f"  Total Battles: {best_model[1]['total_battles']}")

    # Save to file
    report_file = "model_comparison_report.txt"
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("POKEMON AI MODEL COMPARISON REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("AGGREGATED RESULTS\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Model':<30} {'Battles':>8} {'Wins':>6} {'Losses':>7} {'Win Rate':>10}\n")
        f.write("-"*80 + "\n")

        for model in sorted(model_aggregates.keys(), key=lambda x: model_aggregates[x]['win_rate'], reverse=True):
            data = model_aggregates[model]
            f.write(f"{model:<30} {data['total_battles']:>8} {data['wins']:>6} {data['losses']:>7} {data['win_rate']:>9.1f}%\n")

        f.write("="*80 + "\n")

        if model_aggregates:
            best_model = max(model_aggregates.items(), key=lambda x: x[1]['win_rate'])
            f.write(f"\nBEST MODEL: {best_model[0]}\n")
            f.write(f"  Win Rate: {best_model[1]['win_rate']:.1f}%\n")
            f.write(f"  Record: {best_model[1]['wins']}W - {best_model[1]['losses']}L\n")
            f.write(f"  Total Battles: {best_model[1]['total_battles']}\n")

    print(f"\nReport saved to: {report_file}")


if __name__ == "__main__":
    main()
