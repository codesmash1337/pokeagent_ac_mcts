"""
Hyperparameter tuning for Neural-MCTS hybrid model.

This script helps find optimal hyperparameters for:
- c_puct (exploration constant)
- neural_weight (weight of neural prior vs MCTS prior)
- search_time_ms (time budget per move)

Usage:
    python tune_hyperparameters.py --param c_puct --values 0.5 1.0 1.5 2.0
"""

import argparse
import logging
import yaml
from pathlib import Path
from typing import List
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """Tune hyperparameters for Neural-MCTS."""

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.load_config()

    def load_config(self):
        """Load configuration file."""
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        logger.info(f"Loaded config from {self.config_path}")

    def save_config(self, output_path: str = None):
        """Save modified configuration."""
        output_path = output_path or self.config_path
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        logger.info(f"Saved config to {output_path}")

    def set_parameter(self, param_name: str, value: float):
        """Set a hyperparameter value."""
        # Navigate nested config structure
        keys = param_name.split('.')
        config = self.config

        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        config[keys[-1]] = value
        logger.info(f"Set {param_name} = {value}")

    def run_evaluation(self, battles: int = 10) -> dict:
        """Run evaluation with current config."""
        logger.info(f"Running evaluation with {battles} battles...")

        # Save current config temporarily
        temp_config = self.config_path.parent / "temp_config.yaml"
        self.save_config(temp_config)

        # TODO: Run actual evaluation
        # For now, return dummy results
        win_rate = np.random.uniform(0.4, 0.6)  # Placeholder

        return {
            "battles": battles,
            "win_rate": win_rate,
            "avg_time_per_move": 0.5,  # seconds
        }

    def grid_search(self, param_name: str, values: List[float], battles_per_config: int = 10):
        """Grid search over parameter values."""
        logger.info(f"Grid search for {param_name} with values: {values}")

        results = []

        for value in values:
            logger.info(f"\nTesting {param_name} = {value}")

            # Set parameter
            self.set_parameter(param_name, value)

            # Run evaluation
            result = self.run_evaluation(battles_per_config)
            result['param_value'] = value
            results.append(result)

            logger.info(f"  Win rate: {result['win_rate']:.3f}")

        # Find best configuration
        best_result = max(results, key=lambda x: x['win_rate'])
        logger.info(f"\nBest {param_name} = {best_result['param_value']} "
                   f"(win rate: {best_result['win_rate']:.3f})")

        return results

    def print_results(self, results: List[dict], param_name: str):
        """Print tuning results in a nice format."""
        logger.info("\n" + "="*60)
        logger.info(f"TUNING RESULTS: {param_name}")
        logger.info("="*60)
        logger.info(f"{'Value':>10s} {'Win Rate':>12s} {'Avg Time/Move':>15s}")
        logger.info("-"*60)

        for result in results:
            logger.info(f"{result['param_value']:>10.2f} "
                       f"{result['win_rate']:>12.1%} "
                       f"{result['avg_time_per_move']:>15.3f}s")

        logger.info("="*60)


def main():
    parser = argparse.ArgumentParser(description="Tune Neural-MCTS hyperparameters")
    parser.add_argument("--config", default="vendor/neural-mcts/config/neural_mcts_config.yaml",
                       help="Path to config file")
    parser.add_argument("--param", required=True,
                       help="Parameter to tune (e.g., 'neural_search.c_puct')")
    parser.add_argument("--values", nargs='+', type=float, required=True,
                       help="Values to test")
    parser.add_argument("--battles", type=int, default=10,
                       help="Battles per configuration")
    parser.add_argument("--output", help="Output file for best config")

    args = parser.parse_args()

    # Initialize tuner
    tuner = HyperparameterTuner(args.config)

    # Run grid search
    results = tuner.grid_search(args.param, args.values, args.battles)

    # Print results
    tuner.print_results(results, args.param)

    # Save best configuration
    if args.output:
        best_result = max(results, key=lambda x: x['win_rate'])
        tuner.set_parameter(args.param, best_result['param_value'])
        tuner.save_config(args.output)
        logger.info(f"\nSaved best configuration to {args.output}")


if __name__ == "__main__":
    main()
