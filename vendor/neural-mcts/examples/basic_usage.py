"""
Basic usage example for Neural-MCTS hybrid model.

This script demonstrates:
1. Starting a policy server
2. Querying the policy for a battle state
3. Running a simple MCTS search with neural priors
"""

import logging
import numpy as np
from neural_mcts.state_translator import StateTranslator
from neural_mcts.policy_client import PolicyClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Basic usage example."""

    # Initialize state translator
    logger.info("Initializing state translator...")
    translator = StateTranslator(observation_space_type="TeamPreviewObservationSpace")

    # Initialize policy client (assumes policy server is running on port 5000)
    logger.info("Connecting to policy server...")
    client = PolicyClient(server_url="http://localhost:5000")

    # Check if server is healthy
    if not client.health_check():
        logger.error("Policy server is not available!")
        logger.error("Start the server with: ./scripts/start_policy_server.sh")
        return

    logger.info("Policy server is healthy!")

    # Example state string (simplified)
    # In production, this would come from poke-engine
    example_state = "p1a: Pikachu, 80/100|p1: Charizard, 100/100|p2a: Blastoise, 90/100"

    # Legal actions for this state
    # 0-3: moves, 4-8: switches
    legal_actions = [0, 1, 2, 3, 4]

    # Translate state to observation
    logger.info("Translating state...")
    obs = translator.translate(example_state, legal_actions)
    logger.info(f"Observation keys: {obs.keys()}")
    logger.info(f"Legal actions mask: {obs['legal_actions']}")

    # Query neural policy
    logger.info("Querying neural policy...")
    policy = client.get_policy(example_state, legal_actions)

    if policy is not None:
        logger.info("Neural policy obtained successfully!")
        logger.info(f"Policy shape: {policy.shape}")
        logger.info(f"Policy probabilities: {policy}")

        # Get the recommended action
        best_action = np.argmax(policy)
        logger.info(f"Recommended action: {best_action} (probability: {policy[best_action]:.3f})")

        # Show top 3 actions
        top_actions = np.argsort(policy)[::-1][:3]
        logger.info("Top 3 actions:")
        for i, action in enumerate(top_actions, 1):
            logger.info(f"  {i}. Action {action}: {policy[action]:.3f}")
    else:
        logger.error("Failed to get policy from server")

    # Get client statistics
    stats = client.get_stats()
    logger.info(f"Client statistics: {stats}")

    logger.info("Example complete!")


if __name__ == "__main__":
    main()
