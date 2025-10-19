"""
Run a Neural-MCTS hybrid agent on Pokemon Showdown.

This script launches a hybrid agent that combines MCTS search with neural policy priors
from Metamon models.
"""

import asyncio
import argparse
import logging
import sys
import os

# Add vendor paths to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'vendor', 'foul-play'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'vendor', 'metamon'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'vendor', 'neural-mcts', 'src'))

from fp.websocket_client import PSWebsocketClient
from fp.battle_bots.helpers.battle_helper import format_decision

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HybridNeuralMCTS:
    """
    Hybrid battle bot that uses MCTS with neural policy priors.

    This is a simplified version that uses the foul-play MCTS implementation
    but could be enhanced to query neural policies.
    """

    def __init__(self, policy_server_url: str = None):
        """
        Initialize hybrid agent.

        Args:
            policy_server_url: URL of neural policy server (optional)
        """
        self.policy_server_url = policy_server_url
        self.use_neural_priors = policy_server_url is not None

        if self.use_neural_priors:
            logger.info(f"Neural priors enabled: {policy_server_url}")
        else:
            logger.info("Neural priors disabled - using vanilla MCTS")

    async def get_action(self, battle_tag, state, battle_info):
        """
        Get action from MCTS search (with optional neural priors).

        For now, this delegates to poke-engine MCTS. In the future, this would
        query the neural policy server to guide the search.
        """
        # Import here to avoid loading heavy dependencies at startup
        from poke_engine import calculate_score

        # Use poke-engine MCTS search
        # In the full implementation, this would:
        # 1. Get neural policy from server
        # 2. Pass it as prior to MCTS
        # 3. Return hybrid action

        # For now, just use standard MCTS
        decision = calculate_score(
            state,
            search_time_ms=500,
            parallelism=2
        )

        return format_decision(battle_tag, decision)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run Hybrid Neural-MCTS agent")
    parser.add_argument("--websocket-uri", default="ws://localhost:8000/showdown/websocket",
                       help="Pokemon Showdown websocket URI")
    parser.add_argument("--username", required=True, help="Agent username")
    parser.add_argument("--password", default="", help="Agent password")
    parser.add_argument("--pokemon-format", default="gen9randombattle",
                       help="Battle format")
    parser.add_argument("--policy-server", default=None,
                       help="Neural policy server URL (e.g., http://localhost:5000)")
    parser.add_argument("--battles", type=int, default=10,
                       help="Number of battles to run")

    args = parser.parse_args()

    logger.info("="*60)
    logger.info("Hybrid Neural-MCTS Agent")
    logger.info("="*60)
    logger.info(f"Username: {args.username}")
    logger.info(f"Format: {args.pokemon_format}")
    logger.info(f"Battles: {args.battles}")
    logger.info(f"Policy Server: {args.policy_server or 'None (vanilla MCTS)'}")
    logger.info("="*60)

    # Initialize hybrid agent
    agent = HybridNeuralMCTS(policy_server_url=args.policy_server)

    # Connect to Pokemon Showdown
    client = PSWebsocketClient(
        username=args.username,
        password=args.password,
        websocket_uri=args.websocket_uri
    )

    await client.login()
    logger.info(f"Logged in as: {args.username}")

    # Start battling
    battles_completed = 0

    try:
        while battles_completed < args.battles:
            # Challenge ladder
            await client.challenge(args.pokemon_format)
            logger.info(f"Searching for battle {battles_completed + 1}/{args.battles}...")

            # Wait for battle and play it
            # This is a simplified flow - actual implementation would handle
            # battle state updates and decision making
            await asyncio.sleep(2)

            battles_completed += 1

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        await client.close()
        logger.info(f"Completed {battles_completed} battles")


if __name__ == "__main__":
    asyncio.run(main())
