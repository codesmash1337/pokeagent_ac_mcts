#!/usr/bin/env python3
"""
Vanilla MCTS Battle Agent

Runs battles using pure MCTS (no neural priors).
"""

import asyncio
import sys
import os

# Add foul-play to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'vendor/foul-play'))

from poke_env import PlayerConfiguration, ServerConfiguration, RandomPlayer
from poke_env.player import cross_evaluate
from fp.agents.showdown_player import ShowdownPlayer
from constants import BattleType
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VanillaMCTSPlayer(ShowdownPlayer):
    """Player using vanilla MCTS (no neural priors)."""

    def __init__(self, player_configuration, server_configuration, battle_format):
        super().__init__(
            player_configuration=player_configuration,
            server_configuration=server_configuration,
            battle_format=battle_format,
            battle_type=BattleType.RANDOM_BATTLE,
            use_neural_mcts=False  # Disable neural priors
        )
        logger.info("Vanilla MCTS Player initialized (no neural priors)")


async def main():
    """Run vanilla MCTS battles."""

    # Server configuration
    server_config = ServerConfiguration(
        "localhost:8000/showdown/websocket"
    )

    # Player configuration
    player_config = PlayerConfiguration("VanillaMCTS", None)

    # Create player
    logger.info("Creating Vanilla MCTS player...")
    player = VanillaMCTSPlayer(
        player_configuration=player_config,
        server_configuration=server_config,
        battle_format="gen9randombattle"
    )

    logger.info(f"Vanilla MCTS player ready: {player.username}")
    logger.info("Waiting for battles...")

    # Keep alive
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down Vanilla MCTS player...")


if __name__ == "__main__":
    asyncio.run(main())
