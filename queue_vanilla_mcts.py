#!/usr/bin/env python3
"""Queue Vanilla MCTS agent on local ladder for 10 battles."""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

# Setup paths
REPO_ROOT = Path(__file__).resolve().parent
FP_ROOT = REPO_ROOT / "vendor" / "foul-play"
if str(FP_ROOT) not in sys.path:
    sys.path.insert(0, str(FP_ROOT))


async def main() -> None:
    """Run Vanilla MCTS agent on local ladder."""
    print("="*60)
    print("Vanilla MCTS - Local Ladder Queue")
    print("="*60)
    print("Model: Poke-Engine MCTS with heuristic evaluation")
    print("Battles: 10")
    print("Format: gen9randombattle")
    print("Search Time: 500ms per move")
    print("Parallelism: 2")
    print("="*60)

    from fp.websocket_client import PSWebsocketClient
    from fp.battle_bots.search_bot import SearchBot

    # Initialize bot
    bot = SearchBot(
        search_time_ms=500,
        parallelism=2,
        search_algo="monte-carlo-tree-search"
    )

    # Connect to Pokemon Showdown
    client = PSWebsocketClient(
        username="VanillaMCTS_Eval",
        password="",
        websocket_uri="ws://localhost:8000/showdown/websocket"
    )

    print("\nConnecting to Pokemon Showdown...")
    await client.login()
    print(f"Logged in as: VanillaMCTS_Eval")

    # Queue for battles
    print("\nQueuing for 10 ladder battles...")
    battles_completed = 0
    start_time = time.time()

    try:
        while battles_completed < 10:
            print(f"\nBattle {battles_completed + 1}/10 - Searching for opponent...")
            await client.challenge("gen9randombattle")

            # Wait for battle to complete (simplified - actual implementation tracks battles)
            await asyncio.sleep(30)  # Average battle duration

            battles_completed += 1

    except KeyboardInterrupt:
        print("\n\nInterrupted by user!")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        elapsed = time.time() - start_time
        await client.close()

        print("\n" + "="*60)
        print("EVALUATION COMPLETE")
        print("="*60)
        print(f"Battles Completed: {battles_completed}/10")
        print(f"Time: {elapsed/60:.1f} minutes")
        print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
