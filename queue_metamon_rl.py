#!/usr/bin/env python3
"""Queue Metamon RL (Minikazam) agent on local ladder for 10 battles."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Setup paths
REPO_ROOT = Path(__file__).resolve().parent
VENDOR_ROOT = REPO_ROOT / "vendor" / "metamon"
if str(VENDOR_ROOT) not in sys.path:
    sys.path.append(str(VENDOR_ROOT))

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("METAMON_CACHE_DIR", str(REPO_ROOT / ".metamon_cache"))

from metamon.rl.pretrained import get_pretrained_model
from metamon.rl.evaluate import pretrained_vs_ladder


def main() -> None:
    """Run Metamon RL agent on local ladder."""
    print("="*60)
    print("Metamon RL (Minikazam) - Local Ladder Queue")
    print("="*60)
    print("Model: Minikazam (Transformer-based Actor-Critic)")
    print("Battles: 10")
    print("Format: gen9randombattle")
    print("="*60)

    # Load pretrained model
    print("\nLoading Minikazam model...")
    pm = get_pretrained_model("Minikazam")
    print("Model loaded successfully!")

    # Run on local ladder
    print("\nQueuing for ladder battles...")
    start = time.time()

    try:
        result = pretrained_vs_ladder(
            pretrained_model=pm,
            gens=[9],
            formats=["randombattle"],
            total_battles=10,
            username="MetamonRL_Eval",
            battle_backend="poke-env",
            save_trajectories_to=None,
            log_to_wandb=False,
        )

        elapsed = time.time() - start

        print("\n" + "="*60)
        print("EVALUATION COMPLETE")
        print("="*60)
        print(f"Time: {elapsed/60:.1f} minutes")
        print("\nResults:")
        for key, value in result.items():
            print(f"  {key}: {value}")
        print("="*60)

    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
