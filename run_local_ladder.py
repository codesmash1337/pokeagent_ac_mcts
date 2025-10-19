#!/usr/bin/env python3
"""Queue an Abra agent on the PokéAgent Challenge ladder.

Usage example:
    python queue_pokeagent_ladder.py \
        --username PACYourBotName \
        --password "your-password" \
        --team-file filtered_teams/all_teams/team_0.gen9ou_team \
        --battles 20

Notes:
- The username must start with ``PAC`` and correspond to a registered account on
  http://pokeagentshowdown.com.insecure.psim.us (see the Metamon README for
  details).
- Avoid interrupting the script mid-run; leaving a match unfinished counts as a
  loss on the ladder.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

# Ensure vendor path is importable when running from repo root
REPO_ROOT = Path(__file__).resolve().parent
VENDOR_ROOT = REPO_ROOT / "vendor" / "metamon"
if str(VENDOR_ROOT) not in sys.path:
    sys.path.append(str(VENDOR_ROOT))

import os

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("METAMON_CACHE_DIR", str(REPO_ROOT / "metamon_cache"))

from metamon.rl.pretrained import get_pretrained_model  # type: ignore
from metamon.rl.evaluate import pretrained_vs_pokeagent_ladder  # type: ignore
from metamon.env.wrappers import TeamSet  # type: ignore


def prepare_team_set(team_file: Path, battle_format: str) -> tuple[TeamSet, Path]:
    if not team_file.exists():
        raise FileNotFoundError(f"Team file not found: {team_file}")
    tmp_root = Path(tempfile.mkdtemp(prefix="pokeagent_team_"))
    dest_dir = tmp_root / team_file.stem
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(team_file, dest_dir / team_file.name)
    return TeamSet(str(tmp_root), battle_format), tmp_root


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Queue an Abra agent on the PokéAgent Challenge ladder"
    )
    parser.add_argument(
        "--username",
        required=True,
        help="Registered PokéAgent ladder username (must start with PAC)",
    )
    parser.add_argument(
        "--password", required=True, help="Password for the ladder account"
    )
    parser.add_argument(
        "--team-file", type=Path, required=True, help="Path to .gen9ou_team file"
    )
    parser.add_argument(
        "--format", default="gen9ou", help="Showdown battle format (default: gen9ou)"
    )
    parser.add_argument(
        "--battles", type=int, default=10, help="Number of battles to queue"
    )
    parser.add_argument("--avatar", default=None, help="Optional Showdown avatar name")
    parser.add_argument("--checkpoint", type=int, help="Optional Abra checkpoint index")
    parser.add_argument(
        "--results-dir", type=Path, help="Optional directory to save battle logs"
    )
    parser.add_argument(
        "--log-to-wandb", action="store_true", help="Enable Weights & Biases logging"
    )
    args = parser.parse_args()

    if not args.username.startswith("PAC"):
        raise ValueError("PokéAgent ladder usernames must start with 'PAC'")

    team_set, tmp_root = prepare_team_set(args.team_file, args.format)
    try:
        pm = get_pretrained_model("Abra")
        start = time.time()
        result = pretrained_vs_pokeagent_ladder(
            pretrained_model=pm,
            username=args.username,
            password=args.password,
            battle_format=args.format,
            team_set=team_set,
            total_battles=args.battles,
            avatar=args.avatar,
            checkpoint=args.checkpoint,
            battle_backend="poke-env",
            save_trajectories_to=None,
            save_team_results_to=str(args.results_dir) if args.results_dir else None,
            log_to_wandb=args.log_to_wandb,
        )
        elapsed = time.time() - start
        print("Evaluation finished in {:.1f} minutes".format(elapsed / 60))
        print("Summary:")
        for key, value in result.items():
            print(f"  {key}: {value}")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()