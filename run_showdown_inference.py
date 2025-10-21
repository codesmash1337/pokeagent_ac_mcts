#!/usr/bin/env python3
"""Run a Metamon policy on a Showdown server using UniversalState.from_Battle.

The runner queues for battles (local ladder by default), converts the live
pokem-on battle state via ``UniversalState.from_Battle`` each turn, and feeds the
result through the Abra actor-critic policy to pick actions.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
from scipy.stats import entropy

import amago

REPO_ROOT = Path(__file__).resolve().parent
METAMON_ROOT = REPO_ROOT / "vendor" / "metamon"
TEST_UTILS_ROOT = METAMON_ROOT / "test"

for path in (METAMON_ROOT, TEST_UTILS_ROOT):
    if str(path) not in sys.path:
        sys.path.append(str(path))

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("METAMON_CACHE_DIR", str(REPO_ROOT / "metamon_cache"))
(REPO_ROOT / "metamon_cache").mkdir(exist_ok=True)

from metamon.rl.pretrained import get_pretrained_model  # type: ignore
from metamon.env.wrappers import (  # type: ignore
    PokeAgentLadder,
    QueueOnLocalLadder,
    TeamSet,
)
from metamon.interface import UniversalState  # type: ignore

from ac_inference import (  # type: ignore
    PolicyValueInference,
    get_best_action,
    sample_action,
)


def prepare_team_set(team_file: Path, battle_format: str) -> tuple[TeamSet, Path]:
    if not team_file.exists():
        raise FileNotFoundError(f"Team file not found: {team_file}")
    tmp_root = Path(tempfile.mkdtemp(prefix="showdown_team_"))
    dest_dir = tmp_root / team_file.stem
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(team_file, dest_dir / team_file.name)
    return TeamSet(str(tmp_root), battle_format), tmp_root


def copy_obs(obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {
        key: np.array(value, copy=True)
        if isinstance(value, np.ndarray)
        else np.array(value)
        for key, value in obs.items()
    }


def to_legal_actions(raw: Iterable[int]) -> List[int]:
    if raw is None:
        return []
    if isinstance(raw, np.ndarray):
        return raw.astype(int).tolist()
    return [int(x) for x in raw]


class PolicyValueTracker:
    """Tracks policy and value distributions across battles."""

    def __init__(self) -> None:
        self.battle_data: List[Dict] = []
        self.current_battle: List[Dict] = []
        self.current_battle_idx = 0

    def record_step(
        self,
        turn: int,
        action: int,
        action_probs: torch.Tensor,
        q_values: torch.Tensor,
        state_value: torch.Tensor,
        legal_actions: List[int],
    ) -> None:
        """Record statistics for a single step."""
        probs = action_probs.detach().cpu().numpy()
        qs = q_values.detach().cpu().numpy()
        v = state_value.detach().cpu().item()

        # Compute legal action probabilities for entropy calculation
        legal_probs = probs[legal_actions]
        legal_probs = legal_probs / legal_probs.sum()  # Renormalize

        step_data = {
            "turn": turn,
            "action": int(action),
            "max_prob": float(probs.max()),
            "selected_prob": float(probs[action]),
            "entropy": float(entropy(legal_probs)),
            "state_value": v,
            "selected_q_value": float(qs[action]),
            "max_q_value": float(qs.max()),
            "mean_legal_prob": float(legal_probs.mean()),
            "num_legal_actions": len(legal_actions),
            # Store top-3 actions and their probabilities
            "top3_actions": [int(i) for i in np.argsort(probs)[-3:][::-1]],
            "top3_probs": [float(p) for p in sorted(probs)[-3:][::-1]],
        }
        self.current_battle.append(step_data)

    def finish_battle(self, outcome: str, turns: int) -> None:
        """Finalize the current battle and compute summary statistics."""
        if not self.current_battle:
            return

        battle_summary = {
            "battle_id": self.current_battle_idx,
            "outcome": outcome,
            "total_turns": turns,
            "steps": self.current_battle,
            # Summary statistics
            "mean_entropy": float(np.mean([s["entropy"] for s in self.current_battle])),
            "mean_max_prob": float(
                np.mean([s["max_prob"] for s in self.current_battle])
            ),
            "value_trajectory": [s["state_value"] for s in self.current_battle],
            "value_start": self.current_battle[0]["state_value"],
            "value_end": self.current_battle[-1]["state_value"],
            "value_range": float(
                max(s["state_value"] for s in self.current_battle)
                - min(s["state_value"] for s in self.current_battle)
            ),
        }
        self.battle_data.append(battle_summary)
        self.current_battle = []
        self.current_battle_idx += 1

    def save(self, path: Path) -> None:
        """Save tracked data to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                {
                    "total_battles": len(self.battle_data),
                    "battles": self.battle_data,
                    # Overall statistics
                    "overall_mean_entropy": float(
                        np.mean([b["mean_entropy"] for b in self.battle_data])
                    ),
                    "overall_mean_max_prob": float(
                        np.mean([b["mean_max_prob"] for b in self.battle_data])
                    ),
                },
                f,
                indent=2,
            )
        print(f"Policy/value statistics saved to {path}")

    def print_summary(self) -> None:
        """Print summary statistics to console."""
        if not self.battle_data:
            print("No battle data recorded.")
            return

        print("\n=== Policy & Value Distribution Summary ===")
        print(f"Total battles: {len(self.battle_data)}")

        all_entropies = [b["mean_entropy"] for b in self.battle_data]
        all_max_probs = [b["mean_max_prob"] for b in self.battle_data]
        all_value_ranges = [b["value_range"] for b in self.battle_data]

        print(
            f"Mean entropy: {np.mean(all_entropies):.3f} ± {np.std(all_entropies):.3f}"
        )
        print(
            f"Mean max prob: {np.mean(all_max_probs):.3f} ± {np.std(all_max_probs):.3f}"
        )
        print(
            f"Mean value range per battle: {np.mean(all_value_ranges):.3f} ± {np.std(all_value_ranges):.3f}"
        )

        # Distribution of confidence (high confidence = max prob > 0.8)
        high_confidence_battles = sum(
            1 for b in self.battle_data if b["mean_max_prob"] > 0.8
        )
        print(
            f"Battles with high confidence (mean max prob > 0.8): {high_confidence_battles}/{len(self.battle_data)}"
        )
        print("==========================================\n")


class ShowdownInferenceRunner:
    def __init__(
        self,
        *,
        model_name: str,
        checkpoint: Optional[int],
        team_set: TeamSet,
        username: str,
        password: Optional[str],
        avatar: Optional[str],
        battle_format: str,
        ladder: str,
        total_battles: Optional[int],
        battle_backend: str,
        sample_actions: bool,
        temperature: float,
        verbose: bool,
        start_timer: bool,
        print_battle_bar: bool,
        save_trajectories: Optional[Path],
        save_team_results: Optional[Path],
        save_policy_stats: Optional[Path],
        use_flash_attention: bool,
    ) -> None:
        self.model = get_pretrained_model(model_name)

        if not use_flash_attention:
            overrides = dict(self.model.gin_overrides or {})
            overrides["amago.nets.traj_encoders.TformerTrajEncoder.attention_type"] = (
                amago.nets.transformer.VanillaAttention
            )
            overrides.pop("amago.nets.transformer.FlashAttention.window_size", None)
            self.model.gin_overrides = overrides
        self.agent = self.model.initialize_agent(checkpoint=checkpoint, log=False)
        self.policy = self.agent.policy
        self.policy.eval()
        self.device = getattr(
            self.agent,
            "DEVICE",
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        self.policy.to(self.device)

        env_kwargs = dict(
            battle_format=battle_format,
            num_battles=total_battles,
            observation_space=self.model.observation_space,
            action_space=self.model.action_space,
            reward_function=self.model.reward_function,
            player_team_set=team_set,
            player_username=username,
            player_password=password,
            player_avatar=avatar,
            start_timer_on_battle_start=start_timer,
            battle_backend=battle_backend,
            print_battle_bar=print_battle_bar,
            save_trajectories_to=str(save_trajectories) if save_trajectories else None,
            save_team_results_to=str(save_team_results) if save_team_results else None,
        )

        if ladder == "local":
            self.env = QueueOnLocalLadder(**env_kwargs)
        elif ladder == "pokeagent":
            if password is None:
                raise ValueError("Password required for PokéAgent ladder")
            self.env = PokeAgentLadder(**env_kwargs)
        else:
            raise ValueError(f"Unknown ladder type: {ladder}")

        self.total_battles = total_battles or float("inf")
        self.action_dim = self.model.action_space.gym_space.n
        self.sample_actions = sample_actions
        self.temperature = temperature
        self.verbose = verbose
        self.inference = PolicyValueInference(self.policy, self.device)
        self.save_policy_stats = save_policy_stats
        self.tracker = PolicyValueTracker() if save_policy_stats else None

    def close(self) -> None:
        self.env.close()

    def run(self) -> None:
        battles_played = 0
        obs, info = self.env.reset()
        legal_actions = to_legal_actions(info.get("legal_actions"))
        if not legal_actions:
            raise RuntimeError("Environment returned no legal actions on reset")
        self.inference.reset()
        turn_in_battle = 0

        while battles_played < self.total_battles:
            state = UniversalState.from_Battle(self.env.current_battle)
            obs_for_policy = copy_obs(obs)
            action_probs, q_values, state_value = self.inference(
                obs_for_policy, legal_actions
            )
            action = self._select_action(action_probs, legal_actions)

            # Track policy and value statistics
            if self.tracker:
                self.tracker.record_step(
                    turn_in_battle,
                    action,
                    action_probs,
                    q_values,
                    state_value,
                    legal_actions,
                )

            if self.verbose:
                self._log_step(
                    battles_played, state, action, action_probs, q_values, state_value
                )

            obs, reward, terminated, truncated, info = self.env.step(action)
            done = bool(terminated or truncated)
            self.inference.update(float(reward), int(action), done)
            turn_in_battle += 1

            if done:
                battles_played += 1
                self._log_battle_result(battles_played, info)
                if battles_played >= self.total_battles:
                    break
                obs, info = self.env.reset()
                legal_actions = to_legal_actions(info.get("legal_actions"))
                if not legal_actions:
                    raise RuntimeError(
                        "Environment returned no legal actions after reset"
                    )
                self.inference.reset()
                turn_in_battle = 0
                continue

            legal_actions = to_legal_actions(info.get("legal_actions"))
            if not legal_actions:
                legal_actions = list(range(self.action_dim))

        # Save tracker data at the end
        if self.tracker:
            self.tracker.print_summary()
            self.tracker.save(self.save_policy_stats)

    def _select_action(
        self, action_probs: torch.Tensor, legal_actions: List[int]
    ) -> int:
        if self.sample_actions:
            return sample_action(action_probs.cpu(), self.temperature, legal_actions)
        return get_best_action(action_probs.cpu(), legal_actions)

    def _log_step(
        self,
        battle_idx: int,
        state: UniversalState,
        action: int,
        action_probs: torch.Tensor,
        q_values: torch.Tensor,
        state_value: torch.Tensor,
    ) -> None:
        player = state.player_active_pokemon.name
        opponent = state.opponent_active_pokemon.name
        probs = action_probs.detach().cpu().numpy()
        header = (
            f"[Battle {battle_idx + 1}] {player} vs {opponent}\n"
            f"    action={action} prob={action_probs[action].item():.3f}"
            f" q={q_values[action].item():.3f} v={state_value.item():.3f}"
        )
        prob_lines = [f"    [{idx:02d}] {val:.3f}" for idx, val in enumerate(probs)]
        print("\n".join([header, "    probs:", *prob_lines]))

    def _log_battle_result(self, battle_idx: int, info: Dict[str, object]) -> None:
        outcome = "?"
        if "won" in info:
            outcome = "WIN" if info["won"] == 1 else "LOSS"
        turns = info.get("turns", self.env.turn_counter)
        print(f"Battle {battle_idx} finished: {outcome} in {turns} turns")

        # Finalize battle tracking
        if self.tracker:
            self.tracker.finish_battle(outcome, turns)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Abra inference on Showdown")
    parser.add_argument("--model", default="Abra", help="Pretrained model name")
    parser.add_argument("--checkpoint", type=int, help="Checkpoint index")
    parser.add_argument("--username", required=True, help="Showdown username")
    parser.add_argument(
        "--password", help="Showdown password (required for PokéAgent ladder)"
    )
    parser.add_argument("--avatar", help="Optional Showdown avatar")
    parser.add_argument(
        "--team-file", type=Path, required=True, help="Path to team file (.gen9ou_team)"
    )
    parser.add_argument("--format", default="gen9ou", help="Battle format")
    parser.add_argument(
        "--ladder",
        choices=["local", "pokeagent"],
        default="local",
        help="Which ladder to queue on",
    )
    parser.add_argument(
        "--battles", type=int, default=1, help="Number of battles to play"
    )
    parser.add_argument(
        "--battle-backend",
        choices=["metamon", "poke-env"],
        default="metamon",
        help="Battle parsing backend",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Sample actions instead of greedy selection",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print per-turn debug info"
    )
    parser.add_argument(
        "--no-timer",
        action="store_true",
        help="Do not start Showdown timer on battle start",
    )
    parser.add_argument(
        "--no-battle-bar",
        action="store_true",
        help="Disable ASCII battle bar rendering from poke-env",
    )
    parser.add_argument(
        "--save-trajectories", type=Path, help="Directory to store trajectories"
    )
    parser.add_argument(
        "--save-team-results", type=Path, help="Directory to store team result CSVs"
    )
    parser.add_argument(
        "--save-policy-stats",
        type=Path,
        help="JSON file to save policy and value distribution statistics",
    )
    parser.add_argument(
        "--flash-attention",
        action="store_true",
        help="Use FlashAttention if the optional dependency is installed",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    team_set, tmp_dir = prepare_team_set(args.team_file, args.format)
    try:
        runner = ShowdownInferenceRunner(
            model_name=args.model,
            checkpoint=args.checkpoint,
            team_set=team_set,
            username=args.username,
            password=args.password,
            avatar=args.avatar,
            battle_format=args.format,
            ladder=args.ladder,
            total_battles=args.battles,
            battle_backend=args.battle_backend,
            sample_actions=args.sample,
            temperature=args.temperature,
            verbose=args.verbose,
            start_timer=not args.no_timer,
            print_battle_bar=not args.no_battle_bar,
            save_trajectories=args.save_trajectories,
            save_team_results=args.save_team_results,
            save_policy_stats=args.save_policy_stats,
            use_flash_attention=args.flash_attention,
        )
        try:
            runner.run()
        finally:
            runner.close()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
