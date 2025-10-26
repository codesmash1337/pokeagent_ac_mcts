"""Neural inference utilities bridging poke-engine states and Metamon policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING

import numpy as np
import torch

from poke_engine import State as PokeEngineState

from metamon.interface import (
    UniversalAction,
    consistent_move_order,
    consistent_pokemon_order,
)
from metamon.stateful_inference import (
    get_policy_and_value as ac_get_policy_and_value,
    init_inference_inputs as ac_init_inference_inputs,
    prepare_observation as ac_prepare_observation,
    reset_hidden_state_if_done as ac_reset_hidden_state_if_done,
    update_rl2_features as ac_update_rl2_features,
    update_time_index as ac_update_time_index,
)

if TYPE_CHECKING:
    from metamon.interface import ObservationSpace, UniversalState


DEBUG_PRINTS = False


def _debug_print(*args, **kwargs):
    if DEBUG_PRINTS:
        print(*args, **kwargs)


@dataclass
class InferenceResult:
    """Container for the outputs of a neural inference step."""

    universal_state: "UniversalState"
    observation: Dict[str, Any]
    legal_actions: List[int]
    action_probs: torch.Tensor
    q_values: torch.Tensor
    state_value: torch.Tensor
    policy_prior: List[float]


def _prepare_observation(
    obs: Dict[str, np.ndarray],
    legal_actions: List[int],
    num_actions: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    sanitized_legal_actions = [idx for idx in legal_actions if 0 <= idx < num_actions]
    for key, value in obs.items():
        if value.dtype.kind in {"U", "S", "O"}:
            raise ValueError(
                f"Observation field '{key}' is textual; use a tokenized observation space before inference."
            )
    return ac_prepare_observation(
        obs,
        sanitized_legal_actions,
        num_actions,
        device,
    )


_init_inference_inputs = ac_init_inference_inputs
_update_rl2_features = ac_update_rl2_features
_update_time_index = ac_update_time_index
_reset_hidden_state_if_done = ac_reset_hidden_state_if_done
_get_policy_and_value = ac_get_policy_and_value


class DefaultPolicyValueInference:
    def __init__(self, policy: Any, device: torch.device) -> None:
        self.policy = policy
        self.device = device
        self.rl2s, self.time_idxs, self.hidden_state = _init_inference_inputs(
            batch_size=1, device=device, policy=policy
        )
        self.step = 0

    def reset(self) -> None:
        self.rl2s, self.time_idxs, self.hidden_state = _init_inference_inputs(
            batch_size=1, device=self.device, policy=self.policy
        )
        self.step = 0

    def __call__(
        self,
        obs: Dict[str, np.ndarray],
        legal_actions: List[int],
        gamma_idx: int = -1,
    ):
        obs_torch = _prepare_observation(
            obs,
            legal_actions,
            getattr(self.policy, "action_dim", len(legal_actions)),
            self.device,
        )
        action_probs, q_values, state_value, _, self.hidden_state = (
            _get_policy_and_value(
                self.policy,
                obs_torch,
                self.rl2s,
                self.time_idxs,
                self.hidden_state,
                gamma_idx=gamma_idx,
            )
        )
        return action_probs, q_values, state_value

    def update(self, reward: float, action: int, done: bool) -> None:
        self.rl2s = _update_rl2_features(self.rl2s, reward, action, done)
        self.step += 1
        self.time_idxs = _update_time_index(self.time_idxs, self.step)
        self.hidden_state = _reset_hidden_state_if_done(
            self.policy, self.hidden_state, done
        )


class NeuralInferenceRunner:
    """
    Run neural policy/value inference on poke-engine states using Metamon models.

    The runner converts a :class:`poke_engine.State` into Metamon's ``UniversalState``
    via ``poke_engine_adapter``, generates the appropriate observation tensor, and
    evaluates an actor-critic policy (e.g., Abra) to obtain action probabilities and
    Q-values.
    """

    def __init__(
        self,
        *,
        policy: Any,
        observation_space: "ObservationSpace",
        device: torch.device,
        action_space: Optional[Any] = None,
        inference: Optional[Any] = None,
    ) -> None:
        self.policy = policy
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device
        self.action_dim = getattr(policy, "action_dim", 13)
        self._inference = inference or DefaultPolicyValueInference(policy, device)
        self.observation_space.reset()

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "Abra",
        *,
        checkpoint: Optional[int] = None,
        log: bool = False,
    ) -> "NeuralInferenceRunner":
        """Construct a runner from a registered Metamon pretrained model."""

        from metamon.rl.pretrained import get_pretrained_model

        pretrained = get_pretrained_model(model_name)
        experiment = pretrained.initialize_agent(checkpoint=checkpoint, log=log)
        policy = experiment.policy
        policy.eval()
        preferred_device = getattr(experiment, "DEVICE", torch.device("cpu"))
        if torch.backends.mps.is_available():
            preferred_device = torch.device("mps")
        policy = policy.to(preferred_device)
        return cls(
            policy=policy,
            observation_space=pretrained.observation_space,
            device=preferred_device,
            action_space=pretrained.action_space,
        )

    def reset(self) -> None:
        """Reset internal observation/inference state (call between battles)."""

        self.observation_space.reset()
        if hasattr(self._inference, "reset"):
            self._inference.reset()

    def update(self, reward: float, action: int, done: bool) -> None:
        """Update RL2 features inside the inference wrapper after a step."""

        if hasattr(self._inference, "update"):
            self._inference.update(reward=reward, action=action, done=done)

    def infer(
        self,
        state: PokeEngineState,
        *,
        battle_format: str,
        perspective: str = "side_one",
        gamma_idx: int = -1,
        legal_actions: Optional[Sequence[int]] = None,
    ) -> InferenceResult:
        """Run neural inference on a poke-engine state."""

        universal_state = _state_to_universal(
            state,
            battle_format=battle_format,
            perspective=perspective,
        )

        # Get the original order from poke-engine state
        if perspective == "side_one":
            current_side = state.side_one
        else:
            current_side = state.side_two

        active_index = int(current_side.active_index)

        # Original move order (M0, M1, M2, M3 as they appear in poke-engine)
        original_moves = [move for move in current_side.pokemon[active_index].moves]

        # Original pokemon order (all pokemon in team order, excluding active)
        all_pokemon = list(current_side.pokemon)
        original_switches = [
            pkmn
            for i, pkmn in enumerate(all_pokemon)
            if i != active_index and pkmn.hp > 0
        ]

        # Print the original unsorted move order
        _debug_print("\n" + "=" * 68)
        _debug_print("ORIGINAL POKE-ENGINE MOVE ORDER (unsorted)")
        _debug_print("=" * 68)
        _debug_print(f"Current side is {perspective}")
        _debug_print(f"Current active index is {active_index}")
        _debug_print(f"Current pokemon ordering is {[p.id for p in all_pokemon]}")
        for idx, move in enumerate(original_moves):
            _debug_print(f"  M{idx}: {move.id}")
        if original_switches:
            _debug_print("\nOriginal available switches (team order):")
            for idx, pkmn in enumerate(original_switches):
                _debug_print(f"  P{idx}: {pkmn.id}")
        _debug_print("-" * 50)
        _debug_print("universal state")
        if DEBUG_PRINTS:
            from pprint import pprint

            pprint(universal_state.to_dict(), indent=2)
        _debug_print("=" * 68 + "\n")

        # Get consistent (alphabetical) order using metamon's functions
        # Convert to UniversalMove/UniversalPokemon for sorting
        from metamon.poke_engine_adapter import (
            _universal_move_from_pe,
            _universal_pokemon_from_pe,
            _dex_for_format,
            _side_boosts,
        )

        dex = _dex_for_format(battle_format)
        universal_moves = [
            _universal_move_from_pe(m, dex)
            for m in original_moves
            if m.id not in {"", "none"}
        ]
        sorted_moves = consistent_move_order(universal_moves)

        # Create mapping: original_index -> consistent_index
        # This tells us: "move at original position X goes to consistent position Y"
        move_original_to_consistent = {}
        move_consistent_to_original = {}
        for orig_idx, orig_move in enumerate(universal_moves):
            for cons_idx, cons_move in enumerate(sorted_moves):
                if orig_move.name == cons_move.name:
                    move_original_to_consistent[orig_idx] = cons_idx
                    move_consistent_to_original[cons_idx] = orig_idx
                    break

        # Create switch mapping (original position -> consistent position)
        if original_switches:
            boosts = _side_boosts(current_side)
            universal_switches = [
                _universal_pokemon_from_pe(
                    pkmn,
                    dex=dex,
                    is_active=False,
                    boost_tuple=boosts,
                    side=current_side,
                )
                for pkmn in original_switches
            ]
            sorted_switches = consistent_pokemon_order(universal_switches)

            switch_original_to_consistent = {}
            switch_consistent_to_original = {}
            for orig_idx, orig_switch in enumerate(universal_switches):
                for cons_idx, cons_switch in enumerate(sorted_switches):
                    if orig_switch.name == cons_switch.name:
                        switch_original_to_consistent[orig_idx] = cons_idx
                        switch_consistent_to_original[cons_idx] = orig_idx
                        break
        else:
            switch_consistent_to_original = {}

        observation = self.observation_space.state_to_obs(universal_state)

        if legal_actions is None:
            legal_actions = sorted(
                action.action_idx
                for action in UniversalAction.maybe_valid_actions(universal_state)
            )
        else:
            legal_actions = list(legal_actions)

        if not legal_actions:
            raise ValueError("No legal actions available for inference.")

        action_probs, q_values, state_value = self._inference(
            observation,
            legal_actions,
            gamma_idx=gamma_idx,
        )

        # Policy prior is in consistent (alphabetical) order - need to remap to original order
        policy_prior_consistent = action_probs.detach().float().cpu().tolist()
        policy_prior_original = [0.0] * len(policy_prior_consistent)

        # Remap the policy priors from consistent order to original order
        for action_idx in range(len(policy_prior_consistent)):
            if action_idx < 4:
                # Regular moves (0-3): remap from consistent to original
                if action_idx in move_consistent_to_original:
                    original_idx = move_consistent_to_original[action_idx]
                    policy_prior_original[original_idx] = policy_prior_consistent[
                        action_idx
                    ]
                else:
                    policy_prior_original[action_idx] = policy_prior_consistent[
                        action_idx
                    ]
            elif action_idx < 9:
                # Switches (4-8): remap from consistent to original
                switch_idx = action_idx - 4
                if switch_idx in switch_consistent_to_original:
                    original_switch_idx = switch_consistent_to_original[switch_idx]
                    original_action_idx = 4 + original_switch_idx
                    policy_prior_original[original_action_idx] = (
                        policy_prior_consistent[action_idx]
                    )
                else:
                    policy_prior_original[action_idx] = policy_prior_consistent[
                        action_idx
                    ]
            elif action_idx < 13:
                # Tera moves (9-12): same mapping as regular moves
                move_idx = action_idx - 9
                if move_idx in move_consistent_to_original:
                    original_idx = move_consistent_to_original[move_idx]
                    original_action_idx = 9 + original_idx
                    policy_prior_original[original_action_idx] = (
                        policy_prior_consistent[action_idx]
                    )
                else:
                    policy_prior_original[action_idx] = policy_prior_consistent[
                        action_idx
                    ]
            else:
                # Beyond action space, keep as is
                policy_prior_original[action_idx] = policy_prior_consistent[action_idx]

        # Print what the policy network expects (highest prior in original order)
        if DEBUG_PRINTS and policy_prior_original:
            max_idx = max(
                range(len(policy_prior_original)),
                key=lambda i: policy_prior_original[i],
            )
            max_prob = policy_prior_original[max_idx]

            # Get the move name
            move_name = "Unknown"
            if max_idx < len(original_moves):
                # Regular move
                move_name = f"{original_moves[max_idx].id} (M{max_idx})"
            elif max_idx >= 4 and max_idx < 4 + len(original_switches):
                # Switch
                switch_idx = max_idx - 4
                if switch_idx < len(original_switches):
                    move_name = (
                        f"Switch to {original_switches[switch_idx].id} (P{switch_idx})"
                    )
            elif max_idx >= 9 and max_idx < 13:
                # Tera move
                tera_move_idx = max_idx - 9
                if tera_move_idx < len(original_moves):
                    move_name = (
                        f"{original_moves[tera_move_idx].id} + Tera (M{tera_move_idx})"
                    )

            _debug_print(
                "╔════════════════════════════════════════════════════════════════╗"
            )
            _debug_print(
                "║ POLICY NETWORK EXPECTS (highest prior)                        ║"
            )
            _debug_print(
                "╠════════════════════════════════════════════════════════════════╣"
            )
            _debug_print(f"║ Move: {move_name:48} ║")
            _debug_print(
                f"║ Index: {max_idx:2}  Prior: {max_prob:.3f}                                      ║"
            )
            _debug_print(
                "╚════════════════════════════════════════════════════════════════╝"
            )

            # Also print top 3 for comparison
            indexed_priors = [
                (i, p) for i, p in enumerate(policy_prior_original) if p > 0.001
            ]
            indexed_priors.sort(key=lambda x: x[1], reverse=True)
            _debug_print("\nTop 3 policy network predictions (original order):")
            for rank, (idx, prob) in enumerate(indexed_priors[:3], 1):
                name = "Unknown"
                if idx < len(original_moves):
                    name = f"{original_moves[idx].id} (M{idx})"
                elif idx >= 4 and idx < 4 + len(original_switches):
                    switch_idx = idx - 4
                    if switch_idx < len(original_switches):
                        name = f"Switch to {original_switches[switch_idx].id}"
                elif idx >= 9 and idx < 13:
                    tera_move_idx = idx - 9
                    if tera_move_idx < len(original_moves):
                        name = f"{original_moves[tera_move_idx].id} + Tera"
                _debug_print(f"  {rank}. {name:40} | idx {idx:2} | prior: {prob:.3f}")

        return InferenceResult(
            universal_state=universal_state,
            observation=observation,
            legal_actions=legal_actions,
            action_probs=action_probs,
            q_values=q_values,
            state_value=state_value,
            policy_prior=policy_prior_original,  # Return remapped to original order
        )


def _state_to_universal(
    state: PokeEngineState,
    *,
    battle_format: str,
    perspective: str,
):
    from metamon.poke_engine_adapter import poke_engine_state_to_universal_state

    return poke_engine_state_to_universal_state(
        state,
        battle_format=battle_format,
        perspective=perspective,
    )
