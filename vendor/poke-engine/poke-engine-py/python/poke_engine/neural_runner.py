"""Neural inference utilities bridging poke-engine states and Metamon policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    TYPE_CHECKING,
    Union,
    Tuple,
)

import numpy as np
import torch
import time
import sys
from poke_engine import (
    State as PokeEngineState,
    prepare_inference_payload,
)

from metamon.backend.replay_parser.str_parsing import move_name, pokemon_name
from metamon.stateful_inference import (
    get_policy_and_value as ac_get_policy_and_value,
    get_policy_and_value_batch as ac_get_policy_and_value_batch,
    init_inference_inputs as ac_init_inference_inputs,
    prepare_observation as ac_prepare_observation,
    prepare_observation_batch as ac_prepare_observation_batch,
    reset_hidden_state_if_done as ac_reset_hidden_state_if_done,
    update_rl2_features as ac_update_rl2_features,
    update_time_index as ac_update_time_index,
    DEFAULT_TARGET_ENTROPY_RATIO,
    DEFAULT_ADAPT_STRENGTH,
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

    universal_state: Optional["UniversalState"]
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


def _prepare_observation_batch(
    obs_list: List[Dict[str, np.ndarray]],
    legal_actions_list: List[List[int]],
    num_actions: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    return ac_prepare_observation_batch(
        obs_list,
        legal_actions_list,
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
                target_entropy_ratio=DEFAULT_TARGET_ENTROPY_RATIO,
                adapt_strength=DEFAULT_ADAPT_STRENGTH,
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
        observation_space: Optional["ObservationSpace"] = None,
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
        if self.observation_space is not None:
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

        if self.observation_space is not None:
            self.observation_space.reset()
        if hasattr(self._inference, "reset"):
            self._inference.reset()

    def update(self, reward: float, action: int, done: bool) -> None:
        """Forward RL2 feedback to the underlying inference helper, if available."""

        if hasattr(self._inference, "update"):
            self._inference.update(reward, action, done)

    def infer(
        self,
        state: Union[PokeEngineState, str],
        *,
        battle_format: str,
        perspective: str = "side_one",
        legal_actions: Optional[Sequence[int]] = None,
        gamma_idx: int = -1,
    ) -> InferenceResult:
        sys.stderr.write("[NEURAL_DEBUG] path=infer (single state)\n")
        sys.stderr.flush()
        prepped_state = self._prepare_single_state(
            self._ensure_state(state),
            battle_format=battle_format,
            perspective=perspective,
            legal_actions=legal_actions,
        )
        action_probs, q_values, state_value = self._inference(
            prepped_state["observation"],
            prepped_state["legal_actions"],
            gamma_idx=gamma_idx,
        )
        # Return raw policy in canonical 13-action order (0-3 moves, 4-8 switches, 9-12 tera moves)
        policy_prior = action_probs.detach().float().cpu().tolist()
        if DEBUG_PRINTS:
            pass

        return InferenceResult(
            universal_state=prepped_state["universal_state"],
            observation=prepped_state["observation"],
            legal_actions=list(prepped_state["legal_actions"]),
            action_probs=action_probs,
            q_values=q_values,
            state_value=state_value,
            policy_prior=policy_prior,
        )

    def infer_batch(
        self,
        states: Sequence[Union[PokeEngineState, str]],
        *,
        battle_format: str,
        perspective: str = "side_one",
        gamma_idx: int = -1,
    ) -> List[InferenceResult]:
        sys.stderr.write("[NEURAL_DEBUG] path=infer_batch (slow batch)\n")
        sys.stderr.flush()
        if not states:
            return []

        parsed_states = [self._ensure_state(state) for state in states]

        preps = []
        for st in parsed_states:
            prep = self._prepare_single_state(
                st,
                battle_format=battle_format,
                perspective=perspective,
            )
            preps.append(prep)

        obs_list = [prep["observation"] for prep in preps]
        legal_list = [prep["legal_actions"] for prep in preps]
        obs_torch = _prepare_observation_batch(
            obs_list,
            legal_list,
            self.action_dim,
            self.device,
        )

        rl2s, time_idxs, hidden_state = _init_inference_inputs(
            batch_size=len(preps),
            device=self.device,
            policy=self.policy,
        )

        (
            batch_action_probs,
            batch_q_values,
            batch_state_values,
            _,
            _,
        ) = ac_get_policy_and_value_batch(
            self.policy,
            obs_torch,
            rl2s,
            time_idxs,
            hidden_state,
            gamma_idx=gamma_idx,
            target_entropy_ratio=DEFAULT_TARGET_ENTROPY_RATIO,
            adapt_strength=DEFAULT_ADAPT_STRENGTH,
        )

        # Batch GPU→CPU transfer
        batch_action_probs_cpu = batch_action_probs.detach().float().cpu().numpy()

        # Pre-allocate results list for better performance
        results: List[InferenceResult] = []

        for idx, prep in enumerate(preps):
            action_probs = batch_action_probs[idx]
            # Return raw policy in canonical 13-action order
            policy_prior = batch_action_probs_cpu[idx].tolist()
            results.append(
                InferenceResult(
                    universal_state=prep["universal_state"],
                    observation=prep["observation"],
                    legal_actions=prep[
                        "legal_actions"
                    ],  # Already a list, no need to copy
                    action_probs=action_probs,
                    q_values=batch_q_values[idx],
                    state_value=batch_state_values[idx],
                    policy_prior=policy_prior,
                )
            )

        return results

    def infer_from_payload_batch(
        self,
        tokens_np: np.ndarray,
        numbers_np: np.ndarray,
        legal_actions_list: Sequence[Sequence[int]],
        move_mappings_list: Sequence[Mapping[int, int]],
        switch_mappings_list: Sequence[Mapping[int, int]],
        *,
        battle_format: str,
        gamma_idx: int = -1,
    ) -> List[InferenceResult]:
        sys.stderr.write("[NEURAL_DEBUG] path=infer_from_payload_batch (fast batch)\n")
        sys.stderr.flush()
        _ = battle_format  # unused in this fast path but kept for symmetry

        tokens_np = np.asarray(tokens_np)
        numbers_np = np.asarray(numbers_np)

        if tokens_np.ndim != 2:
            raise ValueError(
                f"tokens_np must be a 2D array, got shape {tokens_np.shape}"
            )
        if numbers_np.ndim != 2:
            raise ValueError(
                f"numbers_np must be a 2D array, got shape {numbers_np.shape}"
            )

        batch_size = int(tokens_np.shape[0])
        if numbers_np.shape[0] != batch_size:
            raise ValueError(
                "tokens_np and numbers_np must share the same batch dimension"
            )
        if not (
            len(legal_actions_list)
            == len(move_mappings_list)
            == len(switch_mappings_list)
            == batch_size
        ):
            raise ValueError(
                "Metadata sequences must match the batch dimension of the payload"
            )

        obs_list: List[Dict[str, np.ndarray]] = []
        for idx in range(batch_size):
            obs_list.append(
                {
                    "text_tokens": np.array(tokens_np[idx], copy=False),
                    "numbers": np.array(numbers_np[idx], copy=False),
                }
            )

        legal_actions_sanitized = [
            [int(action) for action in actions] for actions in legal_actions_list
        ]
        move_mappings = [dict(mapping) for mapping in move_mappings_list]
        switch_mappings = [dict(mapping) for mapping in switch_mappings_list]

        obs_torch = _prepare_observation_batch(
            obs_list,
            legal_actions_sanitized,
            self.action_dim,
            self.device,
        )

        rl2s, time_idxs, hidden_state = _init_inference_inputs(
            batch_size=batch_size,
            device=self.device,
            policy=self.policy,
        )

        (
            batch_action_probs,
            batch_q_values,
            batch_state_values,
            _,
            _,
        ) = ac_get_policy_and_value_batch(
            self.policy,
            obs_torch,
            rl2s,
            time_idxs,
            hidden_state,
            gamma_idx=gamma_idx,
            target_entropy_ratio=DEFAULT_TARGET_ENTROPY_RATIO,
            adapt_strength=DEFAULT_ADAPT_STRENGTH,
        )

        batch_action_probs_cpu = batch_action_probs.detach().float().cpu().numpy()

        results: List[InferenceResult] = []
        for idx in range(batch_size):
            policy_prior = batch_action_probs_cpu[idx].tolist()
            results.append(
                InferenceResult(
                    universal_state=None,
                    observation=obs_list[idx],
                    legal_actions=legal_actions_sanitized[idx],
                    action_probs=batch_action_probs[idx],
                    q_values=batch_q_values[idx],
                    state_value=batch_state_values[idx],
                    policy_prior=policy_prior,
                )
            )

        return results

    def _prepare_single_state(
        self,
        state: PokeEngineState,
        *,
        battle_format: str,
        perspective: str,
        legal_actions: Optional[Sequence[int]] = None,
        return_timings: bool = False,
    ) -> Union[Dict[str, Any], Tuple[Dict[str, Any], Dict[str, float]]]:
        t0 = time.perf_counter() if return_timings else None

        perspective = perspective.lower()
        if perspective not in {"side_one", "side_two"}:
            raise ValueError("perspective must be 'side_one' or 'side_two'")

        payload = prepare_inference_payload(
            state,
            perspective=perspective,
            battle_format=battle_format,
        )

        t1 = time.perf_counter() if return_timings else None

        if legal_actions is None:
            legal_actions_list = [int(action) for action in payload["legal_actions"]]
        else:
            legal_actions_list = [int(action) for action in legal_actions]

        text_tokens = np.asarray(payload["text_tokens"], copy=False)
        numbers = np.asarray(payload["numbers"], copy=False)
        observation = {"text_tokens": text_tokens, "numbers": numbers}

        t2 = time.perf_counter() if return_timings else None

        move_mapping = dict(payload["move_mapping"])
        switch_mapping = dict(payload["switch_mapping"])
        original_moves: List[Any] = []
        original_switches: List[Any] = []

        t3 = time.perf_counter() if return_timings else None

        result = {
            "universal_state": None,
            "observation": observation,
            "legal_actions": legal_actions_list,
            "original_moves": original_moves,
            "original_switches": original_switches,
            "move_mapping": move_mapping,
            "switch_mapping": switch_mapping,
        }

        if return_timings:
            timings = {
                "rust_prep": (t1 - t0) * 1000,
                "obs_prep": (t2 - t1) * 1000,
                "metadata_prep": (t3 - t2) * 1000,
            }
            return result, timings

        return result

    @staticmethod
    def _ensure_state(state: Union[PokeEngineState, str]) -> PokeEngineState:
        if isinstance(state, PokeEngineState):
            return state
        if isinstance(state, str):
            return PokeEngineState.from_string(state)
        raise TypeError(
            f"state must be poke_engine.State or serialized string, got {type(state)!r}"
        )

    @staticmethod
    def _build_move_mapping(moves: Sequence[Any]) -> Dict[int, int]:
        indexed_moves = [
            (idx, move) for idx, move in enumerate(moves) if move is not None
        ]
        sorted_moves = sorted(
            indexed_moves,
            key=lambda item: (
                move_name(getattr(item[1], "id", getattr(item[1], "name", ""))),
                item[0],
            ),
        )
        return {
            consistent_idx: original_idx
            for consistent_idx, (original_idx, _) in enumerate(sorted_moves)
        }

    @staticmethod
    def _build_switch_mapping(switches: Sequence[Any]) -> Dict[int, int]:
        indexed_switches = [
            (idx, pokemon)
            for idx, pokemon in enumerate(switches)
            if pokemon is not None
        ]
        sorted_switches = sorted(
            indexed_switches,
            key=lambda item: (
                pokemon_name(getattr(item[1], "id", getattr(item[1], "name", ""))),
                item[0],
            ),
        )
        return {
            consistent_idx: original_idx
            for consistent_idx, (original_idx, _) in enumerate(sorted_switches)
        }

    @staticmethod
    def _remap_policy_prior_numpy(
        policy_prior_consistent: np.ndarray,
        move_mapping: Dict[int, int],
        switch_mapping: Dict[int, int],
    ) -> List[float]:
        """Remap policy prior using numpy array (faster than pure Python)."""
        n = len(policy_prior_consistent)
        policy_prior_original = np.zeros(n, dtype=np.float32)

        # Remap moves (0-3)
        for consistent_idx, original_idx in move_mapping.items():
            if consistent_idx < n and original_idx < n:
                policy_prior_original[original_idx] = policy_prior_consistent[
                    consistent_idx
                ]

        # Remap switches (4-8)
        for consistent_idx, original_idx in switch_mapping.items():
            action_idx = consistent_idx + 4
            target_idx = original_idx + 4
            if action_idx < n and target_idx < n:
                policy_prior_original[target_idx] = policy_prior_consistent[action_idx]

        # Remap tera moves (9-12)
        for consistent_idx, original_idx in move_mapping.items():
            action_idx = consistent_idx + 9
            target_idx = original_idx + 9
            if action_idx < n and target_idx < n:
                policy_prior_original[target_idx] = policy_prior_consistent[action_idx]

        # Copy unmapped actions (13+)
        if n > 13:
            policy_prior_original[13:] = policy_prior_consistent[13:]

        return policy_prior_original.tolist()

    @staticmethod
    def _remap_policy_prior_from_list(
        policy_prior_consistent: List[float],
        move_mapping: Dict[int, int],
        switch_mapping: Dict[int, int],
    ) -> List[float]:
        """Remap policy prior from a pre-converted list (fastest version)."""
        policy_prior_original = [0.0] * len(policy_prior_consistent)

        for action_idx, prob in enumerate(policy_prior_consistent):
            if action_idx < 4:
                original_idx = move_mapping.get(action_idx, action_idx)
                if original_idx < len(policy_prior_original):
                    policy_prior_original[original_idx] = prob
            elif action_idx < 9:
                switch_idx = action_idx - 4
                mapped_idx = switch_mapping.get(switch_idx)
                target_idx = 4 + mapped_idx if mapped_idx is not None else action_idx
                if target_idx < len(policy_prior_original):
                    policy_prior_original[target_idx] = prob
            elif action_idx < 13:
                move_idx = action_idx - 9
                mapped_idx = move_mapping.get(move_idx, move_idx)
                target_idx = 9 + mapped_idx
                if target_idx < len(policy_prior_original):
                    policy_prior_original[target_idx] = prob
            elif action_idx < len(policy_prior_original):
                policy_prior_original[action_idx] = prob

        return policy_prior_original

    @staticmethod
    def _remap_policy_prior_from_cpu(
        action_probs_cpu: torch.Tensor,
        move_mapping: Dict[int, int],
        switch_mapping: Dict[int, int],
    ) -> List[float]:
        """Remap policy prior from a CPU tensor (avoids individual GPU→CPU transfers)."""
        policy_prior_consistent = action_probs_cpu.tolist()
        policy_prior_original = [0.0] * len(policy_prior_consistent)

        for action_idx, prob in enumerate(policy_prior_consistent):
            if action_idx < 4:
                original_idx = move_mapping.get(action_idx, action_idx)
                if original_idx < len(policy_prior_original):
                    policy_prior_original[original_idx] = prob
            elif action_idx < 9:
                switch_idx = action_idx - 4
                mapped_idx = switch_mapping.get(switch_idx)
                target_idx = 4 + mapped_idx if mapped_idx is not None else action_idx
                if target_idx < len(policy_prior_original):
                    policy_prior_original[target_idx] = prob
            elif action_idx < 13:
                move_idx = action_idx - 9
                mapped_idx = move_mapping.get(move_idx, move_idx)
                target_idx = 9 + mapped_idx
                if target_idx < len(policy_prior_original):
                    policy_prior_original[target_idx] = prob
            elif action_idx < len(policy_prior_original):
                policy_prior_original[action_idx] = prob

        return policy_prior_original

    @staticmethod
    def _remap_policy_prior(
        action_probs: torch.Tensor,
        move_mapping: Dict[int, int],
        switch_mapping: Dict[int, int],
    ) -> List[float]:
        """Legacy method that does individual GPU→CPU transfer."""
        policy_prior_consistent = action_probs.detach().float().cpu().tolist()
        policy_prior_original = [0.0] * len(policy_prior_consistent)

        for action_idx, prob in enumerate(policy_prior_consistent):
            if action_idx < 4:
                original_idx = move_mapping.get(action_idx, action_idx)
                if original_idx < len(policy_prior_original):
                    policy_prior_original[original_idx] = prob
            elif action_idx < 9:
                switch_idx = action_idx - 4
                mapped_idx = switch_mapping.get(switch_idx)
                target_idx = 4 + mapped_idx if mapped_idx is not None else action_idx
                if target_idx < len(policy_prior_original):
                    policy_prior_original[target_idx] = prob
            elif action_idx < 13:
                move_idx = action_idx - 9
                mapped_idx = move_mapping.get(move_idx, move_idx)
                target_idx = 9 + mapped_idx
                if target_idx < len(policy_prior_original):
                    policy_prior_original[target_idx] = prob
            elif action_idx < len(policy_prior_original):
                policy_prior_original[action_idx] = prob

        return policy_prior_original

    def _debug_policy_prior(
        self,
        policy_prior_original: List[float],
        original_moves: Sequence[Any],
        original_switches: Sequence[Any],
    ) -> None:
        if not policy_prior_original:
            return

        max_idx = max(
            range(len(policy_prior_original)),
            key=lambda i: policy_prior_original[i],
        )
        max_prob = policy_prior_original[max_idx]

        move_name_str = "Unknown"
        if max_idx < len(original_moves):
            move_name_str = f"{original_moves[max_idx].id} (M{max_idx})"
        elif 4 <= max_idx < 4 + len(original_switches):
            switch_idx = max_idx - 4
            move_name_str = (
                f"Switch to {original_switches[switch_idx].id} (P{switch_idx})"
            )
        elif 9 <= max_idx < 9 + len(original_moves):
            tera_move_idx = max_idx - 9
            move_name_str = (
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
        _debug_print(f"║ Move: {move_name_str:48} ║")
        _debug_print(
            f"║ Index: {max_idx:2}  Prior: {max_prob:.3f}                                      ║"
        )
        _debug_print(
            "╚════════════════════════════════════════════════════════════════╝"
        )

        indexed_priors = [
            (idx, prob)
            for idx, prob in enumerate(policy_prior_original)
            if prob > 0.001
        ]
        indexed_priors.sort(key=lambda item: item[1], reverse=True)
        _debug_print("\nTop 3 policy network predictions (original order):")
        for rank, (idx, prob) in enumerate(indexed_priors[:3], 1):
            name = "Unknown"
            if idx < len(original_moves):
                name = f"{original_moves[idx].id} (M{idx})"
            elif 4 <= idx < 4 + len(original_switches):
                switch_idx = idx - 4
                name = f"Switch to {original_switches[switch_idx].id}"
            elif 9 <= idx < 9 + len(original_moves):
                tera_idx = idx - 9
                name = f"{original_moves[tera_idx].id} + Tera"
            _debug_print(f"  {rank}. {name:40} | idx {idx:2} | prior: {prob:.3f}")
