"""Neural inference utilities bridging poke-engine states and Metamon policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from poke_engine import State as PokeEngineState

from metamon.interface import (
    ObservationSpace,
    UniversalAction,
    UniversalState,
)
from metamon.poke_engine_adapter import poke_engine_state_to_universal_state


@dataclass
class InferenceResult:
    """Container for the outputs of a neural inference step."""

    universal_state: UniversalState
    observation: Dict[str, Any]
    legal_actions: List[int]
    action_probs: torch.Tensor
    q_values: torch.Tensor
    state_value: torch.Tensor


def _prepare_observation(
    obs: Dict[str, np.ndarray],
    legal_actions: List[int],
    num_actions: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    illegal_actions = np.ones(num_actions, dtype=bool)
    for idx in legal_actions:
        if 0 <= idx < num_actions:
            illegal_actions[idx] = False
    obs_with_mask = {**obs, "illegal_actions": illegal_actions}
    return {
        key: _to_torch_tensor(key, value, device)
        for key, value in obs_with_mask.items()
    }


def _to_torch_tensor(key: str, value: np.ndarray, device: torch.device) -> torch.Tensor:
    if value.dtype.kind in {"U", "S", "O"}:
        raise ValueError(
            f"Observation field '{key}' is textual; use a tokenized observation space before inference."
        )
    return torch.from_numpy(value).to(device).unsqueeze(0).unsqueeze(0)


def _init_inference_inputs(
    batch_size: int,
    device: torch.device,
    policy: Any,
):
    rl2s = torch.zeros((batch_size, 1, 14), device=device)
    time_idxs = torch.zeros((batch_size, 1, 1), dtype=torch.long, device=device)
    hidden_state = policy.traj_encoder.init_hidden_state(
        batch_size=batch_size, device=device
    )
    return rl2s, time_idxs, hidden_state


def _update_rl2_features(
    rl2s: torch.Tensor,
    reward: float,
    action: int,
    done: bool,
) -> torch.Tensor:
    rl2s = rl2s.clone()
    rl2s[0, 0, 0] = reward
    rl2s[0, 0, 1] = action
    rl2s[0, 0, 2] = float(done)
    return rl2s


def _update_time_index(time_idxs: torch.Tensor, step: int) -> torch.Tensor:
    time_idxs = time_idxs.clone()
    time_idxs[0, 0, 0] = step
    return time_idxs


def _reset_hidden_state_if_done(policy: Any, hidden_state: Any, done: bool) -> Any:
    if done:
        done_mask = np.array([True])
        hidden_state = policy.traj_encoder.reset_hidden_state(hidden_state, done_mask)
    return hidden_state


def _get_policy_and_value(
    policy: Any,
    obs_torch: Dict[str, torch.Tensor],
    rl2s: torch.Tensor,
    time_idxs: torch.Tensor,
    hidden_state: Any,
    gamma_idx: int,
):
    with torch.no_grad():
        tstep_emb = policy.tstep_encoder(obs=obs_torch, rl2s=rl2s)
        traj_emb, new_hidden_state = policy.traj_encoder(
            tstep_emb, time_idxs=time_idxs, hidden_state=hidden_state
        )

        action_dist = policy.actor(
            traj_emb,
            straight_from_obs={k: obs_torch[k] for k in policy.pass_obs_keys_to_actor},
        )
        all_action_probs = action_dist.probs
        num_actions = policy.action_dim
        num_gammas = len(policy.gammas)
        device = traj_emb.device

        all_actions = torch.eye(num_actions, device=device)
        actions_expanded = all_actions.unsqueeze(1).unsqueeze(1).unsqueeze(2)
        actions_expanded = actions_expanded.expand(num_actions, 1, 1, num_gammas, num_actions)

        all_q_values_dist = policy.critics(traj_emb, actions_expanded)
        all_q_values = policy.critics.bin_dist_to_raw_vals(all_q_values_dist)

        action_probs = all_action_probs[0, 0, gamma_idx, :]
        q_values = all_q_values[:, 0, 0, :, gamma_idx, 0].mean(dim=1)
        state_value = (action_probs * q_values).sum()

        return action_probs, q_values, state_value, new_hidden_state


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
        action_probs, q_values, state_value, self.hidden_state = _get_policy_and_value(
            self.policy,
            obs_torch,
            self.rl2s,
            self.time_idxs,
            self.hidden_state,
            gamma_idx=gamma_idx,
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
        observation_space: ObservationSpace,
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
        return cls(
            policy=policy,
            observation_space=pretrained.observation_space,
            device=experiment.DEVICE,
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

        universal_state = poke_engine_state_to_universal_state(
            state,
            battle_format=battle_format,
            perspective=perspective,
        )

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

        return InferenceResult(
            universal_state=universal_state,
            observation=observation,
            legal_actions=legal_actions,
            action_probs=action_probs,
            q_values=q_values,
            state_value=state_value,
        )
