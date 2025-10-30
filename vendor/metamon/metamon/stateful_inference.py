"""Shared stateful inference helpers for AMAGO/Abra-style policies."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch

__all__ = [
    "prepare_observation",
    "prepare_observation_batch",
    "init_inference_inputs",
    "update_rl2_features",
    "update_time_index",
    "reset_hidden_state_if_done",
    "get_policy_and_value",
    "get_policy_and_value_batch",
]


def prepare_observation(
    obs: Dict[str, np.ndarray],
    legal_actions: list,
    num_actions: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Convert numpy observations into batched tensors with an illegal-action mask."""

    illegal_actions = np.ones(num_actions, dtype=bool)
    for legal_action in legal_actions:
        illegal_actions[legal_action] = False

    obs_with_mask = {**obs, "illegal_actions": illegal_actions}
    converted: Dict[str, torch.Tensor] = {}
    for key, value in obs_with_mask.items():
        tensor = torch.from_numpy(value).to(device)
        if tensor.ndim == 0:
            tensor = tensor.unsqueeze(0)
        converted[key] = tensor.unsqueeze(0).unsqueeze(0)
    return converted


def prepare_observation_batch(
    obs_list: list[Dict[str, np.ndarray]],
    legal_actions_list: list[list[int]],
    num_actions: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Batch version of prepare_observation (optimized for single GPU transfer)."""

    batch_size = len(obs_list)

    # Build illegal actions mask as numpy array (CPU-only)
    illegal_actions_batch = np.ones((batch_size, num_actions), dtype=bool)
    for i, legal_actions in enumerate(legal_actions_list):
        for legal_action in legal_actions:
            illegal_actions_batch[i, legal_action] = False

    # Stack all observations as numpy arrays first (CPU-only)
    batched = {}
    for key in obs_list[0].keys():
        shapes = [obs[key].shape for obs in obs_list]
        if not all(shape == shapes[0] for shape in shapes):
            raise ValueError(
                f"Observation field '{key}' has inconsistent shapes {shapes}"
            )
        stacked = np.stack([obs[key] for obs in obs_list], axis=0)
        batched[key] = stacked

    # Add illegal actions mask
    batched["illegal_actions"] = illegal_actions_batch

    # Single GPU transfer for all data, then add time dimension
    return {
        key: torch.from_numpy(value).to(device).unsqueeze(1)
        for key, value in batched.items()
    }


def init_inference_inputs(
    batch_size: int,
    device: torch.device,
    policy,
) -> Tuple[torch.Tensor, torch.Tensor, Any]:
    """Initialize RL2 features, time indices, and hidden state for inference."""

    rl2s = torch.zeros((batch_size, 1, 14), device=device)
    time_idxs = torch.zeros((batch_size, 1, 1), dtype=torch.long, device=device)
    hidden_state = policy.traj_encoder.init_hidden_state(
        batch_size=batch_size, device=device
    )
    return rl2s, time_idxs, hidden_state


def update_rl2_features(
    rl2s: torch.Tensor,
    reward: float,
    action: int,
    done: bool,
) -> torch.Tensor:
    """Update RL2 features with the latest reward/action/done tuple."""

    rl2s = rl2s.clone()
    rl2s[0, 0, 0] = reward
    rl2s[0, 0, 1] = action
    rl2s[0, 0, 2] = float(done)
    return rl2s


def update_time_index(time_idxs: torch.Tensor, step: int) -> torch.Tensor:
    """Advance the time index tensor to reflect the current step."""

    time_idxs = time_idxs.clone()
    time_idxs[0, 0, 0] = step
    return time_idxs


def reset_hidden_state_if_done(
    policy,
    hidden_state: Any,
    done: bool,
) -> Any:
    """Reset the trajectory encoder state whenever an episode terminates."""

    if done:
        done_mask = np.array([True])
        hidden_state = policy.traj_encoder.reset_hidden_state(hidden_state, done_mask)
    return hidden_state


def get_policy_and_value(
    policy,
    obs_torch: Dict[str, torch.Tensor],
    rl2s: torch.Tensor,
    time_idxs: torch.Tensor,
    hidden_state: Any,
    gamma_idx: int = -1,
    temperature: float = 1,
):
    """Run a forward pass to obtain action probabilities, Q-values, and V(s)."""

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
        actions_expanded = actions_expanded.expand(
            num_actions, 1, 1, num_gammas, num_actions
        )

        all_q_values_dist = policy.critics(traj_emb, actions_expanded)
        all_q_values = policy.critics.bin_dist_to_raw_vals(all_q_values_dist)

        action_probs = all_action_probs[0, 0, gamma_idx, :]
        if temperature != 1.0:
            scaled = torch.pow(action_probs, 1.0 / temperature)
            denom = scaled.sum()
            if denom.item() == 0.0:
                scaled = torch.full_like(scaled, 1.0 / scaled.numel())
            else:
                scaled = scaled / denom
            action_probs = scaled
        q_values = all_q_values[:, 0, 0, :, gamma_idx, 0].mean(dim=1)
        state_value = (action_probs * q_values).sum()

        return action_probs, q_values, state_value, all_action_probs, new_hidden_state


def get_policy_and_value_batch(
    policy,
    obs_torch: Dict[str, torch.Tensor],
    rl2s: torch.Tensor,
    time_idxs: torch.Tensor,
    hidden_state: Any,
    gamma_idx: int = -1,
):
    """Batch forward pass."""
    import time

    with torch.no_grad():
        t_start = time.perf_counter()
        batch_size = rl2s.shape[0]
        tstep_emb = policy.tstep_encoder(obs=obs_torch, rl2s=rl2s)
        t_tstep = time.perf_counter()

        traj_emb, new_hidden_state = policy.traj_encoder(
            tstep_emb, time_idxs=time_idxs, hidden_state=hidden_state
        )
        t_traj = time.perf_counter()

        action_dist = policy.actor(
            traj_emb,
            straight_from_obs={k: obs_torch[k] for k in policy.pass_obs_keys_to_actor},
        )
        all_action_probs = action_dist.probs
        t_actor = time.perf_counter()

        num_actions = policy.action_dim
        num_gammas = len(policy.gammas)
        device = traj_emb.device

        all_actions = torch.eye(num_actions, device=device)
        actions_expanded = all_actions.unsqueeze(1).unsqueeze(1).unsqueeze(2)
        actions_expanded = actions_expanded.expand(
            num_actions, batch_size, 1, num_gammas, num_actions
        )

        all_q_values_dist = policy.critics(traj_emb, actions_expanded)
        all_q_values = policy.critics.bin_dist_to_raw_vals(all_q_values_dist)
        t_critic = time.perf_counter()

        action_probs = all_action_probs[:, 0, gamma_idx, :]
        q_values = all_q_values[:, :, 0, :, gamma_idx, 0].mean(dim=2).permute(1, 0)
        state_value = (action_probs * q_values).sum(dim=1)
        t_post = time.perf_counter()

        tstep_ms = (t_tstep - t_start) * 1000
        traj_ms = (t_traj - t_tstep) * 1000
        actor_ms = (t_actor - t_traj) * 1000
        critic_ms = (t_critic - t_actor) * 1000
        post_ms = (t_post - t_critic) * 1000
        total_ms = (t_post - t_start) * 1000

        print(
            f"[MODEL_TIMING] batch_size={batch_size} tstep={tstep_ms:.2f}ms traj={traj_ms:.2f}ms actor={actor_ms:.2f}ms critic={critic_ms:.2f}ms post={post_ms:.2f}ms total={total_ms:.2f}ms"
        )

        return action_probs, q_values, state_value, all_action_probs, new_hidden_state
