"""Shared stateful inference helpers for AMAGO/Abra-style policies."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import math
import os

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
    "DEFAULT_TARGET_ENTROPY_RATIO",
    "DEFAULT_ADAPT_STRENGTH",
]

# Single source of truth for adaptive temperature parameters
DEFAULT_TARGET_ENTROPY_RATIO = 0.5  # Standard target entropy ratio
DEFAULT_ADAPT_STRENGTH = 3.0  # Standard adaptation strength


def _debug_logs_enabled() -> bool:
    value = os.getenv("POKEENGINE_DEBUG_LOGS", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


_DEBUG_LOGS_ENABLED = _debug_logs_enabled()


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


def _adaptive_temperature_batch(
    probs: torch.Tensor,
    *,
    target_entropy_ratio: float,
    adapt_strength: float,
) -> torch.Tensor:
    if adapt_strength <= 0.0 or probs.shape[-1] <= 1:
        return probs

    max_entropy = math.log(probs.shape[-1])
    if max_entropy <= 0.0:
        return probs

    probs_clamped = probs.clamp_min(1e-6)
    entropy = -(probs_clamped * probs_clamped.log()).sum(dim=-1)
    entropy_ratio = (entropy / max_entropy).clamp(0.0, 1.0)

    temperature = 1.0 + adapt_strength * (target_entropy_ratio - entropy_ratio)
    temperature = temperature.clamp_min(1.0)

    need_adjust = temperature > 1.0001
    if torch.all(~need_adjust):
        return probs

    inv_temperature = temperature.reciprocal().unsqueeze(-1)
    scaled = torch.pow(probs, inv_temperature)
    denom = scaled.sum(dim=-1, keepdim=True)
    uniform = 1.0 / probs.shape[-1]
    scaled = torch.where(
        denom == 0,
        torch.full_like(scaled, uniform),
        scaled / denom,
    )

    need_adjust = need_adjust.unsqueeze(-1)
    return torch.where(need_adjust, scaled, probs)


# Global timing accumulators for MODEL_TIMING
_MODEL_TIMING_COUNT = 0
_MODEL_TIMING_TSTEP = 0.0
_MODEL_TIMING_TRAJ = 0.0
_MODEL_TIMING_ACTOR = 0.0
_MODEL_TIMING_CRITIC = 0.0
_MODEL_TIMING_POST = 0.0
_MODEL_TIMING_TOTAL = 0.0


def reset_model_timing_stats():
    global _MODEL_TIMING_COUNT, _MODEL_TIMING_TSTEP, _MODEL_TIMING_TRAJ
    global \
        _MODEL_TIMING_ACTOR, \
        _MODEL_TIMING_CRITIC, \
        _MODEL_TIMING_POST, \
        _MODEL_TIMING_TOTAL
    _MODEL_TIMING_COUNT = 0
    _MODEL_TIMING_TSTEP = 0.0
    _MODEL_TIMING_TRAJ = 0.0
    _MODEL_TIMING_ACTOR = 0.0
    _MODEL_TIMING_CRITIC = 0.0
    _MODEL_TIMING_POST = 0.0
    _MODEL_TIMING_TOTAL = 0.0


def log_model_timing_stats():
    global _MODEL_TIMING_COUNT, _MODEL_TIMING_TSTEP, _MODEL_TIMING_TRAJ
    global \
        _MODEL_TIMING_ACTOR, \
        _MODEL_TIMING_CRITIC, \
        _MODEL_TIMING_POST, \
        _MODEL_TIMING_TOTAL
    if not _DEBUG_LOGS_ENABLED:
        return
    if _MODEL_TIMING_COUNT == 0:
        return
    print(
        f"[MODEL_TIMING] batches={_MODEL_TIMING_COUNT} tstep={_MODEL_TIMING_TSTEP:.1f}ms "
        f"traj={_MODEL_TIMING_TRAJ:.1f}ms actor={_MODEL_TIMING_ACTOR:.1f}ms "
        f"critic={_MODEL_TIMING_CRITIC:.1f}ms post={_MODEL_TIMING_POST:.1f}ms total={_MODEL_TIMING_TOTAL:.1f}ms"
    )


def get_policy_and_value_batch(
    policy,
    obs_torch: Dict[str, torch.Tensor],
    rl2s: torch.Tensor,
    time_idxs: torch.Tensor,
    hidden_state: Any,
    *,
    target_entropy_ratio: float,
    adapt_strength: float,
):
    """Batch forward pass.

    Always returns the average expected value across all critic heads.
    """
    import time

    global _MODEL_TIMING_COUNT, _MODEL_TIMING_TSTEP, _MODEL_TIMING_TRAJ
    global \
        _MODEL_TIMING_ACTOR, \
        _MODEL_TIMING_CRITIC, \
        _MODEL_TIMING_POST, \
        _MODEL_TIMING_TOTAL
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

        # Average expected value across all critic heads (vectorized)
        averaged_q_values = all_q_values[:, :, 0, :, :, 0].mean(dim=2)

        # Reshape for vectorized processing: [batch_size * num_gammas, num_actions]
        all_gamma_probs = all_action_probs[:, 0, :, :].reshape(
            batch_size * num_gammas, num_actions
        )

        # Apply adaptive temperature to all gamma heads at once
        all_gamma_probs = _adaptive_temperature_batch(
            all_gamma_probs,
            target_entropy_ratio=target_entropy_ratio,
            adapt_strength=adapt_strength,
        )

        # Reshape back: [batch_size, num_gammas, num_actions]
        all_gamma_probs = all_gamma_probs.reshape(batch_size, num_gammas, num_actions)

        # Transpose q_values to [batch_size, num_actions, num_gammas]
        q_values_transposed = averaged_q_values.permute(1, 0, 2)

        # Compute expected values for all gammas: [batch_size, num_gammas]
        # For each batch and gamma, sum over actions: probs[b,g,a] * q[b,a,g]
        expected_values_all = torch.einsum(
            "bga,bag->bg", all_gamma_probs, q_values_transposed
        )

        # Average across all gamma heads: [batch_size]
        state_value = expected_values_all.mean(dim=1)

        # Use the last gamma index for action_probs and q_values (for compatibility)
        # Extract from already computed all_gamma_probs to avoid recomputation
        action_probs = all_gamma_probs[:, -1, :]
        q_values = averaged_q_values[:, :, -1].permute(1, 0)

        t_post = time.perf_counter()

        tstep_ms = (t_tstep - t_start) * 1000
        traj_ms = (t_traj - t_tstep) * 1000
        actor_ms = (t_actor - t_traj) * 1000
        critic_ms = (t_critic - t_actor) * 1000
        post_ms = (t_post - t_critic) * 1000
        total_ms = (t_post - t_start) * 1000

        # Accumulate timing stats
        _MODEL_TIMING_COUNT += 1
        _MODEL_TIMING_TSTEP += tstep_ms
        _MODEL_TIMING_TRAJ += traj_ms
        _MODEL_TIMING_ACTOR += actor_ms
        _MODEL_TIMING_CRITIC += critic_ms
        _MODEL_TIMING_POST += post_ms
        _MODEL_TIMING_TOTAL += total_ms

        return action_probs, q_values, state_value, all_action_probs, new_hidden_state


def get_policy_and_value(
    policy,
    obs_torch: Dict[str, torch.Tensor],
    rl2s: torch.Tensor,
    time_idxs: torch.Tensor,
    hidden_state: Any,
    *,
    target_entropy_ratio: float,
    adapt_strength: float,
):
    """Compatibility wrapper that reuses the batched inference path for batch=1."""

    action_probs_b, q_values_b, state_value_b, all_action_probs, new_hidden_state = (
        get_policy_and_value_batch(
            policy,
            obs_torch,
            rl2s,
            time_idxs,
            hidden_state,
            target_entropy_ratio=target_entropy_ratio,
            adapt_strength=adapt_strength,
        )
    )

    action_probs = action_probs_b[0]
    q_values = q_values_b[0]
    state_value = state_value_b[0]
    return action_probs, q_values, state_value, all_action_probs, new_hidden_state
