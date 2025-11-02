"""Shared stateful inference helpers for AMAGO/Abra-style policies."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import math

import numpy as np
import torch
import sys

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


def _adaptive_temperature_single(
    probs: torch.Tensor,
    *,
    target_entropy_ratio: float,
    adapt_strength: float,
) -> torch.Tensor:
    if adapt_strength <= 0.0 or probs.numel() <= 1:
        return probs

    max_entropy = math.log(probs.numel())
    if max_entropy <= 0.0:
        return probs

    probs_clamped = probs.clamp_min(1e-6)
    entropy = -(probs_clamped * probs_clamped.log()).sum()
    entropy_ratio = float((entropy / max_entropy).clamp(0.0, 1.0).item())

    if entropy_ratio >= target_entropy_ratio:
        return probs

    temperature = 1.0 + adapt_strength * (target_entropy_ratio - entropy_ratio)
    temperature = max(1.0, temperature)

    if temperature <= 1.0001:
        return probs

    scaled = torch.pow(probs, 1.0 / temperature)
    denom = scaled.sum()
    if denom.item() == 0.0:
        scaled = torch.full_like(scaled, 1.0 / scaled.numel())
    else:
        scaled = scaled / denom
    return scaled


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


def get_policy_and_value(
    policy,
    obs_torch: Dict[str, torch.Tensor],
    rl2s: torch.Tensor,
    time_idxs: torch.Tensor,
    hidden_state: Any,
    gamma_idx: int = -1,
    *,
    target_entropy_ratio: float,
    adapt_strength: float,
):
    """Run a forward pass to obtain action probabilities, Q-values, and V(s)."""

    sys.stderr.write("[CRITIC_DEBUG] path=get_policy_and_value (single)\n")
    sys.stderr.flush()

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

        averaged_q_values = all_q_values[:, 0, 0, :, :, 0].mean(dim=1)

        action_probs = all_action_probs[0, 0, gamma_idx, :]
        action_probs = _adaptive_temperature_single(
            action_probs,
            target_entropy_ratio=target_entropy_ratio,
            adapt_strength=adapt_strength,
        )
        q_values = averaged_q_values[:, gamma_idx]

        for gamma_offset in range(num_gammas):
            gamma_q = averaged_q_values[:, gamma_offset]
            gamma_action_probs = all_action_probs[0, 0, gamma_offset, :]
            gamma_action_probs = _adaptive_temperature_single(
                gamma_action_probs,
                target_entropy_ratio=target_entropy_ratio,
                adapt_strength=adapt_strength,
            )
            gamma_expected = (gamma_action_probs * gamma_q).sum()
            gamma_max = gamma_q.max()
            sys.stderr.write(
                f"[CRITIC_DEBUG] gamma_idx={gamma_offset} expected={gamma_expected.item()} max={gamma_max.item()}\n"
            )
            sys.stderr.flush()

        state_value = (action_probs * q_values).sum()

        return action_probs, q_values, state_value, all_action_probs, new_hidden_state


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
    gamma_idx: int = -1,
    *,
    target_entropy_ratio: float,
    adapt_strength: float,
    use_argmax_value: bool = False,
    use_longest_horizon_only: bool = False,
):
    """Batch forward pass."""
    import time

    global _MODEL_TIMING_COUNT, _MODEL_TIMING_TSTEP, _MODEL_TIMING_TRAJ
    global \
        _MODEL_TIMING_ACTOR, \
        _MODEL_TIMING_CRITIC, \
        _MODEL_TIMING_POST, \
        _MODEL_TIMING_TOTAL
    print("Getting batch policy value")
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

        selected_gamma_idx = num_gammas - 1 if use_longest_horizon_only else gamma_idx
        if selected_gamma_idx < 0:
            selected_gamma_idx = num_gammas + selected_gamma_idx
        selected_gamma_idx = max(0, min(selected_gamma_idx, num_gammas - 1))

        all_actions = torch.eye(num_actions, device=device)
        actions_expanded = all_actions.unsqueeze(1).unsqueeze(1).unsqueeze(2)
        actions_expanded = actions_expanded.expand(
            num_actions, batch_size, 1, num_gammas, num_actions
        )

        all_q_values_dist = policy.critics(traj_emb, actions_expanded)
        all_q_values = policy.critics.bin_dist_to_raw_vals(all_q_values_dist)
        t_critic = time.perf_counter()

        action_probs = all_action_probs[:, 0, selected_gamma_idx, :]
        action_probs = _adaptive_temperature_batch(
            action_probs,
            target_entropy_ratio=target_entropy_ratio,
            adapt_strength=adapt_strength,
        )
        averaged_q_values = all_q_values[:, :, 0, :, :, 0].mean(dim=2)
        q_values = averaged_q_values[:, :, selected_gamma_idx].permute(1, 0)
        print(f"Num gamms {num_gammas}")
        with torch.no_grad():
            for gamma_offset in range(num_gammas):
                gamma_q = averaged_q_values[:, :, gamma_offset].permute(1, 0)
                gamma_max = gamma_q.max(dim=1).values
                gamma_action_probs = all_action_probs[:, 0, gamma_offset, :]
                gamma_action_probs = _adaptive_temperature_batch(
                    gamma_action_probs,
                    target_entropy_ratio=target_entropy_ratio,
                    adapt_strength=adapt_strength,
                )
                gamma_expected = (gamma_action_probs * gamma_q).sum(dim=1)
                print(
                    f"[CRITIC_DEBUG] gamma_idx={gamma_offset} expected={gamma_expected.cpu().numpy()} max={gamma_max.cpu().numpy()}\n"
                )
        if use_argmax_value:
            state_value = q_values.max(dim=1).values
        else:
            state_value = (action_probs * q_values).sum(dim=1)
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
