"""
Actor-Critic inference utilities for getting policy and value estimates.

This module provides core functions for extracting policy (action probabilities)
and value (Q-values, state values) from AMAGO-based Pokemon battle agents like Abra.

Perfect for integration with planning algorithms like MCTS.
"""

import torch
import numpy as np
from typing import Tuple, Dict, Optional, Any


def get_policy_and_value(
    policy,
    obs_torch: Dict[str, torch.Tensor],
    rl2s: torch.Tensor,
    time_idxs: torch.Tensor,
    hidden_state: Any,
    gamma_idx: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
    """
    Get policy and value estimates from an AMAGO agent in a single forward pass.

    This performs one forward pass through the actor-critic network to get:
    - Action probabilities (policy)
    - Q-values for all actions (critic)
    - State value estimate

    Args:
        policy: The agent's policy network (e.g., Abra policy)
        obs_torch: Dictionary of observation tensors with shape [batch, length, ...]
        rl2s: RL2 features (prev reward, action, done) with shape [batch, length, rl2_dim]
        time_idxs: Time indices for positional encoding [batch, length, 1]
        hidden_state: Hidden state from trajectory encoder (or None)
        gamma_idx: Index of discount factor to use for return values (-1 = main gamma)

    Returns:
        Tuple containing:
        - action_probs: Action probabilities [num_actions] for selected gamma
        - q_values: Q-values [num_actions] averaged over critic ensemble
        - state_value: Expected value under current policy (scalar)
        - all_action_probs: Full action probs [batch, length, num_gammas, num_actions]
        - new_hidden_state: Updated hidden state
    """
    with torch.no_grad():
        # Step 1: Timestep Encoder
        # Embeds raw observations (text tokens + numbers) into a fixed-size vector
        tstep_emb = policy.tstep_encoder(obs=obs_torch, rl2s=rl2s)
        # Shape: (batch=1, length=1, tstep_dim)
        # Example: (1, 1, 100) - single timestep embedding

        # Step 2: Trajectory Encoder (Transformer)
        # Processes sequence of timestep embeddings with attention over battle history
        traj_emb, new_hidden_state = policy.traj_encoder(
            tstep_emb, time_idxs=time_idxs, hidden_state=hidden_state
        )
        # Shape: (batch=1, length=1, traj_dim)
        # Example: (1, 1, 100) - contextual state representation
        # hidden_state maintains context across battle turns

        # Step 3a: Actor Network (Policy)
        # Maps state to action distribution for all gammas
        action_dist = policy.actor(
            traj_emb,
            straight_from_obs={k: obs_torch[k] for k in policy.pass_obs_keys_to_actor},
        )
        all_action_probs = action_dist.probs
        # Shape: (batch=1, length=1, num_gammas, num_actions)
        # Example: (1, 1, 6, 13)
        # - 6 gammas: [0.1, 0.9, 0.95, 0.97, 0.99, 0.995]
        # - 13 actions: 4 moves + 5 switches + 4 tera moves
        # Each gamma has its own probability distribution over actions

        # Step 3b: Prepare Critic Inputs
        # Get Q-values for ALL possible actions simultaneously
        num_actions = policy.action_dim  # 13 for Pokemon
        num_gammas = len(policy.gammas)  # 6 discount factors
        device = traj_emb.device

        # Create identity matrix for one-hot action vectors
        all_actions = torch.eye(num_actions).to(device)
        # Shape: (num_actions, num_actions)
        # Example: (13, 13) - each row is a one-hot vector for one action
        # [[1,0,0,...,0],  <- action 0
        #  [0,1,0,...,0],  <- action 1
        #  ...
        #  [0,0,0,...,1]]  <- action 12

        # Expand to critic input format: (K, batch, length, gammas, action_dim)
        actions_expanded = all_actions.unsqueeze(1).unsqueeze(1).unsqueeze(2)
        # Shape after unsqueeze: (num_actions, 1, 1, 1, num_actions)
        # Example: (13, 1, 1, 1, 13)

        actions_expanded = actions_expanded.expand(
            num_actions, 1, 1, num_gammas, num_actions
        )
        # Shape: (K=num_actions, batch=1, length=1, gammas, action_dim)
        # Example: (13, 1, 1, 6, 13)
        # K=13: We're evaluating all 13 actions at once!
        # This is the key difference from training (where K=1)

        # Step 3c: Critic Network (Q-values)
        # For discrete actions, critic outputs Q-values for all actions internally,
        # then uses one-hot vectors to select the Q-value for each input action
        all_q_values_dist = policy.critics(traj_emb, actions_expanded)
        # Returns: Categorical distribution over value bins (Abra uses NCriticsTwoHot)
        # The categorical represents a distribution over discretized Q-value bins

        # Convert from categorical distribution over bins to scalar Q-values
        all_q_values = policy.critics.bin_dist_to_raw_vals(all_q_values_dist)
        # Shape: (K=num_actions, batch=1, length=1, num_critics, gammas, 1)
        # Example: (13, 1, 1, 4, 6, 1)
        # Breakdown:
        #   K=13: One Q-value for each of the 13 actions we're evaluating
        #   batch=1: Single state
        #   length=1: Single timestep
        #   num_critics=4: Ensemble of 4 Q-networks for robustness
        #   gammas=6: Q-values for each discount factor
        #   1: Scalar Q-value

        # Step 4: Extract Values for Selected Gamma
        # Get action probabilities for the main gamma (usually index -1 = 0.995)
        action_probs = all_action_probs[0, 0, gamma_idx, :]
        # Shape: (num_actions,)
        # Example: (13,) - probability for each action
        # [π(a₀|s), π(a₁|s), ..., π(a₁₂|s)]

        # Get Q-values for the main gamma, averaged over critic ensemble
        q_values = all_q_values[:, 0, 0, :, gamma_idx, 0].mean(dim=1)
        # Shape: (num_actions,)
        # Example: (13,) - Q-value for each action
        # [Q(s,a₀), Q(s,a₁), ..., Q(s,a₁₂)]
        # .mean(dim=1) averages over the 4 critics for robustness

        # Step 5: Compute State Value
        # V(s) = Expected Q-value under current policy = Σ π(a|s) * Q(s,a)
        state_value = (action_probs * q_values).sum()
        # Shape: scalar
        # This is the value of the current state following the policy

        return action_probs, q_values, state_value, all_action_probs, new_hidden_state


def get_action_probs(
    policy,
    obs_torch: Dict[str, torch.Tensor],
    rl2s: torch.Tensor,
    time_idxs: torch.Tensor,
    hidden_state: Any,
    gamma_idx: int = -1,
) -> Tuple[torch.Tensor, Any]:
    """
    Get only action probabilities (faster than full policy+value).

    Use this when you only need the policy and not the value estimates.

    Args:
        policy: The agent's policy network
        obs_torch: Dictionary of observation tensors
        rl2s: RL2 features
        time_idxs: Time indices
        hidden_state: Hidden state from trajectory encoder
        gamma_idx: Index of discount factor to use (-1 = main gamma)

    Returns:
        Tuple containing:
        - action_probs: Action probabilities [num_actions]
        - new_hidden_state: Updated hidden state
    """
    with torch.no_grad():
        tstep_emb = policy.tstep_encoder(obs=obs_torch, rl2s=rl2s)
        traj_emb, new_hidden_state = policy.traj_encoder(
            tstep_emb, time_idxs=time_idxs, hidden_state=hidden_state
        )

        action_dist = policy.actor(
            traj_emb,
            straight_from_obs={k: obs_torch[k] for k in policy.pass_obs_keys_to_actor},
        )
        action_probs = action_dist.probs[0, 0, gamma_idx, :]

        return action_probs, new_hidden_state


def prepare_observation(
    obs: Dict[str, np.ndarray],
    legal_actions: list,
    num_actions: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Prepare observation for inference by adding illegal action mask and converting to torch.

    Args:
        obs: Dictionary of numpy observation arrays
        legal_actions: List of legal action indices
        num_actions: Total number of possible actions
        device: Torch device to use

    Returns:
        Dictionary of torch tensors ready for inference
    """
    # Create illegal actions mask
    illegal_actions = np.ones(num_actions, dtype=bool)
    for legal_action in legal_actions:
        illegal_actions[legal_action] = False

    # Create new dict with illegal_actions added (don't mutate input)
    obs_with_mask = {**obs, "illegal_actions": illegal_actions}

    # Convert to torch tensors with batch and length dimensions
    obs_torch = {
        k: torch.from_numpy(v).to(device).unsqueeze(0).unsqueeze(0)
        for k, v in obs_with_mask.items()
    }

    return obs_torch


def init_inference_inputs(
    batch_size: int,
    device: torch.device,
    policy,
) -> Tuple[torch.Tensor, torch.Tensor, Any]:
    """
    Initialize RL2 features, time indices, and hidden state for inference.

    Args:
        batch_size: Batch size (usually 1 for inference)
        device: Torch device
        policy: The agent's policy network

    Returns:
        Tuple of (rl2s, time_idxs, hidden_state)
    """
    # RL2 features: 14-dimensional vector (reward + one-hot action + done)
    rl2s = torch.zeros((batch_size, 1, 14)).to(device)

    # Time indices (start at 0)
    time_idxs = torch.zeros((batch_size, 1, 1), dtype=torch.long).to(device)

    # Hidden state
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
    """
    Update RL2 features with the latest step information.

    Args:
        rl2s: Current RL2 tensor [batch, length, 3]
        reward: Reward from last step
        action: Action taken in last step
        done: Whether episode ended

    Returns:
        Updated RL2 tensor
    """
    rl2s = rl2s.clone()
    rl2s[0, 0, 0] = reward
    rl2s[0, 0, 1] = action
    rl2s[0, 0, 2] = float(done)
    return rl2s


def update_time_index(time_idxs: torch.Tensor, step: int) -> torch.Tensor:
    """
    Update time index for the current step.

    Args:
        time_idxs: Current time index tensor
        step: Current step number

    Returns:
        Updated time index tensor
    """
    time_idxs = time_idxs.clone()
    time_idxs[0, 0, 0] = step
    return time_idxs


def reset_hidden_state_if_done(
    policy,
    hidden_state: Any,
    done: bool,
) -> Any:
    """
    Reset hidden state if episode is done.

    Args:
        policy: The agent's policy network
        hidden_state: Current hidden state
        done: Whether episode ended

    Returns:
        Updated hidden state
    """
    if done:
        done_mask = np.array([True])
        hidden_state = policy.traj_encoder.reset_hidden_state(hidden_state, done_mask)
    return hidden_state


def sample_action(
    action_probs: torch.Tensor,
    temperature: float = 1.0,
    legal_actions: Optional[list] = None,
) -> int:
    """
    Sample an action from the policy.

    Args:
        action_probs: Action probabilities [num_actions]
        temperature: Temperature for sampling (1.0 = use raw probs, <1.0 = more greedy)
        legal_actions: Optional list of legal action indices (for additional filtering)

    Returns:
        Sampled action index
    """
    if temperature != 1.0:
        action_probs = torch.pow(action_probs, 1.0 / temperature)
        action_probs = action_probs / action_probs.sum()

    if legal_actions is not None:
        # Further filter to only legal actions
        mask = torch.zeros_like(action_probs)
        mask[legal_actions] = 1.0
        action_probs = action_probs * mask
        action_probs = action_probs / action_probs.sum()

    return torch.multinomial(action_probs, num_samples=1).item()


def get_best_action(
    action_probs: torch.Tensor,
    legal_actions: Optional[list] = None,
) -> int:
    """
    Get the best action (greedy selection).

    Args:
        action_probs: Action probabilities [num_actions]
        legal_actions: Optional list of legal action indices

    Returns:
        Best action index
    """
    if legal_actions is not None:
        # Only consider legal actions
        mask = torch.zeros_like(action_probs)
        mask[legal_actions] = 1.0
        action_probs = action_probs * mask

    return torch.argmax(action_probs).item()


# Convenience class for stateful inference
class PolicyValueInference:
    """
    Stateful wrapper for policy-value inference.

    Maintains hidden state and handles sequential inference across battle steps.
    """

    def __init__(self, policy, device: torch.device):
        """
        Initialize inference wrapper.

        Args:
            policy: The agent's policy network
            device: Torch device to use
        """
        self.policy = policy
        self.device = device
        self.rl2s, self.time_idxs, self.hidden_state = init_inference_inputs(
            batch_size=1, device=device, policy=policy
        )
        self.step = 0

    def reset(self):
        """Reset to initial state."""
        self.rl2s, self.time_idxs, self.hidden_state = init_inference_inputs(
            batch_size=1, device=self.device, policy=self.policy
        )
        self.step = 0

    def __call__(
        self,
        obs: Dict[str, np.ndarray],
        legal_actions: list,
        gamma_idx: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get policy and value for current observation.

        Args:
            obs: Dictionary of numpy observation arrays
            legal_actions: List of legal action indices
            gamma_idx: Discount factor index to use

        Returns:
            Tuple of (action_probs, q_values, state_value)
        """
        obs_torch = prepare_observation(
            obs, legal_actions, self.policy.action_dim, self.device
        )

        action_probs, q_values, state_value, _, self.hidden_state = (
            get_policy_and_value(
                self.policy,
                obs_torch,
                self.rl2s,
                self.time_idxs,
                self.hidden_state,
                gamma_idx=gamma_idx,
            )
        )

        return action_probs, q_values, state_value

    def update(self, reward: float, action: int, done: bool):
        """
        Update internal state after taking an action.

        Args:
            reward: Reward received
            action: Action taken
            done: Whether episode ended
        """
        self.rl2s = update_rl2_features(self.rl2s, reward, action, done)
        self.step += 1
        self.time_idxs = update_time_index(self.time_idxs, self.step)
        self.hidden_state = reset_hidden_state_if_done(
            self.policy, self.hidden_state, done
        )
