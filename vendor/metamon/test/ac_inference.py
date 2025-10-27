"""
Actor-Critic inference utilities for getting policy and value estimates.

This module provides core functions for extracting policy (action probabilities)
and value (Q-values, state values) from AMAGO-based Pokemon battle agents like Abra.

Perfect for integration with planning algorithms like MCTS.

## Statefulness in Abra Models

Abra and other AMAGO-based models maintain state across multiple timesteps in a battle
through three key mechanisms:

1. **RL2 Features (Reinforcement Learning Squared)**: A 14-dimensional tensor that encodes
   the recent history of rewards, actions, and episode status:
   - Element 0: Reward from previous step
   - Element 1: Action index from previous step (0-12)
   - Element 2: Done flag (1.0 if episode ended, 0.0 otherwise)
   - Elements 3-13: Reserved for one-hot action encoding (11 dims)

2. **Time Indices**: Track the current timestep in the battle sequence, used for
   positional encoding in the transformer.

3. **Hidden State**: The recurrent state of the trajectory encoder (transformer),
   which maintains context from all previous observations in the battle.

## Updating State After Actions

When using the model for sequential inference (e.g., MCTS simulation):

```python
# Initialize
policy = experiment.policy
rl2s, time_idxs, hidden_state = init_inference_inputs(1, device, policy)

# First inference (no prior action)
obs_torch = prepare_observation(obs, legal_actions, num_actions, device)
action_probs, q_values, state_value, _, hidden_state = get_policy_and_value(
    policy, obs_torch, rl2s, time_idxs, hidden_state, gamma_idx=-1
)

# After taking action and observing reward
rl2s = update_rl2_features(rl2s, reward=reward, action=action_idx, done=False)
time_idxs = update_time_index(time_idxs, step=1)
# hidden_state is already updated from the previous inference

# Next inference
obs_torch = prepare_observation(next_obs, next_legal_actions, num_actions, device)
action_probs, q_values, state_value, _, hidden_state = get_policy_and_value(
    policy, obs_torch, rl2s, time_idxs, hidden_state, gamma_idx=-1
)
```

## Reward Values

For Pokemon battles, rewards are typically:
- **0.0**: No reward (default for most timesteps, and for first inference)
- **+1.0**: Win
- **-1.0**: Loss
- **0.0**: Draw or mid-battle

The policy was trained with shaped rewards, but for inference in MCTS, you typically
use 0.0 for intermediate steps since you're doing forward simulation without knowing
the actual reward until the battle ends.
"""

import torch
import numpy as np
from typing import Tuple, Dict, Optional, Any

from metamon.stateful_inference import (
    get_policy_and_value,
    init_inference_inputs,
    prepare_observation,
    reset_hidden_state_if_done,
    update_rl2_features,
    update_time_index,
)


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

    def __init__(self, policy, device: torch.device, temperature: float = 1.0):
        """
        Initialize inference wrapper.

        Args:
            policy: The agent's policy network
            device: Torch device to use
        """
        self.policy = policy
        self.device = device
        self.temperature = temperature
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

        action_probs, q_values, state_value, _, self.hidden_state = get_policy_and_value(
            self.policy,
            obs_torch,
            self.rl2s,
            self.time_idxs,
            self.hidden_state,
            gamma_idx=gamma_idx,
            temperature=self.temperature,
        )
        best_action = get_best_action(action_probs, legal_actions)

        self.update(reward=0, action=best_action, done=False)

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
