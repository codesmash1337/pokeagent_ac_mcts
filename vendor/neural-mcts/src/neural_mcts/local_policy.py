"""
Local Policy Provider - loads Minikazam model directly without Flask server
"""

import logging
import numpy as np
from typing import Optional, List, Dict
import torch

logger = logging.getLogger(__name__)


class LocalPolicyProvider:
    """
    Load and query Minikazam Actor-Critic model directly in-process.
    No Flask server required.
    """

    def __init__(self, model_name: str = "Minikazam", device: str = "cpu"):
        """
        Initialize local policy provider.

        Args:
            model_name: Name of pretrained Metamon model
            device: Device to run model on ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.device = device
        self.agent = None
        self.query_count = 0
        self.cache_hits = 0

        logger.info(f"Initializing LocalPolicyProvider with model: {model_name}")
        self._load_model()

    def _load_model(self):
        """Load the Minikazam model from Metamon."""
        try:
            from metamon.rl.pretrained import get_pretrained_model

            logger.info("Loading pretrained Minikazam model...")
            pretrained_model = get_pretrained_model(self.model_name)

            # Initialize the agent (without logging to wandb)
            # Use default checkpoint (40 for Minikazam)
            self.agent = pretrained_model.initialize_agent(checkpoint=None, log=False)

            logger.info(f"Successfully loaded {self.model_name} model")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def get_policy(self, obs: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        """
        Get policy distribution for a single observation.

        Args:
            obs: Observation dictionary from StateTranslator with:
                - "text_tokens": int32 array of token IDs
                - "numbers": float32 array of numerical features
                - "illegal_actions": bool array marking invalid actions

        Returns:
            13-dimensional policy distribution (masked and normalized), or None if error
        """
        if self.agent is None:
            logger.error("Model not loaded!")
            return None

        try:
            self.query_count += 1

            # Convert observation to AMAGO format (add batch and time dimensions)
            # AMAGO expects shape: (batch_size, seq_len, ...)
            obs_batch = {
                "text_tokens": obs["text_tokens"][np.newaxis, np.newaxis, :],  # (1, 1, token_dim)
                "numbers": obs["numbers"][np.newaxis, np.newaxis, :],  # (1, 1, num_dim)
            }

            # Convert to PyTorch tensors
            obs_torch = {
                k: torch.from_numpy(v).to(self.device) for k, v in obs_batch.items()
            }

            # Get policy from agent (returns logits)
            with torch.no_grad():
                # Agent's policy network outputs action logits
                policy_logits = self.agent.actor(obs_torch)  # (1, 1, 13)

                # Convert to probabilities
                policy_probs = torch.softmax(policy_logits, dim=-1)

                # Remove batch and time dimensions
                policy_probs = policy_probs[0, 0].cpu().numpy()  # (13,)

            # Mask illegal actions
            illegal_mask = obs.get("illegal_actions", np.zeros(13, dtype=bool))
            policy_probs[illegal_mask] = 0.0

            # Renormalize
            if policy_probs.sum() > 0:
                policy_probs = policy_probs / policy_probs.sum()
            else:
                # Fallback: uniform over legal actions
                logger.warning("All actions illegal or zero probability, using uniform")
                policy_probs = (~illegal_mask).astype(float)
                if policy_probs.sum() > 0:
                    policy_probs = policy_probs / policy_probs.sum()
                else:
                    policy_probs = np.ones(13) / 13.0

            return policy_probs

        except Exception as e:
            logger.error(f"Error getting policy: {e}", exc_info=True)
            return None

    def get_policy_batch(self, obs_list: List[Dict[str, np.ndarray]]) -> List[Optional[np.ndarray]]:
        """
        Get policies for a batch of observations.

        Args:
            obs_list: List of observation dictionaries

        Returns:
            List of policy distributions
        """
        # For now, process sequentially (could optimize with true batching later)
        return [self.get_policy(obs) for obs in obs_list]

    def get_stats(self) -> dict:
        """Get provider statistics."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "model_loaded": self.agent is not None,
            "query_count": self.query_count,
            "cache_hits": self.cache_hits,
        }
