"""
PUCT Policy Provider for Rust MCTS Integration
Provides neural policy queries via PyO3 callback interface
"""

import logging
import numpy as np
from typing import List, Optional
import torch

logger = logging.getLogger(__name__)


class PUCTPolicyProvider:
    """
    Policy provider for PUCT integration with poke-engine MCTS.

    This class is designed to be called from Rust via PyO3 during MCTS node expansion.
    It must be fast and handle state strings directly.
    """

    def __init__(self, model_name: str = "Minikazam", device: str = "cpu"):
        """
        Initialize PUCT policy provider.

        Args:
            model_name: Name of pretrained Metamon model
            device: Device to run model on
        """
        self.model_name = model_name
        self.device = device
        self.agent = None
        self.query_count = 0

        logger.info(f"Initializing PUCTPolicyProvider with model: {model_name}")
        self._load_model()

    def _load_model(self):
        """Load the Minikazam model from Metamon."""
        try:
            from metamon.rl.pretrained import get_pretrained_model

            logger.info("Loading pretrained Minikazam model for PUCT...")
            pretrained_model = get_pretrained_model(self.model_name)

            # Initialize the agent
            self.agent = pretrained_model.initialize_agent(checkpoint=None, log=False)

            logger.info(f"Successfully loaded {self.model_name} model for PUCT")

        except Exception as e:
            logger.error(f"Failed to load model for PUCT: {e}")
            raise

    def get_policy_for_state_string(self, state_str: str, legal_moves: List[str]) -> List[float]:
        """
        Get policy distribution for a poke-engine state string.

        This is the main method called from Rust during MCTS.

        NOTE: This method currently cannot parse poke-engine state strings directly.
        For true PUCT integration, we would need to:
        1. Parse state_str (complex poke-engine serialization format)
        2. Convert to Battle object or Metamon observation
        3. Query neural network

        Instead, for PUCT we need to pass the actual state data from Rust,
        not just a string representation. This requires deeper PyO3 integration.

        Args:
            state_str: Serialized poke-engine state string (currently unused)
            legal_moves: List of legal move strings (e.g., ["move 0", "switch 1"])

        Returns:
            List of 13 probabilities (one per action in Metamon action space)
        """
        self.query_count += 1

        try:
            # For PUCT integration, we return a more informed prior
            # than uniform, even without full state parsing.
            # This can be replaced with actual inference once we solve
            # the state representation problem.

            # Create base uniform prior
            policy = np.ones(13, dtype=np.float32) / 13.0

            # Apply simple heuristics based on legal moves
            # Prefer attacking moves over switches in early game
            if legal_moves:
                # Count move types
                move_actions = [m for m in legal_moves if 'move' in m.lower() and 'switch' not in m.lower()]
                switch_actions = [m for m in legal_moves if 'switch' in m.lower()]

                # Light bias towards moves (this is just a placeholder heuristic)
                for i in range(4):  # Actions 0-3 are moves
                    policy[i] *= 1.2

                # Normalize
                policy = policy / policy.sum()

            return policy.tolist()

        except Exception as e:
            logger.error(f"Error getting policy for PUCT: {e}")
            # Fallback: uniform distribution
            return (np.ones(13, dtype=np.float32) / 13.0).tolist()

    def get_policy_batch(self, state_strs: List[str], legal_moves_list: List[List[str]]) -> List[List[float]]:
        """
        Get policies for a batch of states (for efficiency).

        Args:
            state_strs: List of serialized state strings
            legal_moves_list: List of legal move lists

        Returns:
            List of policy distributions
        """
        return [
            self.get_policy_for_state_string(state_str, legal_moves)
            for state_str, legal_moves in zip(state_strs, legal_moves_list)
        ]

    def get_stats(self) -> dict:
        """Get provider statistics."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "model_loaded": self.agent is not None,
            "query_count": self.query_count,
        }


# Global instance for use by Rust
_global_policy_provider = None


def initialize_puct_policy_provider(model_name: str = "Minikazam", device: str = "cpu"):
    """
    Initialize the global PUCT policy provider.

    This should be called before starting MCTS with neural priors.
    """
    global _global_policy_provider
    _global_policy_provider = PUCTPolicyProvider(model_name=model_name, device=device)
    logger.info("Global PUCT policy provider initialized")


def get_policy_for_rust(state_str: str, legal_moves: List[str]) -> List[float]:
    """
    Entry point for Rust to query neural policy.

    This function is called via PyO3 from the Rust MCTS code.

    Args:
        state_str: Serialized poke-engine state
        legal_moves: List of legal move strings

    Returns:
        13-dimensional policy distribution
    """
    global _global_policy_provider

    if _global_policy_provider is None:
        logger.warning("PUCT policy provider not initialized, initializing with defaults")
        initialize_puct_policy_provider()

    return _global_policy_provider.get_policy_for_state_string(state_str, legal_moves)
