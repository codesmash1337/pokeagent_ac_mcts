"""
State Translation Layer
Converts foul-play Battle objects to Metamon observation format
"""

import logging
from typing import Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class StateTranslator:
    """
    Translates foul-play Battle objects to Metamon observation format.

    Uses Metamon's built-in UniversalState.from_Battle() conversion,
    then generates observations using the appropriate observation space.
    """

    def __init__(self, observation_space_type="TeamPreviewObservationSpace"):
        """
        Initialize the state translator.

        Args:
            observation_space_type: Which Metamon observation space to use
        """
        try:
            from metamon.interface import (
                UniversalState,
                get_observation_space,
                TokenizedObservationSpace,
                UniversalAction,
                get_action_space
            )
            from metamon.tokenizer import get_tokenizer

            self.UniversalState = UniversalState
            self.UniversalAction = UniversalAction

            # Create observation space (tokenized for model compatibility)
            base_obs_space = get_observation_space(observation_space_type)
            tokenizer = get_tokenizer("DefaultObservationSpace-v1")
            self.obs_space = TokenizedObservationSpace(base_obs_space, tokenizer)

            # Action space for legal action computation
            self.action_space = get_action_space("DefaultActionSpace")

            logger.info(f"Initialized StateTranslator with {observation_space_type}")

        except Exception as e:
            logger.error(f"Failed to initialize StateTranslator: {e}")
            raise

    def translate(self, battle, legal_actions: Optional[List[int]] = None) -> Dict[str, np.ndarray]:
        """
        Translate a foul-play Battle object to Metamon observation.

        Args:
            battle: foul-play Battle object (NOT poke-env Battle)
            legal_actions: Optional list of legal action indices (for masking)

        Returns:
            Dictionary containing observation tensors compatible with Metamon
        """
        try:
            # Wrap foul-play Battle to make it compatible with Metamon
            from neural_mcts.battle_adapter import adapt_battle
            adapted_battle = adapt_battle(battle)

            # Convert adapted Battle to UniversalState
            universal_state = self.UniversalState.from_Battle(adapted_battle)

            # Generate tokenized observation
            obs = self.obs_space.state_to_obs(universal_state)

            # obs contains:
            # - "text_tokens": int32 array of token IDs
            # - "numbers": float32 array of numerical features
            # - "illegal_actions": bool array marking invalid actions

            # Override with provided legal_actions if given
            if legal_actions is not None:
                legal_mask = np.zeros(13, dtype=bool)
                for action_idx in legal_actions:
                    if 0 <= action_idx < 13:
                        legal_mask[action_idx] = True
                obs["illegal_actions"] = ~legal_mask  # Metamon uses illegal_actions (inverted)

            return obs

        except Exception as e:
            logger.error(f"Error translating battle to observation: {e}")
            raise

    def get_legal_actions(self, battle) -> List[int]:
        """
        Get legal action indices for a battle state.

        Args:
            battle: foul-play Battle object

        Returns:
            List of legal action indices (0-12)
        """
        try:
            # Wrap foul-play Battle to make it compatible
            from neural_mcts.battle_adapter import adapt_battle
            adapted_battle = adapt_battle(battle)

            # Convert to UniversalState
            universal_state = self.UniversalState.from_Battle(adapted_battle)

            # Get definitely valid actions
            legal_universal_actions = self.UniversalAction.definitely_valid_actions(
                universal_state, adapted_battle
            )

            # Convert to action indices
            legal_indices = [
                self.action_space.action_to_agent_output(universal_state, action)
                for action in legal_universal_actions
            ]

            return legal_indices

        except Exception as e:
            logger.error(f"Error getting legal actions: {e}")
            # Fallback: assume all actions legal
            return list(range(13))
