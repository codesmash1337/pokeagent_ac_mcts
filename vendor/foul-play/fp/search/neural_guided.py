"""
Neural-Guided MCTS Move Selection
Integrates Actor-Critic policy priors from Minikazam into MCTS search
"""

import logging
import sys
import os

# Add neural-mcts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../neural-mcts/src'))

from neural_mcts.local_policy import LocalPolicyProvider
from neural_mcts.state_translator import StateTranslator
from neural_mcts.neural_search import NeuralGuidedSearch

logger = logging.getLogger(__name__)

# Global instances (initialized lazily)
_policy_provider = None
_state_translator = None
_neural_search = None


def get_neural_components(c_puct=1.0):
    """
    Get or initialize neural components (singleton pattern).

    Args:
        c_puct: Exploration constant (not used in post-search reranking)

    Returns:
        Tuple of (LocalPolicyProvider, StateTranslator, NeuralGuidedSearch)
    """
    global _policy_provider, _state_translator, _neural_search

    if _policy_provider is None:
        logger.info("Initializing Abra model for neural-guided search...")
        try:
            # Initialize state translator
            _state_translator = StateTranslator(
                observation_space_type="TeamPreviewObservationSpace"
            )
            logger.info("StateTranslator initialized")

            # Initialize policy provider with Abra (stronger Gen9 transformer model)
            # Abra: Medium multitask agent, 50% GXE in Gen9OU, trained on Gen1-9
            # Alternative: "Minikazam" (smaller RNN, faster but less capable)
            _policy_provider = LocalPolicyProvider(
                model_name="Abra",
                device="cpu"
            )
            logger.info("LocalPolicyProvider initialized")

            # Initialize neural search
            _neural_search = NeuralGuidedSearch(
                policy_provider=_policy_provider,
                state_translator=_state_translator,
                c_puct=c_puct,
                use_neural_prior=True,
                fallback_to_uniform=True
            )
            logger.info("Neural-guided search initialized successfully!")

        except Exception as e:
            logger.error(f"Failed to initialize neural guidance: {e}", exc_info=True)
            logger.warning("Falling back to vanilla MCTS")
            return None, None, None

    return _policy_provider, _state_translator, _neural_search


def select_move_with_neural_guidance(mcts_results, battles, c_puct=1.0):
    """
    Select move using neural-guided MCTS.

    Args:
        mcts_results: List of (MctsResult, sample_chance, index) tuples
        battles: List of Battle objects corresponding to MCTS results
        c_puct: Exploration constant (for future use)

    Returns:
        Selected move string, or None if error (triggers fallback)
    """
    _, _, neural_search = get_neural_components(c_puct=c_puct)

    if neural_search is None:
        # Fallback to vanilla MCTS if neural components failed
        logger.warning("Neural components not available, using vanilla MCTS")
        return None

    try:
        move = neural_search.select_move_with_neural_prior(mcts_results, battles)
        logger.info(f"Neural-guided move selection: {move}")
        return move
    except Exception as e:
        logger.error(f"Error in neural-guided selection: {e}", exc_info=True)
        return None  # Will fallback to vanilla MCTS in caller
