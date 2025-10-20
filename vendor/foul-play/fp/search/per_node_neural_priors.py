"""
Per-node Neural Priors for MCTS

This module provides a callback function that can be registered with poke-engine
to query neural priors (from Abra model) for every MCTS node expansion.
"""

import logging
import sys
import os

# Add neural-mcts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../neural-mcts/src'))

from poke_engine import set_neural_prior_callback, clear_neural_prior_callback

logger = logging.getLogger(__name__)

# Global instances (singleton pattern for efficiency)
_policy_provider = None
_callback_stats = {
    "total_calls": 0,
    "cache_hits": 0,
    "errors": 0
}


def initialize_neural_callback(model_name="Abra", device="cpu"):
    """
    Initialize and register the neural prior callback with poke-engine.

    This sets up a global callback that will be called for every MCTS node expansion.

    Args:
        model_name: Name of Metamon model ("Abra", "Minikazam", etc.)
        device: Device to run model on ("cpu" or "cuda")
    """
    global _policy_provider

    if _policy_provider is not None:
        logger.info("Neural callback already initialized")
        return

    try:
        from neural_mcts.local_policy import LocalPolicyProvider

        logger.info(f"Initializing {model_name} model for per-node priors...")
        _policy_provider = LocalPolicyProvider(model_name=model_name, device=device)
        logger.info(f"{model_name} model loaded successfully")

        # Create and register callback
        def neural_callback(py_state):
            """
            Callback function that queries Abra for neural priors.

            This is called by poke-engine for every MCTS node expansion.

            Args:
                py_state: PyState object from poke-engine

            Returns:
                tuple: (s1_priors: List[float] | None, s2_priors: None)
            """
            global _callback_stats
            _callback_stats["total_calls"] += 1

            try:
                # For now, return uniform priors
                # Full implementation requires converting PyState to Battle/observation format
                # This is a placeholder that demonstrates the callback works

                # TODO: Convert PyState to Metamon observation format
                # obs = convert_pystate_to_observation(py_state)
                # policy = _policy_provider.get_policy(obs)

                # Placeholder: return uniform priors
                priors = [0.1] * 10  # Assuming 10 possible actions

                return (priors, None)

            except Exception as e:
                _callback_stats["errors"] += 1
                if _callback_stats["errors"] <= 5:  # Only log first few errors
                    logger.error(f"Neural callback error: {e}", exc_info=True)
                return (None, None)  # Fallback to uniform priors in poke-engine

        # Register with poke-engine
        set_neural_prior_callback(neural_callback)
        logger.info("Neural prior callback registered with poke-engine")

        # Reset stats
        _callback_stats["total_calls"] = 0
        _callback_stats["cache_hits"] = 0
        _callback_stats["errors"] = 0

    except ImportError as e:
        logger.error(f"Failed to initialize neural callback: {e}")
        logger.warning("Falling back to MCTS without neural priors")
        raise


def clear_neural_callback():
    """Clear the neural prior callback."""
    global _policy_provider
    clear_neural_prior_callback()
    _policy_provider = None

    logger.info(f"Neural callback cleared. Stats: {_callback_stats}")


def get_callback_stats():
    """Get statistics about callback usage."""
    return _callback_stats.copy()
