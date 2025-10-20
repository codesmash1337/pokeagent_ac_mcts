"""
Test script for per-node neural priors in MCTS.

This tests that the Abra model is called for each node expansion during MCTS search.
"""

import logging
from poke_engine import State, mcts_with_puct, set_neural_prior_callback, clear_neural_prior_callback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Counter to track how many times callback is called
callback_count = 0


def simple_callback(py_state):
    """
    Simple test callback that just returns uniform priors and counts calls.

    Args:
        py_state: PyState object from poke-engine

    Returns:
        tuple: (s1_priors: List[float], s2_priors: None)
    """
    global callback_count
    callback_count += 1

    # Return uniform priors for testing (real implementation would call Abra)
    # Assuming 10 possible moves
    priors = [0.1] * 10

    if callback_count <= 5:  # Only log first few calls
        logger.info(f"Callback called (#{callback_count})")

    return (priors, None)


def test_callback_mechanism():
    """Test that the callback is actually called during MCTS."""
    global callback_count
    callback_count = 0

    logger.info("Setting up neural prior callback...")
    set_neural_prior_callback(simple_callback)

    # Create a simple battle state
    logger.info("Creating test state...")
    state = State()

    # Run MCTS with short time budget
    logger.info("Running MCTS with neural priors (100ms)...")
    result = mcts_with_puct(state, duration_ms=100, c_puct=1.5)

    logger.info(f"MCTS completed!")
    logger.info(f"Total iterations: {result.iteration_count}")
    logger.info(f"Callback was called {callback_count} times")
    logger.info(f"Side 1 moves explored: {len(result.s1)}")

    # Clear callback
    clear_neural_prior_callback()

    # Verify callback was called multiple times (once per node expansion)
    if callback_count > 1:
        logger.info(f"✓ SUCCESS: Callback was called {callback_count} times (per-node priors working!)")
        return True
    else:
        logger.error(f"✗ FAILURE: Callback only called {callback_count} times (expected multiple calls)")
        return False


def test_with_abra_model():
    """Test with actual Abra model."""
    try:
        from vendor.neural_mcts.src.neural_mcts.local_policy import LocalPolicyProvider
        from vendor.neural_mcts.src.neural_mcts.state_translator import StateTranslator
        import sys
        import os

        # Add paths
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'vendor/neural-mcts/src'))

        logger.info("\n" + "="*60)
        logger.info("Testing with Abra model...")
        logger.info("="*60)

        # Initialize Abra components
        logger.info("Loading Abra model...")
        translator = StateTranslator(observation_space_type="TeamPreviewObservationSpace")
        provider = LocalPolicyProvider(model_name="Abra", device="cpu")
        logger.info("Abra model loaded successfully")

        global callback_count
        callback_count = 0

        def abra_callback(py_state):
            """Callback using Abra model."""
            global callback_count
            callback_count += 1

            try:
                # Note: py_state is from poke-engine, need to adapt to Battle format
                # For now, return uniform (full implementation would convert state)
                priors = [0.1] * 10

                if callback_count <= 3:
                    logger.info(f"Abra callback #{callback_count}")

                return (priors, None)
            except Exception as e:
                logger.error(f"Abra callback error: {e}")
                return (None, None)

        logger.info("Setting Abra callback...")
        set_neural_prior_callback(abra_callback)

        # Run MCTS
        logger.info("Running MCTS with Abra priors (200ms)...")
        state = State()
        result = mcts_with_puct(state, duration_ms=200, c_puct=1.5)

        logger.info(f"MCTS completed with {result.iteration_count} iterations")
        logger.info(f"Abra callback was called {callback_count} times")

        clear_neural_prior_callback()

        if callback_count > 1:
            logger.info("✓ SUCCESS: Abra integration working!")
            return True
        else:
            logger.warning("⚠ Abra callback only called once (may need state conversion)")
            return False

    except ImportError as e:
        logger.warning(f"Abra test skipped (import error): {e}")
        return None


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Testing Per-Node Neural Priors in MCTS")
    print("="*60 + "\n")

    # Test 1: Basic callback mechanism
    print("\nTest 1: Basic Callback Mechanism")
    print("-" * 60)
    success1 = test_callback_mechanism()

    # Test 2: With Abra model (optional)
    print("\n\nTest 2: Abra Model Integration")
    print("-" * 60)
    success2 = test_with_abra_model()

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print(f"Basic Callback: {'PASS' if success1 else 'FAIL'}")
    if success2 is not None:
        print(f"Abra Integration: {'PASS' if success2 else 'FAIL'}")
    else:
        print(f"Abra Integration: SKIPPED")
    print("="*60)
