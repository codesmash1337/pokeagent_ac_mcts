#!/usr/bin/env python3
"""
Test script to verify PyMctsResult attribute fixes.

This tests that the Rust PyMctsResult object attributes (s1, s2, iteration_count)
are correctly accessed and converted to picklable Python structures.
"""

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_vanilla_mcts():
    """Test vanilla MCTS result conversion."""
    from poke_engine import State, mcts

    # Create a simple test state
    state = State()

    # Run a short MCTS search
    logger.info("Testing vanilla MCTS...")
    result = mcts(state, 100)  # 100ms search

    # Verify attributes exist
    assert hasattr(result, 's1'), "PyMctsResult missing 's1' attribute"
    assert hasattr(result, 's2'), "PyMctsResult missing 's2' attribute"
    assert hasattr(result, 'iteration_count'), "PyMctsResult missing 'iteration_count' attribute"

    # Verify they are the correct types
    assert isinstance(result.s1, list), f"s1 should be list, got {type(result.s1)}"
    assert isinstance(result.s2, list), f"s2 should be list, got {type(result.s2)}"
    assert isinstance(result.iteration_count, int), f"iteration_count should be int, got {type(result.iteration_count)}"

    # Verify we can access move node attributes
    if len(result.s1) > 0:
        node = result.s1[0]
        assert hasattr(node, 'move_choice'), "MoveNode missing 'move_choice'"
        assert hasattr(node, 'visits'), "MoveNode missing 'visits'"
        assert hasattr(node, 'total_score'), "MoveNode missing 'total_score'"
        logger.info(f"  First move: {node.move_choice}, visits: {node.visits}, score: {node.total_score}")

    logger.info(f"  Total iterations: {result.iteration_count}")
    logger.info("  ✓ Vanilla MCTS test passed")
    return result

def test_puct_mcts():
    """Test PUCT MCTS result conversion."""
    try:
        from poke_engine import State, mcts_with_puct
    except ImportError:
        logger.warning("  ⚠ PUCT MCTS not available, skipping")
        return None

    # Create a simple test state
    state = State()

    # Run a short PUCT MCTS search with uniform priors
    logger.info("Testing PUCT MCTS...")
    result = mcts_with_puct(state, 100, c_puct=1.5)  # 100ms search

    # Verify attributes exist (same as vanilla)
    assert hasattr(result, 's1'), "PyMctsResult missing 's1' attribute"
    assert hasattr(result, 's2'), "PyMctsResult missing 's2' attribute"
    assert hasattr(result, 'iteration_count'), "PyMctsResult missing 'iteration_count' attribute"

    # Verify they are the correct types
    assert isinstance(result.s1, list), f"s1 should be list, got {type(result.s1)}"
    assert isinstance(result.s2, list), f"s2 should be list, got {type(result.s2)}"
    assert isinstance(result.iteration_count, int), f"iteration_count should be int, got {type(result.iteration_count)}"

    if len(result.s1) > 0:
        node = result.s1[0]
        logger.info(f"  First move: {node.move_choice}, visits: {node.visits}, score: {node.total_score}")

    logger.info(f"  Total iterations: {result.iteration_count}")
    logger.info("  ✓ PUCT MCTS test passed")
    return result

def test_picklable_conversion():
    """Test conversion to picklable format."""
    from poke_engine import State, mcts

    logger.info("Testing picklable conversion...")

    # Create and run MCTS
    state = State()
    result = mcts(state, 100)

    # Test that we can successfully extract all attributes
    # (actual pickling will be tested in multiprocessing context)
    side_one_count = len(result.s1)
    side_two_count = len(result.s2)
    total = result.iteration_count

    # Verify we can access all node attributes
    for node in result.s1:
        _ = node.move_choice
        _ = node.visits
        _ = node.total_score

    logger.info(f"  Extracted {side_one_count} moves for side 1")
    logger.info(f"  Extracted {side_two_count} moves for side 2")
    logger.info(f"  Total iterations: {total}")
    logger.info("  ✓ Attribute extraction test passed")

def test_foul_play_integration():
    """Test the actual foul-play integration."""
    try:
        from fp.search.main import get_result_from_mcts, get_result_from_puct_mcts
        from poke_engine import State

        logger.info("Testing foul-play integration...")

        # Create a simple state and serialize it
        state = State()
        state_str = state.to_string()

        # Test vanilla MCTS
        result = get_result_from_mcts(state_str, 100, 0)
        assert hasattr(result, 'side_one'), "Result missing 'side_one'"
        assert hasattr(result, 'total_visits'), "Result missing 'total_visits'"
        logger.info(f"  Vanilla MCTS returned {len(result.side_one)} moves")

        # Test PUCT MCTS if available
        try:
            result = get_result_from_puct_mcts(state_str, 100, 0)
            assert hasattr(result, 'side_one'), "Result missing 'side_one'"
            assert hasattr(result, 'total_visits'), "Result missing 'total_visits'"
            logger.info(f"  PUCT MCTS returned {len(result.side_one)} moves")
        except ImportError:
            logger.warning("  ⚠ PUCT MCTS not available in foul-play")

        logger.info("  ✓ Foul-play integration test passed")

    except ImportError as e:
        logger.warning(f"  ⚠ Foul-play not available: {e}")

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Testing PyMctsResult Attribute Fixes")
    logger.info("=" * 60)

    try:
        test_vanilla_mcts()
        test_puct_mcts()
        test_picklable_conversion()
        test_foul_play_integration()

        logger.info("=" * 60)
        logger.info("✓ All tests passed!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"✗ Test failed: {e}", exc_info=True)
        raise
