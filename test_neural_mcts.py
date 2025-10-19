#!/usr/bin/env python3
"""
Test script for Neural-Guided MCTS integration

This script tests the complete pipeline:
1. StateTranslator: Battle -> Metamon observation
2. LocalPolicyProvider: observation -> policy distribution
3. NeuralGuidedSearch: combines MCTS + neural policies

Usage:
    python test_neural_mcts.py
"""

import logging
import sys
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)-8s %(message)s'
)
logger = logging.getLogger(__name__)

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'vendor/neural-mcts/src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'vendor/metamon'))

def test_state_translator():
    """Test StateTranslator initialization and basic translation."""
    logger.info("=" * 60)
    logger.info("TEST 1: StateTranslator")
    logger.info("=" * 60)

    try:
        from neural_mcts.state_translator import StateTranslator
        from metamon.env.wrappers import BattleAgainstBaseline
        from metamon.baselines.heuristic.basic import RandomBaseline
        from metamon.env.wrappers import get_metamon_teams

        # Initialize translator
        translator = StateTranslator(observation_space_type="TeamPreviewObservationSpace")
        logger.info("✓ StateTranslator initialized successfully")

        # Create a simple battle environment
        teams = get_metamon_teams("gen9randombattle", "competitive")
        env = BattleAgainstBaseline(
            battle_format="gen9randombattle",
            team_set=teams[:5],  # Just a few teams
            opponent_type=RandomBaseline,
        )

        # Get initial observation
        obs, info = env.reset()
        battle = env.unwrapped.battle

        logger.info(f"✓ Created test battle: {battle.battle_tag}")

        # Translate battle to observation
        metamon_obs = translator.translate(battle)

        logger.info(f"✓ Translation successful!")
        logger.info(f"  - text_tokens shape: {metamon_obs['text_tokens'].shape}")
        logger.info(f"  - numbers shape: {metamon_obs['numbers'].shape}")
        logger.info(f"  - illegal_actions shape: {metamon_obs['illegal_actions'].shape}")

        # Get legal actions
        legal_actions = translator.get_legal_actions(battle)
        logger.info(f"✓ Legal actions: {legal_actions}")

        env.close()
        return True

    except Exception as e:
        logger.error(f"✗ StateTranslator test failed: {e}", exc_info=True)
        return False


def test_policy_provider():
    """Test LocalPolicyProvider initialization and policy inference."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: LocalPolicyProvider")
    logger.info("=" * 60)

    try:
        from neural_mcts.state_translator import StateTranslator
        from neural_mcts.local_policy import LocalPolicyProvider
        from metamon.env.wrappers import BattleAgainstBaseline
        from metamon.baselines.heuristic.basic import RandomBaseline
        from metamon.env.wrappers import get_metamon_teams

        # Initialize components
        translator = StateTranslator(observation_space_type="TeamPreviewObservationSpace")
        policy_provider = LocalPolicyProvider(model_name="Minikazam", device="cpu")
        logger.info("✓ LocalPolicyProvider initialized successfully")

        # Create test battle
        teams = get_metamon_teams("gen9randombattle", "competitive")
        env = BattleAgainstBaseline(
            battle_format="gen9randombattle",
            team_set=teams[:5],
            opponent_type=RandomBaseline,
        )
        obs, info = env.reset()
        battle = env.unwrapped.battle

        # Get observation and policy
        metamon_obs = translator.translate(battle)
        policy = policy_provider.get_policy(metamon_obs)

        if policy is not None:
            logger.info(f"✓ Policy inference successful!")
            logger.info(f"  - Policy shape: {policy.shape}")
            logger.info(f"  - Policy sum: {policy.sum():.4f} (should be ~1.0)")
            logger.info(f"  - Top 3 actions:")
            top_actions = sorted(enumerate(policy), key=lambda x: x[1], reverse=True)[:3]
            for idx, prob in top_actions:
                logger.info(f"    Action {idx}: {prob:.4f}")
        else:
            logger.error("✗ Policy inference returned None")
            env.close()
            return False

        # Test stats
        stats = policy_provider.get_stats()
        logger.info(f"✓ Provider stats: {stats}")

        env.close()
        return True

    except Exception as e:
        logger.error(f"✗ LocalPolicyProvider test failed: {e}", exc_info=True)
        return False


def test_neural_search():
    """Test NeuralGuidedSearch with mock MCTS results."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: NeuralGuidedSearch")
    logger.info("=" * 60)

    try:
        from neural_mcts.state_translator import StateTranslator
        from neural_mcts.local_policy import LocalPolicyProvider
        from neural_mcts.neural_search import NeuralGuidedSearch
        from metamon.env.wrappers import BattleAgainstBaseline
        from metamon.baselines.heuristic.basic import RandomBaseline
        from metamon.env.wrappers import get_metamon_teams

        # Initialize components
        translator = StateTranslator(observation_space_type="TeamPreviewObservationSpace")
        policy_provider = LocalPolicyProvider(model_name="Minikazam", device="cpu")
        neural_search = NeuralGuidedSearch(
            policy_provider=policy_provider,
            state_translator=translator,
            c_puct=1.0,
            use_neural_prior=True,
            fallback_to_uniform=True
        )
        logger.info("✓ NeuralGuidedSearch initialized successfully")

        # Create test battles
        teams = get_metamon_teams("gen9randombattle", "competitive")
        env = BattleAgainstBaseline(
            battle_format="gen9randombattle",
            team_set=teams[:5],
            opponent_type=RandomBaseline,
        )
        obs, info = env.reset()
        battle = env.unwrapped.battle

        # Create mock MCTS results
        # In real usage, these come from poke-engine MCTS
        from collections import namedtuple

        MctsResult = namedtuple('MctsResult', ['side_one', 'total_visits'])
        MoveOption = namedtuple('MoveOption', ['move_choice', 'visits', 'total_score'])

        mock_mcts_result = MctsResult(
            side_one=[
                MoveOption(move_choice="move 0", visits=100, total_score=50.0),
                MoveOption(move_choice="move 1", visits=80, total_score=40.0),
                MoveOption(move_choice="switch 0", visits=20, total_score=10.0),
            ],
            total_visits=200
        )

        mcts_results = [(mock_mcts_result, 1.0, 0)]  # (result, sample_chance, index)
        battles = [battle]

        # Test move selection
        selected_move = neural_search.select_move_with_neural_prior(mcts_results, battles)

        logger.info(f"✓ Move selection successful!")
        logger.info(f"  - Selected move: {selected_move}")

        env.close()
        return True

    except Exception as e:
        logger.error(f"✗ NeuralGuidedSearch test failed: {e}", exc_info=True)
        return False


def main():
    """Run all tests."""
    logger.info("Starting Neural-MCTS Integration Tests\n")

    results = []

    # Test 1: StateTranslator
    results.append(("StateTranslator", test_state_translator()))

    # Test 2: LocalPolicyProvider
    results.append(("LocalPolicyProvider", test_policy_provider()))

    # Test 3: NeuralGuidedSearch
    results.append(("NeuralGuidedSearch", test_neural_search()))

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)

    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{test_name:25s} {status}")
        if not passed:
            all_passed = False

    logger.info("=" * 60)

    if all_passed:
        logger.info("All tests passed! ✓")
        logger.info("\nYou can now use neural-guided MCTS with:")
        logger.info("  --use-neural-mcts --neural-c-puct 1.0")
        return 0
    else:
        logger.error("Some tests failed. ✗")
        return 1


if __name__ == "__main__":
    sys.exit(main())
