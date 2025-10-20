#!/usr/bin/env python3
"""
Test script for Abra model integration with Neural-Guided MCTS

This script tests that the Abra model can replace Minikazam:
1. Verify Abra model loads correctly
2. Check observation space compatibility (TeamPreviewObservationSpace)
3. Test policy inference with Abra
4. Verify integration with NeuralGuidedSearch

Usage:
    python test_abra_integration.py
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

def test_abra_loading():
    """Test that Abra model loads correctly."""
    logger.info("=" * 60)
    logger.info("TEST 1: Abra Model Loading")
    logger.info("=" * 60)

    try:
        from metamon.rl.pretrained import get_pretrained_model

        logger.info("Loading Abra model from Metamon registry...")
        abra_model = get_pretrained_model("Abra")

        logger.info(f"✓ Abra model class retrieved: {abra_model.__class__.__name__}")
        logger.info(f"  - Model name: {abra_model.model_name}")
        logger.info(f"  - Config: {abra_model.model_gin_config}")
        logger.info(f"  - Training config: {abra_model.train_gin_config}")
        logger.info(f"  - Observation space: {abra_model.observation_space.__class__.__name__}")
        logger.info(f"  - Default checkpoint: {abra_model.default_checkpoint}")

        # Verify observation space is TeamPreviewObservationSpace
        base_obs_space = abra_model.observation_space.base_obs_space
        logger.info(f"  - Base observation space: {base_obs_space.__class__.__name__}")

        if "TeamPreview" in base_obs_space.__class__.__name__:
            logger.info("✓ Observation space is compatible (TeamPreviewObservationSpace)")
        else:
            logger.warning(f"⚠ Unexpected observation space: {base_obs_space.__class__.__name__}")

        return True

    except Exception as e:
        logger.error(f"✗ Abra loading test failed: {e}", exc_info=True)
        return False


def test_abra_policy_provider():
    """Test LocalPolicyProvider with Abra model."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Abra Policy Provider")
    logger.info("=" * 60)

    try:
        from neural_mcts.local_policy import LocalPolicyProvider

        logger.info("Initializing LocalPolicyProvider with Abra...")
        policy_provider = LocalPolicyProvider(model_name="Abra", device="cpu")

        logger.info("✓ LocalPolicyProvider initialized successfully")

        # Check stats
        stats = policy_provider.get_stats()
        logger.info(f"✓ Provider stats:")
        for key, value in stats.items():
            logger.info(f"    {key}: {value}")

        if not stats['model_loaded']:
            logger.error("✗ Model not loaded!")
            return False

        return True

    except Exception as e:
        logger.error(f"✗ Policy provider test failed: {e}", exc_info=True)
        return False


def test_abra_inference():
    """Test Abra inference on a real battle state."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Abra Inference")
    logger.info("=" * 60)

    try:
        from neural_mcts.state_translator import StateTranslator
        from neural_mcts.local_policy import LocalPolicyProvider
        from metamon.env.wrappers import BattleAgainstBaseline
        from metamon.baselines.heuristic.basic import RandomBaseline
        from metamon.env.wrappers import get_metamon_teams

        # Initialize components with Abra
        translator = StateTranslator(observation_space_type="TeamPreviewObservationSpace")
        policy_provider = LocalPolicyProvider(model_name="Abra", device="cpu")

        logger.info("✓ Components initialized")

        # Create test battle
        teams = get_metamon_teams("gen9randombattle", "competitive")
        env = BattleAgainstBaseline(
            battle_format="gen9randombattle",
            team_set=teams[:5],
            opponent_type=RandomBaseline,
        )
        obs, info = env.reset()
        battle = env.unwrapped.battle

        logger.info(f"✓ Created test battle: {battle.battle_tag}")

        # Translate battle to observation
        metamon_obs = translator.translate(battle)
        logger.info(f"✓ Translated battle to observation")
        logger.info(f"    text_tokens shape: {metamon_obs['text_tokens'].shape}")
        logger.info(f"    numbers shape: {metamon_obs['numbers'].shape}")

        # Get policy from Abra
        policy = policy_provider.get_policy(metamon_obs)

        if policy is not None:
            logger.info(f"✓ Abra policy inference successful!")
            logger.info(f"    Policy shape: {policy.shape}")
            logger.info(f"    Policy sum: {policy.sum():.4f} (should be ~1.0)")
            logger.info(f"    Top 3 actions:")
            top_actions = sorted(enumerate(policy), key=lambda x: x[1], reverse=True)[:3]
            for idx, prob in top_actions:
                logger.info(f"      Action {idx}: {prob:.4f}")
        else:
            logger.error("✗ Policy inference returned None")
            env.close()
            return False

        env.close()
        return True

    except Exception as e:
        logger.error(f"✗ Abra inference test failed: {e}", exc_info=True)
        return False


def test_abra_neural_search():
    """Test NeuralGuidedSearch with Abra model."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 4: Abra with NeuralGuidedSearch")
    logger.info("=" * 60)

    try:
        from neural_mcts.state_translator import StateTranslator
        from neural_mcts.local_policy import LocalPolicyProvider
        from neural_mcts.neural_search import NeuralGuidedSearch
        from metamon.env.wrappers import BattleAgainstBaseline
        from metamon.baselines.heuristic.basic import RandomBaseline
        from metamon.env.wrappers import get_metamon_teams

        # Initialize components with Abra
        translator = StateTranslator(observation_space_type="TeamPreviewObservationSpace")
        policy_provider = LocalPolicyProvider(model_name="Abra", device="cpu")
        neural_search = NeuralGuidedSearch(
            policy_provider=policy_provider,
            state_translator=translator,
            c_puct=1.0,
            use_neural_prior=True,
            fallback_to_uniform=True
        )
        logger.info("✓ NeuralGuidedSearch initialized with Abra")

        # Create test battle
        teams = get_metamon_teams("gen9randombattle", "competitive")
        env = BattleAgainstBaseline(
            battle_format="gen9randombattle",
            team_set=teams[:5],
            opponent_type=RandomBaseline,
        )
        obs, info = env.reset()
        battle = env.unwrapped.battle

        # Create mock MCTS results
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

        mcts_results = [(mock_mcts_result, 1.0, 0)]
        battles = [battle]

        # Test move selection with Abra
        selected_move = neural_search.select_move_with_neural_prior(mcts_results, battles)

        logger.info(f"✓ Move selection with Abra successful!")
        logger.info(f"    Selected move: {selected_move}")

        env.close()
        return True

    except Exception as e:
        logger.error(f"✗ NeuralGuidedSearch test failed: {e}", exc_info=True)
        return False


def main():
    """Run all Abra integration tests."""
    logger.info("Starting Abra Integration Tests\n")

    results = []

    # Test 1: Abra model loading
    results.append(("Abra Loading", test_abra_loading()))

    # Test 2: Policy provider with Abra
    results.append(("Abra Policy Provider", test_abra_policy_provider()))

    # Test 3: Abra inference
    results.append(("Abra Inference", test_abra_inference()))

    # Test 4: NeuralGuidedSearch with Abra
    results.append(("Abra NeuralSearch", test_abra_neural_search()))

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
        logger.info("All Abra integration tests passed! ✓")
        logger.info("\nAbra model is ready to use with neural-guided MCTS")
        logger.info("The integration has been updated in:")
        logger.info("  - vendor/foul-play/fp/search/neural_guided.py")
        logger.info("  - vendor/neural-mcts/src/neural_mcts/local_policy.py")
        logger.info("  - vendor/neural-mcts/src/neural_mcts/neural_search.py")
        return 0
    else:
        logger.error("Some tests failed. ✗")
        return 1


if __name__ == "__main__":
    sys.exit(main())
