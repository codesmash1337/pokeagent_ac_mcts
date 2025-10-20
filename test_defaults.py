#!/usr/bin/env python3
"""
Test that battle translation works with default values for missing data
"""
import logging
import sys
import os

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

# Set environment
os.environ['METAMON_CACHE_DIR'] = '/Users/yadureddy/Documents/GitHub/clean/pokeagent_ac_mcts/.metamon_cache'

# Add paths
sys.path.insert(0, 'vendor/neural-mcts/src')
sys.path.insert(0, 'vendor/foul-play')
sys.path.insert(0, 'vendor/metamon')

print("Testing battle translation with defaults...")
print("=" * 60)

try:
    # Import components
    from neural_mcts.state_translator import StateTranslator
    from neural_mcts.local_policy import LocalPolicyProvider
    from fp.battle import Battle

    print("✓ Imports successful")

    # Create state translator
    translator = StateTranslator(observation_space_type="TeamPreviewObservationSpace")
    print("✓ StateTranslator initialized")

    # Create a minimal empty battle (like MCTS sampled states)
    battle = Battle("test-battle")
    print(f"✓ Empty battle created: {battle.battle_tag}")
    print(f"  - Has user: {hasattr(battle, 'user') and battle.user is not None}")
    print(f"  - Has user.active: {hasattr(battle, 'user') and battle.user and hasattr(battle.user, 'active') and battle.user.active is not None}")

    # Try to translate (should use defaults)
    print("\nAttempting translation with defaults...")
    obs = translator.translate(battle)
    print("✓ Translation successful!")
    print(f"  - Observation keys: {list(obs.keys())}")
    print(f"  - text_tokens shape: {obs['text_tokens'].shape}")
    print(f"  - numbers shape: {obs['numbers'].shape}")

    # Try to get policy
    print("\nTesting policy provider...")
    provider = LocalPolicyProvider(model_name="Abra", device="cpu")
    print("✓ Abra model loaded")

    policy = provider.get_policy(obs)
    if policy is not None:
        print(f"✓ Policy computed successfully!")
        print(f"  - Policy shape: {policy.shape}")
        print(f"  - Policy sum: {policy.sum():.4f}")
        print(f"  - Top 3 actions: {sorted(enumerate(policy), key=lambda x: x[1], reverse=True)[:3]}")
    else:
        print("✗ Policy returned None")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED!")
    print("Battle translation with defaults is working correctly.")
    print("Abra neural MCTS should now work with MCTS sampled states.")

except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
