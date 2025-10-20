#!/usr/bin/env python3
"""
Quick test to debug battle translation issues
"""
import logging
import sys
import os

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')

# Set environment
os.environ['METAMON_CACHE_DIR'] = '/Users/yadureddy/Documents/GitHub/clean/pokeagent_ac_mcts/.metamon_cache'

# Add paths
sys.path.insert(0, 'vendor/neural-mcts/src')
sys.path.insert(0, 'vendor/foul-play')
sys.path.insert(0, 'vendor/metamon')

print("Testing battle translation...")

try:
    # Import state translator
    from neural_mcts.state_translator import StateTranslator
    print("✓ StateTranslator imported")

    # Import foul-play Battle
    from fp.battle import Battle
    print("✓ foul-play Battle imported")

    # Create state translator
    translator = StateTranslator(observation_space_type="TeamPreviewObservationSpace")
    print("✓ StateTranslator initialized")

    # Create a minimal foul-play Battle for testing
    print("\nCreating test battle...")
    battle = Battle("test-battle")
    print(f"✓ Battle created: {battle.battle_tag}")

    # Try to translate
    print("\nAttempting translation...")
    obs = translator.translate(battle)
    print("✓ Translation successful!")
    print(f"  Observation keys: {obs.keys()}")

except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓ All tests passed!")
