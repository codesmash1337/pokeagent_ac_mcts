"""
Test getting policy and value estimates from Abra model.

This file demonstrates how to:
1. Create a test observation state
2. Get policy (action probabilities) and value (Q-values) in a single forward pass
3. Use these for planning algorithms like MCTS

Run with: python vendor/metamon/test/test_inference.py
"""

import torch
import numpy as np
from metamon.rl.pretrained import get_pretrained_model
from metamon.interface import (
    UniversalState,
    UniversalPokemon,
    UniversalMove,
)
from test.ac_inference import (
    get_policy_and_value,
    PolicyValueInference,
    prepare_observation,
    init_inference_inputs,
    sample_action,
    get_best_action,
)


def create_test_state():
    """Create a simple test state for Gen 9 OU battle."""
    # Create a simple Pikachu
    pikachu_moves = [
        UniversalMove(
            name="thunderbolt",
            move_type="electric",
            category="special",
            base_power=90,
            accuracy=1.0,
            priority=0,
            current_pp=15,
            max_pp=24,
        ),
        UniversalMove(
            name="quickattack",
            move_type="normal",
            category="physical",
            base_power=40,
            accuracy=1.0,
            priority=1,
            current_pp=30,
            max_pp=48,
        ),
        UniversalMove(
            name="irontail",
            move_type="steel",
            category="physical",
            base_power=100,
            accuracy=0.75,
            priority=0,
            current_pp=15,
            max_pp=24,
        ),
        UniversalMove(
            name="voltswitch",
            move_type="electric",
            category="special",
            base_power=70,
            accuracy=1.0,
            priority=0,
            current_pp=20,
            max_pp=32,
        ),
    ]

    pikachu = UniversalPokemon(
        name="pikachu",
        types="electric notype",
        ability="static",
        item="lightball",
        hp_pct=0.85,
        status="nostatus",
        effect="noeffect",
        terastallized=False,
        tera_type="electric",
        base_hp=35,
        base_atk=55,
        base_def=40,
        base_spa=50,
        base_spd=50,
        base_spe=90,
        atk_boost=0,
        def_boost=0,
        spa_boost=0,
        spd_boost=0,
        spe_boost=0,
        accuracy_boost=0,
        evasion_boost=0,
        moves=pikachu_moves,
    )

    # Create opponent Charizard
    charizard_moves = [
        UniversalMove(
            name="flamethrower",
            move_type="fire",
            category="special",
            base_power=90,
            accuracy=1.0,
            priority=0,
            current_pp=15,
            max_pp=24,
        ),
    ]

    charizard = UniversalPokemon(
        name="charizard",
        types="fire flying",
        ability="blaze",
        item="no_item",
        hp_pct=1.0,
        status="nostatus",
        effect="noeffect",
        terastallized=False,
        tera_type="fire",
        base_hp=78,
        base_atk=84,
        base_def=78,
        base_spa=109,
        base_spd=85,
        base_spe=100,
        atk_boost=0,
        def_boost=0,
        spa_boost=0,
        spd_boost=0,
        spe_boost=0,
        accuracy_boost=0,
        evasion_boost=0,
        moves=charizard_moves,
    )

    # Create bench Pokemon (simplified)
    bench_pokemon = [UniversalPokemon.placeholder() for _ in range(5)]
    opponent_bench = [UniversalPokemon.placeholder() for _ in range(5)]

    # Create state
    state = UniversalState(
        our_active=pikachu,
        our_team=bench_pokemon,
        our_alive_count=6,
        opponent_active=charizard,
        opponent_team=opponent_bench,
        opponent_alive_count=6,
        weather="noweather",
        terrain="noterrain",
        trick_room=False,
        our_side_conditions="nosideconditions",
        opponent_side_conditions="nosideconditions",
        our_last_move="nomove",
        opponent_last_move="flamethrower",
        opponent_teampreview_species=[],
    )

    return state


def test_basic_inference():
    """Test basic policy and value inference."""
    print("="*70)
    print("TEST 1: BASIC POLICY AND VALUE INFERENCE")
    print("="*70)

    # Load Abra model
    print("\n1. Loading Abra model...")
    abra = get_pretrained_model("Abra")
    agent = abra.initialize_agent(checkpoint=40, log=False)
    policy = agent.policy
    policy.eval()
    device = agent.DEVICE
    print(f"   Device: {device}")
    print(f"   Num gammas: {len(policy.gammas)}")
    print(f"   Gammas: {policy.gammas.tolist()}")
    print(f"   Num actions: {abra.action_space.gym_space.n}")
    print(f"   Num critics: {policy.num_critics}")

    # Create test state
    print("\n2. Creating test state...")
    state = create_test_state()
    print(f"   Our Pokemon: {state.our_active.name} ({state.our_active.hp_pct*100:.0f}% HP)")
    print(f"   Opponent: {state.opponent_active.name} ({state.opponent_active.hp_pct*100:.0f}% HP)")
    print(f"   Our moves: {[m.name for m in state.our_active.moves]}")

    # Convert state to observation
    print("\n3. Converting state to observation...")
    obs = abra.observation_space.base_obs_space.state_to_obs(state)
    legal_actions = list(range(4))  # All 4 moves are legal

    obs_torch = prepare_observation(
        obs, legal_actions, abra.action_space.gym_space.n, device
    )
    print(f"   Observation keys: {list(obs_torch.keys())}")

    # Initialize inference inputs
    rl2s, time_idxs, hidden_state = init_inference_inputs(1, device, policy)

    # Get policy and value
    print("\n4. Running forward pass...")
    action_probs, q_values, state_value, _, _ = get_policy_and_value(
        policy, obs_torch, rl2s, time_idxs, hidden_state, gamma_idx=-1
    )

    print(f"   Action probs shape: {action_probs.shape}")
    print(f"   Q-values shape: {q_values.shape}")
    print(f"   State value: {state_value.item():.4f}")

    # Analyze policy
    print(f"\n5. Policy Analysis (Main Gamma γ={policy.gammas[-1].item():.3f}):")
    sorted_probs, sorted_actions = torch.sort(action_probs, descending=True)

    print(f"\n   Top 5 Actions by Probability:")
    print(f"   {'Rank':<6} {'Action':<8} {'Prob':<10} {'Q-Value':<10} {'Move'}")
    print(f"   {'-'*60}")
    for i in range(min(5, len(sorted_actions))):
        action_idx = sorted_actions[i].item()
        prob = sorted_probs[i].item()
        q_val = q_values[action_idx].item()

        if action_idx < 4:
            move_name = state.our_active.moves[action_idx].name
        elif action_idx < 9:
            move_name = f"switch-{action_idx - 3}"
        else:
            move_name = f"tera+{action_idx - 9}"

        print(f"   {i+1:<6} {action_idx:<8} {prob:<10.4f} {q_val:<10.4f} {move_name}")

    # Sample and greedy actions
    print("\n6. Action selection:")
    sampled = sample_action(action_probs, temperature=1.0, legal_actions=legal_actions)
    greedy = get_best_action(action_probs, legal_actions=legal_actions)

    print(f"   Sampled action: {sampled} ({state.our_active.moves[sampled].name})")
    print(f"   Greedy action: {greedy} ({state.our_active.moves[greedy].name})")


def test_stateful_inference():
    """Test stateful inference wrapper."""
    print("\n" + "="*70)
    print("TEST 2: STATEFUL INFERENCE WITH PolicyValueInference")
    print("="*70)

    # Load model
    print("\n1. Loading Abra model...")
    abra = get_pretrained_model("Abra")
    agent = abra.initialize_agent(checkpoint=40, log=False)
    policy = agent.policy
    policy.eval()
    device = agent.DEVICE

    # Create stateful inference
    print("\n2. Creating stateful inference wrapper...")
    inference = PolicyValueInference(policy, device)

    # Create test state
    state = create_test_state()
    obs = abra.observation_space.base_obs_space.state_to_obs(state)
    legal_actions = list(range(4))

    # Get policy and value
    print("\n3. Getting policy and value (step 0)...")
    action_probs, q_values, state_value = inference(obs, legal_actions)

    print(f"   State value: {state_value.item():.4f}")
    best_action = get_best_action(action_probs, legal_actions)
    print(f"   Best action: {best_action} ({state.our_active.moves[best_action].name})")
    print(f"   Prob: {action_probs[best_action].item():.4f}")
    print(f"   Q-value: {q_values[best_action].item():.4f}")

    # Simulate taking an action
    print("\n4. Updating after taking action...")
    inference.update(reward=0.1, action=best_action, done=False)
    print("   Internal state updated (step 1)")

    # Get next policy
    print("\n5. Getting policy for next step...")
    action_probs, q_values, state_value = inference(obs, legal_actions)
    print(f"   State value: {state_value.item():.4f}")

    # Reset
    print("\n6. Resetting inference state...")
    inference.reset()
    print("   Reset complete")


def test_multi_gamma():
    """Test multi-gamma analysis."""
    print("\n" + "="*70)
    print("TEST 3: MULTI-GAMMA ANALYSIS")
    print("="*70)

    # Load model
    abra = get_pretrained_model("Abra")
    agent = abra.initialize_agent(checkpoint=40, log=False)
    policy = agent.policy
    policy.eval()
    device = agent.DEVICE

    # Create test state
    state = create_test_state()
    obs = abra.observation_space.base_obs_space.state_to_obs(state)
    legal_actions = list(range(4))

    obs_torch = prepare_observation(
        obs, legal_actions, abra.action_space.gym_space.n, device
    )
    rl2s, time_idxs, hidden_state = init_inference_inputs(1, device, policy)

    print("\nHow different discount factors affect action selection:")
    print(f"\n{'Gamma':<10} {'Top Action':<12} {'Prob':<10} {'Q-Value':<10} {'Move'}")
    print("-"*60)

    for gamma_idx, gamma in enumerate(policy.gammas):
        action_probs, q_values, state_value, _, _ = get_policy_and_value(
            policy, obs_torch, rl2s, time_idxs, hidden_state, gamma_idx=gamma_idx
        )

        top_action = torch.argmax(action_probs).item()
        top_prob = action_probs[top_action].item()
        top_q = q_values[top_action].item()

        if top_action < 4:
            move_name = state.our_active.moves[top_action].name[:10]
        elif top_action < 9:
            move_name = f"switch-{top_action-3}"
        else:
            move_name = f"tera+{top_action-9}"

        print(f"{gamma.item():<10.3f} {top_action:<12} {top_prob:<10.4f} {top_q:<10.4f} {move_name}")


def main():
    """Run all tests."""
    test_basic_inference()
    test_stateful_inference()
    test_multi_gamma()

    print("\n" + "="*70)
    print("ALL TESTS COMPLETE!")
    print("="*70)
    print("\nKey Takeaways:")
    print("- Use get_policy_and_value() for one-off inference")
    print("- Use PolicyValueInference() for stateful sequential inference")
    print("- action_probs gives you π(a|s) for all actions")
    print("- q_values gives you Q(s,a) for all actions")
    print("- state_value is V(s) = E_a[Q(s,a)] under current policy")
    print("- Perfect for MCTS: policy prior + value estimate in one pass!")


if __name__ == "__main__":
    main()
