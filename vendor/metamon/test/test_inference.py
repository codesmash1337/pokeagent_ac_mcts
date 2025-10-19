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
    get_observation_space,
    TokenizedObservationSpace,
)
from metamon.tokenizer import get_tokenizer
from metamon.backend.showdown_dex import Dex
from metamon.backend.replay_parser.str_parsing import clean_name, pokemon_name
from ac_inference import (
    get_policy_and_value,
    PolicyValueInference,
    prepare_observation,
    init_inference_inputs,
    sample_action,
    get_best_action,
)


def create_universal_move_from_dex(dex, move_id, current_pp=None):
    """Create a UniversalMove from Dex data."""
    from metamon.interface import UniversalMove

    move_data = dex.moves[move_id]
    max_pp = move_data["pp"] * 1.6
    if current_pp is None:
        current_pp = max_pp

    return UniversalMove(
        name=clean_name(move_data["name"]),
        move_type=clean_name(move_data["type"]),
        category=clean_name(move_data["category"]),
        base_power=move_data.get("basePower", 0),
        accuracy=move_data.get("accuracy", 100) / 100.0 if move_data.get("accuracy") != True else 1.0,
        priority=move_data.get("priority", 0),
        current_pp=current_pp,
        max_pp=max_pp,
    )


def create_universal_pokemon_from_dex(dex, species, move_ids, move_pps, ability, tera_type, hp_pct, item="unknownitem", status="nostatus", terastallized=False, stat_boosts=None):
    """Create a UniversalPokemon from Dex data."""
    from metamon.interface import UniversalPokemon

    poke_data = dex.get_pokedex_entry(species)
    ability = clean_name(ability)
    tera_type = clean_name(tera_type)

    # Get types
    if terastallized:
        types_list = [tera_type, "notype"]
    else:
        types_list = [clean_name(poke_data["types"][0])]
        if len(poke_data["types"]) > 1:
            types_list.append(clean_name(poke_data["types"][1]))
        else:
            types_list.append("notype")
    types = " ".join(sorted(types_list))

    # Create moves
    moves = [create_universal_move_from_dex(dex, move_id, move_pps[i]) for i, move_id in enumerate(move_ids[:4])]

    # Get base stats
    base_stats = poke_data["baseStats"]

    # Get stat boosts
    if stat_boosts is None:
        stat_boosts = {
            "atk_boost": 0, "spa_boost": 0, "def_boost": 0,
            "spd_boost": 0, "spe_boost": 0, "accuracy_boost": 0, "evasion_boost": 0,
        }

    return UniversalPokemon(
        name=pokemon_name(poke_data["name"]),
        hp_pct=hp_pct,
        types=types,
        item=clean_name(item),
        ability=ability,
        lvl=100,
        status=status,
        effect="noeffect",
        moves=moves,
        base_atk=base_stats["atk"],
        base_spa=base_stats["spa"],
        base_def=base_stats["def"],
        base_spd=base_stats["spd"],
        base_spe=base_stats["spe"],
        base_hp=base_stats["hp"],
        tera_type=tera_type,
        base_species=pokemon_name(poke_data.get("baseSpecies", poke_data["name"])),
        **stat_boosts,
    )


def create_test_state():
    """
    Create test state: Choice Specs Iron Valiant vs Iron Defense Zamazenta (50% HP).
    All other slots fainted.
    """
    dex = Dex.from_gen(9)

    # Player: Choice Specs Iron Valiant with Moonblast
    iron_valiant = create_universal_pokemon_from_dex(
        dex=dex,
        species="Iron Valiant",
        move_ids=["moonblast", "closecombat", "psyshock", "trick"],
        move_pps=[24, 8, 16, 16],
        ability="Quark Drive",
        tera_type="Fairy",
        hp_pct=1.0,
        item="choicespecs",
        status="nostatus",
    )

    # Opponent: Iron Defense Zamazenta at 50% HP
    zamazenta = create_universal_pokemon_from_dex(
        dex=dex,
        species="Zamazenta",
        move_ids=["bodypress", "irondefense", "crunch", "substitute"],
        move_pps=[16, 24, 24, 16],
        ability="Dauntless Shield",
        tera_type="Fighting",
        hp_pct=0.5,
        item="leftovers",
        status="nostatus",
        stat_boosts={
            "atk_boost": 0,
            "spa_boost": 0,
            "def_boost": 2,  # +2 Defense from Iron Defense
            "spd_boost": 0,
            "spe_boost": 0,
            "accuracy_boost": 0,
            "evasion_boost": 0,
        },
    )

    # All other slots fainted
    fainted_pokemon = []
    for _ in range(5):
        fainted = create_universal_pokemon_from_dex(
            dex=dex,
            species="Toxapex",  # Placeholder species
            move_ids=["scald", "toxic", "recover", "haze"],
            move_pps=[0, 0, 0, 0],
            ability="Regenerator",
            tera_type="Poison",
            hp_pct=0.0,
            item="blacksludge",
            status="fnt",
        )
        fainted_pokemon.append(fainted)

    # Previous moves
    player_prev_move = create_universal_move_from_dex(dex, "moonblast")
    opponent_prev_move = create_universal_move_from_dex(dex, "irondefense")

    # Create state
    state = UniversalState(
        format="gen9ou",
        player_active_pokemon=iron_valiant,
        opponent_active_pokemon=zamazenta,
        available_switches=fainted_pokemon,
        player_prev_move=player_prev_move,
        opponent_prev_move=opponent_prev_move,
        opponents_remaining=1,
        player_conditions="noconditions",
        opponent_conditions="noconditions",
        weather="noweather",
        battle_field="nofield",
        forced_switch=False,
        battle_won=False,
        battle_lost=False,
        can_tera=True,
        opponent_teampreview=["zamazenta", "toxapex", "toxapex", "toxapex", "toxapex", "toxapex"],
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
    print(f"   Our Pokemon: {state.player_active_pokemon.name} ({state.player_active_pokemon.hp_pct*100:.0f}% HP)")
    print(f"   Opponent: {state.opponent_active_pokemon.name} ({state.opponent_active_pokemon.hp_pct*100:.0f}% HP)")
    print(f"   Our moves: {[m.name for m in state.player_active_pokemon.moves]}")

    # Convert state to observation
    print("\n3. Converting state to observation...")
    obs = abra.observation_space.state_to_obs(state)  # Use full observation space (includes tokenization)
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
            move_name = state.player_active_pokemon.moves[action_idx].name
        elif action_idx < 9:
            move_name = f"switch-{action_idx - 3}"
        else:
            move_name = f"tera+{action_idx - 9}"

        print(f"   {i+1:<6} {action_idx:<8} {prob:<10.4f} {q_val:<10.4f} {move_name}")

    # Sample and greedy actions
    print("\n6. Action selection:")
    sampled = sample_action(action_probs, temperature=1.0, legal_actions=legal_actions)
    greedy = get_best_action(action_probs, legal_actions=legal_actions)

    print(f"   Sampled action: {sampled} ({state.player_active_pokemon.moves[sampled].name})")
    print(f"   Greedy action: {greedy} ({state.player_active_pokemon.moves[greedy].name})")

    # Assertions for test scenario
    print("\n7. Test Assertions:")
    best_action = torch.argmax(action_probs).item()
    print(f"   Best action: {best_action}")
    if best_action < 4:
        print(f"      Best move: {state.player_active_pokemon.moves[best_action].name}")
    print(f"   State value: {state_value.item():.4f}")

    # Basic sanity checks
    assert action_probs.sum().item() > 0.99, "Action probs should sum to ~1"
    assert len(q_values) == 13, f"Should have Q-values for all 13 actions"
    print("   ✓ Basic sanity checks passed")


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
    obs = abra.observation_space.state_to_obs(state)  # Use full observation space
    legal_actions = list(range(4))

    # Get policy and value
    print("\n3. Getting policy and value (step 0)...")
    action_probs, q_values, state_value = inference(obs, legal_actions)

    print(f"   State value: {state_value.item():.4f}")
    best_action = get_best_action(action_probs, legal_actions)
    print(f"   Best action: {best_action} ({state.player_active_pokemon.moves[best_action].name})")
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
    obs = abra.observation_space.state_to_obs(state)  # Use full observation space
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
            move_name = state.player_active_pokemon.moves[top_action].name[:10]
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
