# This file is meant to test going from a state in different format -> abra state

import numpy as np
from metamon.interface import (
    UniversalState,
    UniversalPokemon,
    UniversalMove,
    TokenizedObservationSpace,
    get_observation_space,
)
from metamon.tokenizer import get_tokenizer
from metamon.backend.showdown_dex import Dex
from metamon.backend.replay_parser.str_parsing import clean_name, pokemon_name


def create_universal_move_from_dex(
    dex: Dex,
    move_id: str,
    current_pp: int | None = None,
) -> UniversalMove:
    """Create a UniversalMove from Dex data."""
    move_data = dex.moves[move_id]

    max_pp = move_data["pp"] * 1.6  # Showdown uses 1.6x the pp
    if current_pp is None:
        current_pp = max_pp

    return UniversalMove(
        name=clean_name(move_data["name"]),
        move_type=clean_name(move_data["type"]),
        category=clean_name(move_data["category"]),
        base_power=move_data["basePower"],
        accuracy=move_data["accuracy"] / 100.0
        if move_data["accuracy"] is not True
        else 1.0,
        priority=move_data["priority"],
        current_pp=current_pp,
        max_pp=max_pp,
    )


def create_universal_pokemon_from_dex(
    dex: Dex,
    species: str,
    move_ids: list[str],
    move_pps: list[int],
    ability: str,
    tera_type: str,
    hp_pct,
    terastallized: bool = False,
    item: str = "no_item",
    status: str = "nostatus",
    effect: str = "noeffect",
    stat_boosts: dict | None = None,
) -> UniversalPokemon:
    """Create a UniversalPokemon from Dex data."""
    # Get Pokemon data from dex
    poke_data = dex.get_pokedex_entry(species)

    # Check if ability is valid for this Pokemon
    ability = clean_name(ability)
    valid_abilities = {clean_name(a) for a in poke_data["abilities"].values()}
    if ability not in valid_abilities:
        raise ValueError(
            f"Invalid ability '{ability}' for {poke_data['name']}. Valid abilities are: {valid_abilities}"
        )
    tera_type = clean_name(tera_type)
    # Get types
    if terastallized:  # This is how we're handling terastalization afaik
        types_list = [tera_type, "notype"]
    else:
        types_list = [clean_name(poke_data["types"][0])]
        if len(poke_data["types"]) > 1:
            types_list.append(clean_name(poke_data["types"][1]))
        else:
            types_list.append("notype")
    types = " ".join(sorted(types_list))

    # Create moves
    moves = [
        create_universal_move_from_dex(dex, move_id, move_pps[i])
        for i, move_id in enumerate(move_ids[:4])
    ]

    # Get base stats
    base_stats = poke_data["baseStats"]

    # Get stat boosts
    if stat_boosts is None:
        stat_boosts = {
            "atk_boost": 0,
            "spa_boost": 0,
            "def_boost": 0,
            "spd_boost": 0,
            "spe_boost": 0,
            "accuracy_boost": 0,
            "evasion_boost": 0,
        }

    return UniversalPokemon(
        name=pokemon_name(poke_data["name"]),
        hp_pct=hp_pct,
        types=types,
        item=clean_name(item),
        ability=ability,
        lvl=100,  # Gen 9 OU uses level 100
        status=status,
        effect=effect,
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


def create_sample_universal_state() -> UniversalState:
    """Create a sample UniversalState for testing TeamPreviewObservationSpace.

    This represents a Gen 9 OU battle state with team preview information.
    Uses real Pokemon data from the showdown dex.
    """
    # Load Gen 9 dex
    dex = Dex.from_gen(9)

    # Player's active Pokemon (Kingambit)
    player_active = create_universal_pokemon_from_dex(
        dex=dex,
        species="Kingambit",
        move_ids=["suckerpunch", "ironhead", "kowtowcleave", "swordsdance"],
        move_pps=[8, 24, 12, 32],
        item="blackglasses",
        ability="Supreme Overlord",
        hp_pct=0.75,
        status="nostatus",
        tera_type="Dark",
        stat_boosts={
            "atk_boost": 2,
            "spa_boost": 0,
            "def_boost": 0,
            "spd_boost": 0,
            "spe_boost": 0,
            "accuracy_boost": 0,
            "evasion_boost": 0,
        },
    )

    # Opponent's active Pokemon (Great Tusk)
    opponent_active = create_universal_pokemon_from_dex(
        dex=dex,
        species="Great Tusk",
        move_ids=["earthquake", "closecombat", "rapidspin", "knockoff"],
        move_pps=[16, 8, 64, 32],
        item="leftovers",
        ability="Protosynthesis",
        hp_pct=0.60,
        status="nostatus",
        tera_type="Ground",
    )

    # Available switches for player
    available_switches = [
        create_universal_pokemon_from_dex(
            dex=dex,
            species="Gholdengo",
            move_ids=["shadowball", "makeitrain", "thunderbolt", "nastyplot"],
            move_pps=[24, 8, 24, 32],
            item="choicescarf",
            ability="Good as Gold",
            hp_pct=1.0,
            tera_type="Steel",
        ),
        create_universal_pokemon_from_dex(
            dex=dex,
            species="Dragapult",
            move_ids=["dragondarts", "uturn", "willowisp", "shadowball"],
            move_pps=[16, 32, 24, 24],
            item="heavydutyboots",
            ability="Infiltrator",
            hp_pct=0.85,
            tera_type="Ghost",
        ),
        create_universal_pokemon_from_dex(
            dex=dex,
            species="Landorus-Therian",
            move_ids=["earthquake", "uturn", "stealthrock", "knockoff"],
            move_pps=[16, 32, 32, 32],
            item="choicescarf",
            ability="Intimidate",
            hp_pct=0.50,
            tera_type="Ground",
        ),
        create_universal_pokemon_from_dex(
            dex=dex,
            species="Toxapex",
            move_ids=["venomdrench", "sludgebomb", "darkpulse", "gunkshot"],
            move_pps=[0, 0, 0, 0],
            item="blacksludge",
            ability="Regenerator",
            hp_pct=0.0,
            status="fnt",
            tera_type="Poison",
        ),
        create_universal_pokemon_from_dex(
            dex=dex,
            species="Zamazenta",
            move_ids=["bodypress", "irondefense", "crunch", "substitute"],
            move_pps=[0, 0, 0, 0],
            item="leftovers",
            ability="dauntless shield",
            hp_pct=0,
            status="fnt",
            tera_type="Fighting",
        ),
    ]

    # Previous moves
    player_prev_move = create_universal_move_from_dex(dex, "suckerpunch")
    opponent_prev_move = create_universal_move_from_dex(dex, "earthquake")

    # Team preview - opponent's full team revealed at start (Gen 9 feature)
    opponent_teampreview = [
        "greattusk",
        "dragapult",
        "gholdengo",
        "corviknight",
        "toxapex",
        "zamazenta",
    ]

    return UniversalState(
        format="gen9ou",
        player_active_pokemon=player_active,
        opponent_active_pokemon=opponent_active,
        available_switches=available_switches,
        player_prev_move=player_prev_move,
        opponent_prev_move=opponent_prev_move,
        opponents_remaining=5,
        player_conditions="noconditions",
        opponent_conditions="stealthrock",
        weather="noweather",
        battle_field="nofield",
        forced_switch=False,
        battle_won=False,
        battle_lost=False,
        can_tera=True,
        opponent_teampreview=opponent_teampreview,
    )


def test_teampreview_observation_space():
    """Test TeamPreviewObservationSpace with a sample state."""
    # Create observation space
    obs_space = get_observation_space("TeamPreviewObservationSpace")

    # Create sample state
    state = create_sample_universal_state()

    # Generate observation
    obs = obs_space(state)

    print("=== TeamPreviewObservationSpace Test ===\n")
    print(f"Observation keys: {obs.keys()}\n")

    # Text observation
    print("Text observation:")
    print(f"  Type: {type(obs['text'])}")
    print(f"  Shape: {obs['text'].shape}")
    print("  Content preview (first 200 chars):")
    text_content = obs["text"].item()
    print(f"  {text_content[:200]}...\n")

    # Count tokens in text
    tokens = text_content.split()
    print(f"  Total tokens: {len(tokens)}")
    print("  Expected: 106 (87 base + 13 expanded + 6 teampreview)\n")

    # Numbers observation
    print("Numbers observation:")
    print(f"  Type: {type(obs['numbers'])}")
    print(f"  Shape: {obs['numbers'].shape}")
    print("  Expected shape: (55,)")
    print(f"  Min value: {obs['numbers'].min():.3f}")
    print(f"  Max value: {obs['numbers'].max():.3f}")
    print(f"  Sample values (first 10): {obs['numbers'][:10]}\n")

    # Verify opponent team preview in text
    print("Team preview verification:")
    opponent_team = [
        "greattusk",
        "dragapult",
        "gholdengo",
        "corviknight",
        "toxapex",
        "zamazenta",
    ]
    for pokemon in opponent_team:
        if pokemon in text_content:
            print(f"  ✓ {pokemon} found in observation")
        else:
            print(f"  ✗ {pokemon} NOT found in observation")

    print("\nFull text observation:")
    print(f"  {text_content}\n")

    return obs


def test_tokenized_observation_space():
    """Test TokenizedObservationSpace (what Abra actually uses)."""
    # Create tokenizer and observation space
    tokenizer = get_tokenizer("DefaultObservationSpace-v1")
    base_obs_space = get_observation_space("TeamPreviewObservationSpace")
    tokenized_obs_space = TokenizedObservationSpace(
        base_obs_space=base_obs_space,
        tokenizer=tokenizer,
    )

    # Create sample state
    state = create_sample_universal_state()

    # Generate observation
    obs = tokenized_obs_space(state)

    print("\n=== TokenizedObservationSpace Test (Abra's actual input) ===\n")
    print(f"Observation keys: {obs.keys()}\n")

    # Text tokens observation
    print("Text tokens observation:")
    print(f"  Type: {type(obs['text_tokens'])}")
    print(f"  Shape: {obs['text_tokens'].shape}")
    print(f"  Dtype: {obs['text_tokens'].dtype}")
    print("  Expected shape: (106,)")
    print(f"  Min token ID: {obs['text_tokens'].min()}")
    print(f"  Max token ID: {obs['text_tokens'].max()}")
    print(f"  Sample token IDs (first 20): {obs['text_tokens'][:20]}\n")
    print(f"  Unknown tokens (ID=-1): {np.sum(obs['text_tokens'] == -1)}")

    # Numbers observation (unchanged from base)
    print("\nNumbers observation:")
    print(f"  Type: {type(obs['numbers'])}")
    print(f"  Shape: {obs['numbers'].shape}")
    print("  Expected shape: (55,)\n")

    # Verify tokenizer vocabulary size
    print("Tokenizer info:")
    print(f"  Name: {tokenizer.name}")
    print(f"  Vocabulary size: {len(tokenizer)}")

    return obs


if __name__ == "__main__":
    # Run tests
    print("Running TeamPreviewObservationSpace tests...\n")

    obs1 = test_teampreview_observation_space()
    obs2 = test_tokenized_observation_space()

    print("\n" + "=" * 60)
    print("Tests completed successfully!")
    print("=" * 60)
