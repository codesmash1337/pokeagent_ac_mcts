#!/usr/bin/env python3
"""
Create example poke-engine State objects for testing observation generation.

This creates pickled State objects that can be used with compare_observations.py
"""

import sys
import pickle
from pathlib import Path

# Add vendor paths
sys.path.insert(0, str(Path(__file__).parent / "vendor" / "poke-engine" / "poke-engine-py" / "python"))

from poke_engine import State, Side, Pokemon, Move, PokemonMoves


def create_simple_battle_state():
    """Create a simple mid-battle state with Kingambit vs Landorus-Therian."""

    # Create Kingambit (active, damaged, +1 attack boost)
    kingambit_moves = PokemonMoves(
        m0=Move(id="ironhead", pp=24, disabled=False),
        m1=Move(id="suckerpunch", pp=7, disabled=False),
        m2=Move(id="lowkick", pp=32, disabled=False),
        m3=Move(id="swordsdance", pp=31, disabled=False),
    )

    kingambit = Pokemon(
        id="kingambit",
        level=100,
        types=("dark", "steel"),
        base_types=("dark", "steel"),
        hp=104,  # ~32% HP
        maxhp=320,
        ability="supremeoverlord",
        base_ability="supremeoverlord",
        item="leftovers",
        nature="adamant",
        evs=(252, 252, 0, 0, 4, 0),
        attack=369,
        defense=264,
        special_attack=140,
        special_defense=206,
        speed=122,
        status="none",
        rest_turns=0,
        sleep_turns=0,
        weight_kg=120.0,
        moves=kingambit_moves,
        terastallized=False,
        tera_type="ghost",
    )

    # Create Hatterene (bench, low HP)
    hatterene_moves = PokemonMoves(
        m0=Move(id="healingwish", pp=16, disabled=False),
        m1=Move(id="dazzlinggleam", pp=16, disabled=False),
        m2=Move(id="psychicnoise", pp=12, disabled=False),
        m3=Move(id="nuzzle", pp=32, disabled=False),
    )

    hatterene = Pokemon(
        id="hatterene",
        level=100,
        types=("psychic", "fairy"),
        base_types=("psychic", "fairy"),
        hp=25,  # Low HP
        maxhp=159,
        ability="magicbounce",
        base_ability="magicbounce",
        item="noitem",
        nature="quiet",
        evs=(252, 0, 0, 252, 4, 0),
        attack=176,
        defense=226,
        special_attack=387,
        special_defense=224,
        speed=72,
        status="none",
        rest_turns=0,
        sleep_turns=0,
        weight_kg=5.1,
        moves=hatterene_moves,
        terastallized=False,
        tera_type="water",
    )

    # Create Landorus-Therian (bench, damaged)
    landorus_moves = PokemonMoves(
        m0=Move(id="stealthrock", pp=32, disabled=False),
        m1=Move(id="earthquake", pp=13, disabled=False),
        m2=Move(id="stoneedge", pp=7, disabled=False),
        m3=Move(id="grassknot", pp=32, disabled=False),
    )

    landorus = Pokemon(
        id="landorustherian",
        level=100,
        types=("ground", "flying"),
        base_types=("ground", "flying"),
        hp=64,  # ~21% HP
        maxhp=302,
        ability="intimidate",
        base_ability="intimidate",
        item="rockyhelmet",
        nature="jolly",
        evs=(0, 252, 0, 0, 4, 252),
        attack=397,
        defense=216,
        special_attack=221,
        special_defense=196,
        speed=259,
        status="none",
        rest_turns=0,
        sleep_turns=0,
        weight_kg=68.0,
        moves=landorus_moves,
        terastallized=False,
        tera_type="water",
    )

    # Create fainted Pokemon for remaining slots
    fainted_moves = PokemonMoves(
        m0=Move(id="tackle", pp=0, disabled=False),
        m1=Move(id="tackle", pp=0, disabled=False),
        m2=Move(id="tackle", pp=0, disabled=False),
        m3=Move(id="tackle", pp=0, disabled=False),
    )

    fainted = []
    for _ in range(3):
        fainted.append(
            Pokemon(
                id="missingno",
                level=1,
                types=("typeless", "typeless"),
                base_types=("typeless", "typeless"),
                hp=0,
                maxhp=1,
                ability="none",
                base_ability="none",
                item="none",
                nature="serious",
                evs=(0, 0, 0, 0, 0, 0),
                attack=1,
                defense=1,
                special_attack=1,
                special_defense=1,
                speed=1,
                status="none",
                rest_turns=0,
                sleep_turns=0,
                weight_kg=0.1,
                moves=fainted_moves,
                terastallized=False,
                tera_type="typeless",
            )
        )

    # Create opponent's Landorus-Therian (active, full HP)
    opp_landorus_moves = PokemonMoves(
        m0=Move(id="earthquake", pp=15, disabled=False),
        m1=Move(id="stealthrock", pp=32, disabled=False),
        m2=Move(id="uturn", pp=32, disabled=False),
        m3=Move(id="taunt", pp=32, disabled=False),
    )

    opp_landorus = Pokemon(
        id="landorustherian",
        level=100,
        types=("ground", "flying"),
        base_types=("ground", "flying"),
        hp=302,  # Full HP
        maxhp=302,
        ability="intimidate",
        base_ability="intimidate",
        item="rockyhelmet",
        nature="jolly",
        evs=(0, 252, 0, 0, 4, 252),
        attack=397,
        defense=216,
        special_attack=221,
        special_defense=196,
        speed=259,
        status="none",
        rest_turns=0,
        sleep_turns=0,
        weight_kg=68.0,
        moves=opp_landorus_moves,
        terastallized=False,
        tera_type="steel",
    )

    # Create opponent's bench (hidden/unknown Pokemon)
    opponent_bench = []
    for species in ["zamazenta", "hatterene", "kyurem", "kingambit", "ironmoth"]:
        unknown_moves = PokemonMoves(
            m0=Move(id="tackle", pp=32, disabled=False),
            m1=Move(id="tackle", pp=32, disabled=False),
            m2=Move(id="tackle", pp=32, disabled=False),
            m3=Move(id="tackle", pp=32, disabled=False),
        )

        opponent_bench.append(
            Pokemon(
                id=species,
                level=100,
                types=("typeless", "typeless"),  # Unknown
                base_types=("typeless", "typeless"),
                hp=300,
                maxhp=300,
                ability="none",
                base_ability="none",
                item="none",
                nature="serious",
                evs=(0, 0, 0, 0, 0, 0),
                attack=200,
                defense=200,
                special_attack=200,
                special_defense=200,
                speed=200,
                status="none",
                rest_turns=0,
                sleep_turns=0,
                weight_kg=100.0,
                moves=unknown_moves,
                terastallized=False,
                tera_type="typeless",
            )
        )

    # Create sides
    side_one = Side(
        pokemon=[kingambit, hatterene, landorus] + fainted,
        active_index=0,
        force_switch=False,
        force_trapped=False,
        attack_boost=1,  # Kingambit has +1 attack from Swords Dance
        defense_boost=0,
        special_attack_boost=0,
        special_defense_boost=0,
        speed_boost=0,
        accuracy_boost=0,
        evasion_boost=0,
        last_used_move="move:3",  # Swords Dance
    )

    side_two = Side(
        pokemon=[opp_landorus] + opponent_bench,
        active_index=0,
        force_switch=False,
        force_trapped=False,
        attack_boost=0,
        defense_boost=0,
        special_attack_boost=0,
        special_defense_boost=0,
        speed_boost=0,
        accuracy_boost=0,
        evasion_boost=0,
        last_used_move="move:0",  # Earthquake
    )

    # Create state
    state = State(
        side_one=side_one,
        side_two=side_two,
        weather="none",
        weather_turns_remaining=0,
        terrain="none",
        terrain_turns_remaining=0,
        trick_room=False,
        trick_room_turns_remaining=0,
        team_preview=False,
    )

    return state


def create_forced_switch_state():
    """Create a state where the player is forced to switch."""
    state = create_simple_battle_state()
    # Modify to force switch
    state.side_one.force_switch = True
    state.side_one.pokemon[0].hp = 0  # Kingambit fainted
    return state


def main():
    print("Creating example poke-engine State objects...")
    print("=" * 80)

    # Create and save simple battle state
    print("\n1. Creating 'example_state' - normal mid-battle situation")
    state1 = create_simple_battle_state()
    with open("example_state", "wb") as f:
        pickle.dump(state1, f)
    print("   ✓ Saved to 'example_state'")
    print(f"   - Player: Kingambit (HP: {state1.side_one.pokemon[0].hp}/{state1.side_one.pokemon[0].maxhp}, +{state1.side_one.attack_boost} Atk)")
    print(f"   - Opponent: Landorus-T (HP: {state1.side_two.pokemon[0].hp}/{state1.side_two.pokemon[0].maxhp})")
    print(f"   - Available switches: 2 (Hatterene, Landorus-T)")

    # Create and save forced switch state
    print("\n2. Creating 'example_forced_switch' - forced switch situation")
    state2 = create_forced_switch_state()
    with open("example_forced_switch", "wb") as f:
        pickle.dump(state2, f)
    print("   ✓ Saved to 'example_forced_switch'")
    print(f"   - Player: Kingambit fainted, must switch")
    print(f"   - Available switches: 2 (Hatterene, Landorus-T)")

    print("\n" + "=" * 80)
    print("✓ Example states created successfully!")
    print("\nYou can now test with:")
    print("  python compare_observations.py example_state")
    print("  python compare_observations.py example_forced_switch")


if __name__ == "__main__":
    main()
