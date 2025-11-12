#!/usr/bin/env python3
"""
Convert a JSON UniversalState to a pickled poke-engine State.

This is useful because weird_state files are JSON UniversalStates,
but compare_observations.py needs poke-engine States.
"""

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "vendor" / "poke-engine" / "poke-engine-py" / "python"))
sys.path.insert(0, str(Path(__file__).parent / "vendor" / "metamon"))

from poke_engine import State, Side, Pokemon, Move


def universal_to_pe_state(universal_dict):
    """Convert a UniversalState dict to a poke-engine State."""

    # Helper to create Move from UniversalMove dict
    def make_move(move_dict):
        return Move(
            id=move_dict['name'],
            pp=move_dict['current_pp'],
            disabled=False
        )

    # Helper to create Pokemon from UniversalPokemon dict
    def make_pokemon(pkmn_dict, is_active=False):
        moves_list = [make_move(m) for m in pkmn_dict['moves'][:4]]

        # Pad to 4 moves if needed
        while len(moves_list) < 4:
            moves_list.append(Move(id="tackle", pp=0, disabled=False))

        # Map types string to tuple
        types_str = pkmn_dict['types']
        if ' ' in types_str:
            type1, type2 = types_str.split(' ', 1)
        else:
            type1 = types_str
            type2 = "typeless"

        # Calculate actual HP from percentage
        maxhp = 300  # Default reasonable max HP
        hp = int(pkmn_dict['hp_pct'] * maxhp)

        return Pokemon(
            id=pkmn_dict['name'],
            level=pkmn_dict['lvl'],
            types=(type1, type2),
            base_types=(type1, type2),
            hp=max(0, hp),
            maxhp=maxhp,
            ability=pkmn_dict['ability'] if pkmn_dict['ability'] != 'noability' else 'none',
            base_ability=pkmn_dict['ability'] if pkmn_dict['ability'] != 'noability' else 'none',
            item=pkmn_dict['item'] if pkmn_dict['item'] != 'noitem' else 'none',
            nature="serious",
            evs=(85, 85, 85, 85, 85, 85),
            attack=pkmn_dict['base_atk'] * 2,
            defense=pkmn_dict['base_def'] * 2,
            special_attack=pkmn_dict['base_spa'] * 2,
            special_defense=pkmn_dict['base_spd'] * 2,
            speed=pkmn_dict['base_spe'] * 2,
            status=pkmn_dict['status'] if pkmn_dict['status'] != 'nostatus' else 'none',
            rest_turns=0,
            sleep_turns=0,
            weight_kg=100.0,
            moves=moves_list,
            terastallized=pkmn_dict.get('tera_type', 'notype') != 'notype',
            tera_type=pkmn_dict.get('tera_type', 'typeless'),
        )

    # Create player's team
    player_active = make_pokemon(universal_dict['player_active_pokemon'], is_active=True)
    player_team = [player_active]

    for switch in universal_dict['available_switches']:
        player_team.append(make_pokemon(switch))

    # Pad to 6 Pokemon
    while len(player_team) < 6:
        player_team.append(Pokemon.create_fainted())

    # Create opponent's team (we only know the active Pokemon for sure)
    opponent_active = make_pokemon(universal_dict['opponent_active_pokemon'], is_active=True)
    opponent_team = [opponent_active]

    # Fill rest with placeholders based on teampreview if available
    teampreview = universal_dict.get('opponent_teampreview', [])
    for species_name in teampreview[1:6]:  # Skip first (it's the active one)
        if species_name and species_name != '<blank>':
            # Create placeholder
            placeholder = Pokemon(
                id=species_name,
                level=100,
                types=("typeless", "typeless"),
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
                moves=[Move(id="tackle", pp=32, disabled=False) for _ in range(4)],
                terastallized=False,
                tera_type="typeless",
            )
            opponent_team.append(placeholder)

    # Pad to 6
    while len(opponent_team) < 6:
        opponent_team.append(Pokemon.create_fainted())

    # Create sides
    player_dict = universal_dict['player_active_pokemon']
    side_one = Side(
        pokemon=player_team,
        active_index="0",
        force_switch=universal_dict['forced_switch'],
        force_trapped=False,
        attack_boost=player_dict['atk_boost'],
        defense_boost=player_dict['def_boost'],
        special_attack_boost=player_dict['spa_boost'],
        special_defense_boost=player_dict['spd_boost'],
        speed_boost=player_dict['spe_boost'],
        accuracy_boost=player_dict['accuracy_boost'],
        evasion_boost=player_dict['evasion_boost'],
        last_used_move=f"move:0",  # Default
    )

    opponent_dict = universal_dict['opponent_active_pokemon']
    side_two = Side(
        pokemon=opponent_team,
        active_index="0",
        force_switch=False,
        force_trapped=False,
        attack_boost=opponent_dict['atk_boost'],
        defense_boost=opponent_dict['def_boost'],
        special_attack_boost=opponent_dict['spa_boost'],
        special_defense_boost=opponent_dict['spd_boost'],
        speed_boost=opponent_dict['spe_boost'],
        accuracy_boost=opponent_dict['accuracy_boost'],
        evasion_boost=opponent_dict['evasion_boost'],
        last_used_move=f"move:0",  # Default
    )

    # Create state
    weather = universal_dict.get('weather', 'noweather')
    if weather == 'noweather':
        weather = 'none'

    terrain = universal_dict.get('battle_field', 'nofield')
    if terrain == 'nofield':
        terrain = 'none'

    state = State(
        side_one=side_one,
        side_two=side_two,
        weather=weather,
        weather_turns_remaining=0,
        terrain=terrain,
        terrain_turns_remaining=0,
        trick_room=False,
        trick_room_turns_remaining=0,
        team_preview=False,
    )

    return state


def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_universal_to_pe_state.py <json_universal_state> [output_name]")
        print("\nExample:")
        print("  python convert_universal_to_pe_state.py weird_state")
        print("  python convert_universal_to_pe_state.py weird_state weird_state_pe")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else f"{input_file}_pe"

    print(f"Converting '{input_file}' to poke-engine State...")

    # Load JSON (handle Python-style trailing commas)
    try:
        with open(input_file, 'r') as f:
            content = f.read()
            # Try to eval as Python dict first (handles trailing commas)
            try:
                universal_dict = eval(content)
            except:
                # Fall back to JSON
                universal_dict = json.loads(content)
        print(f"✓ Loaded UniversalState JSON")
    except Exception as e:
        print(f"✗ Failed to load: {e}")
        sys.exit(1)

    # Convert
    try:
        state = universal_to_pe_state(universal_dict)
        print(f"✓ Converted to poke-engine State")
        print(f"  - Player active: {state.side_one.pokemon[0].id} (HP: {state.side_one.pokemon[0].hp}/{state.side_one.pokemon[0].maxhp})")
        print(f"  - Opponent active: {state.side_two.pokemon[0].id} (HP: {state.side_two.pokemon[0].hp}/{state.side_two.pokemon[0].maxhp})")
        print(f"  - Player switches: {len([p for p in state.side_one.pokemon[1:] if p.hp > 0])}")
    except Exception as e:
        print(f"✗ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Save
    try:
        with open(output_file, 'wb') as f:
            pickle.dump(state, f)
        print(f"✓ Saved to '{output_file}'")
        print(f"\nYou can now test with:")
        print(f"  python compare_observations.py {output_file}")
    except Exception as e:
        print(f"✗ Failed to save: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
