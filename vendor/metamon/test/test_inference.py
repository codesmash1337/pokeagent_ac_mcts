"""
Test getting policy and value estimates from Abra model.

This file demonstrates how to:
1. Create a test observation state
2. Get policy (action probabilities) and value (Q-values) in a single forward pass
3. Use these for planning algorithms like MCTS

IMPORTANT: Action Index Mapping
================================
The model's action space has 13 actions total:
- Actions 0-3: Moves (in ALPHABETICAL order, not original order!)
- Actions 4-8: Switches (5 available switch slots)
- Actions 9-12: Tera moves (in ALPHABETICAL order, not original order!)

When state_to_obs() processes moves, it sorts them alphabetically using consistent_move_order().
This means action indices map to alphabetically sorted moves, NOT the original move order!

Example:
  Original moves: ["thundershock", "nuzzle", "nightslash", "moonblast"]
  Alphabetical:   ["moonblast", "nightslash", "nuzzle", "thundershock"]
  Action 0 -> "moonblast"
  Action 1 -> "nightslash"
  Action 2 -> "nuzzle"
  Action 3 -> "thundershock"

Run with: python vendor/metamon/test/test_inference.py
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
from poke_engine import Move as PEMove
from poke_engine import Pokemon as PEPokemon
from poke_engine import Side as PESide
from poke_engine import State as PEState
from poke_engine import PokemonIndex

from metamon.rl.pretrained import get_pretrained_model
from metamon.interface import (
    UniversalState,
    consistent_move_order,
    get_observation_space,
)
from metamon.backend.showdown_dex import Dex
from metamon.backend.replay_parser.str_parsing import clean_name, pokemon_name
from metamon.poke_engine_adapter import (
    poke_engine_state_to_observation,
    poke_engine_state_to_universal_state,
)
from ac_inference import (
    get_policy_and_value,
    PolicyValueInference,
    prepare_observation,
    init_inference_inputs,
    sample_action,
    get_best_action,
)


TEST_CASES_FILE = Path(__file__).with_name("test_cases.md")


def load_test_cases() -> List[Dict[str, object]]:
    """Load structured test cases from the markdown file."""

    text = TEST_CASES_FILE.read_text(encoding="utf-8")
    marker = "```json"
    start = text.find(marker)
    if start == -1:
        raise ValueError("Could not locate JSON block in test_cases.md")

    start = text.find("\n", start)
    if start == -1:
        raise ValueError("Malformed JSON fence in test_cases.md")
    start += 1

    end = text.find("```", start)
    if end == -1:
        raise ValueError("Unterminated JSON block in test_cases.md")

    json_blob = text[start:end].strip()
    if not json_blob:
        raise ValueError("Empty JSON block in test_cases.md")

    return json.loads(json_blob)


def create_universal_move_from_dex(dex, move_id, current_pp=None):
    """Create a UniversalMove from Dex data."""
    from metamon.interface import UniversalMove

    move_key = clean_name(move_id)
    if move_key not in dex.moves:
        raise KeyError(f"Move '{move_id}' not found in Dex")
    move_data = dex.moves[move_key]
    max_pp = move_data["pp"] * 1.6
    if current_pp is None:
        current_pp = max_pp

    return UniversalMove(
        name=clean_name(move_data["name"]),
        move_type=clean_name(move_data["type"]),
        category=clean_name(move_data["category"]),
        base_power=move_data.get("basePower", 0),
        accuracy=move_data.get("accuracy", 100) / 100.0
        if move_data.get("accuracy") != True
        else 1.0,
        priority=move_data.get("priority", 0),
        current_pp=current_pp,
        max_pp=max_pp,
    )


def create_universal_pokemon_from_dex(
    dex,
    species,
    move_ids,
    move_pps,
    ability,
    tera_type,
    hp_pct,
    item="unknownitem",
    status="nostatus",
    terastallized=False,
    stat_boosts=None,
):
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


def create_fainted_switches(dex, count: int = 5) -> List:
    """Create a list of fainted placeholder Pokemon for bench slots."""

    fainted = []
    for _ in range(count):
        fainted.append(
            create_universal_pokemon_from_dex(
                dex=dex,
                species="Blissey",
                move_ids=["softboiled", "seismictoss", "thunderwave", "teleport"],
                move_pps=[0, 0, 0, 0],
                ability="Natural Cure",
                tera_type="Normal",
                hp_pct=0.0,
                item="leftovers",
                status="fnt",
            )
        )
    return fainted


@dataclass
class PokemonSpec:
    species: str
    move_ids: List[str]
    move_pps: List[int]
    ability: str
    tera_type: str
    hp_pct: float
    item: str = "unknownitem"
    status: str = "nostatus"
    terastallized: bool = False


def _status_to_poke_engine(status: str) -> str:
    mapping = {
        "nostatus": "none",
        "brn": "burn",
        "slp": "sleep",
        "frz": "freeze",
        "par": "paralyze",
        "psn": "poison",
        "tox": "toxic",
        "fnt": "none",
    }
    status = status.lower()
    return mapping.get(status, status)


def _tera_type_to_poke_engine(tera_type: str) -> str:
    tera_type = clean_name(tera_type)
    if tera_type in {"notype", "typeless"}:
        return "typeless"
    return tera_type


def _types_from_dex(
    dex: Dex, species: str, terastallized: bool, tera_type: str
) -> Tuple[str, str]:
    entry = dex.get_pokedex_entry(species)
    base_types = [clean_name(t) for t in entry["types"]]
    while len(base_types) < 2:
        base_types.append("typeless")
    if terastallized:
        return (_tera_type_to_poke_engine(tera_type), "typeless")
    return tuple(base_types[:2])  # type: ignore[return-value]


def create_poke_engine_pokemon_from_spec(dex: Dex, spec: PokemonSpec) -> PEPokemon:
    entry = dex.get_pokedex_entry(spec.species)
    if len(spec.move_ids) != 4 or len(spec.move_pps) != 4:
        raise ValueError(
            f"PokemonSpec for '{spec.species}' must provide exactly four moves"
        )
    moves = [
        PEMove(id=clean_name(m_id), pp=pp, disabled=False)
        for m_id, pp in zip(spec.move_ids, spec.move_pps)
    ]

    atk = entry["baseStats"]["atk"]
    df = entry["baseStats"]["def"]
    spa = entry["baseStats"]["spa"]
    spd = entry["baseStats"]["spd"]
    spe = entry["baseStats"]["spe"]

    max_hp = 400
    hp = int(round(spec.hp_pct * max_hp))

    types = _types_from_dex(dex, spec.species, spec.terastallized, spec.tera_type)
    base_types = _types_from_dex(dex, spec.species, False, spec.tera_type)

    return PEPokemon(
        id=clean_name(spec.species),
        level=100,
        types=types,
        base_types=base_types,
        hp=hp,
        maxhp=max_hp,
        ability=clean_name(spec.ability),
        base_ability=clean_name(spec.ability),
        item=clean_name(spec.item),
        nature="serious",
        evs=(0, 0, 0, 0, 0, 0),
        attack=atk,
        defense=df,
        special_attack=spa,
        special_defense=spd,
        speed=spe,
        status=_status_to_poke_engine(spec.status),
        rest_turns=0,
        sleep_turns=0,
        weight_kg=float(entry.get("weightkg", 50)),
        moves=moves,
        terastallized=spec.terastallized,
        tera_type=_tera_type_to_poke_engine(spec.tera_type),
    )


def build_poke_engine_side(
    dex: Dex,
    specs: List[PokemonSpec],
    *,
    last_move_index: int,
) -> PESide:
    if len(specs) != 6:
        raise ValueError(
            "Exactly 6 PokemonSpec entries are required to build a poke-engine side"
        )
    pokemon = [create_poke_engine_pokemon_from_spec(dex, spec) for spec in specs]
    return PESide(
        pokemon=pokemon,
        active_index=PokemonIndex.P0,
        last_used_move=f"move:{last_move_index}",
    )


def build_poke_engine_state_from_specs(
    dex: Dex,
    player_specs: List[PokemonSpec],
    opponent_specs: List[PokemonSpec],
    *,
    player_prev_index: int,
    opponent_prev_index: int,
) -> PEState:
    side_one = build_poke_engine_side(
        dex, player_specs, last_move_index=player_prev_index
    )
    side_two = build_poke_engine_side(
        dex, opponent_specs, last_move_index=opponent_prev_index
    )
    return PEState(
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


def build_universal_state_from_specs(
    dex: Dex,
    player_specs: List[PokemonSpec],
    opponent_specs: List[PokemonSpec],
    *,
    player_prev_index: int,
    opponent_prev_index: int,
) -> UniversalState:
    if len(player_specs) != 6 or len(opponent_specs) != 6:
        raise ValueError(
            "Universal state builder expects exactly 6 PokemonSpec entries per side"
        )

    player_pokemon = [
        create_universal_pokemon_from_dex(
            dex,
            spec.species,
            spec.move_ids,
            spec.move_pps,
            spec.ability,
            spec.tera_type,
            spec.hp_pct,
            item=spec.item,
            status=spec.status,
            terastallized=spec.terastallized,
        )
        for spec in player_specs
    ]
    opponent_pokemon = [
        create_universal_pokemon_from_dex(
            dex,
            spec.species,
            spec.move_ids,
            spec.move_pps,
            spec.ability,
            spec.tera_type,
            spec.hp_pct,
            item=spec.item,
            status=spec.status,
            terastallized=spec.terastallized,
        )
        for spec in opponent_specs
    ]

    player_active = player_pokemon[0]
    opponent_active = opponent_pokemon[0]

    available_switches = [
        poke
        for poke, spec in zip(player_pokemon[1:], player_specs[1:])
        if spec.hp_pct > 0.0
    ]

    player_prev_move = player_active.moves[player_prev_index]
    opponent_prev_move = opponent_active.moves[opponent_prev_index]

    opponents_remaining = sum(1 for spec in opponent_specs if spec.hp_pct > 0.0)

    opponent_teampreview = [pokemon_name(spec.species) for spec in opponent_specs]
    while len(opponent_teampreview) < 6:
        opponent_teampreview.append("<blank>")

    return UniversalState(
        format="gen9ou",
        player_active_pokemon=player_active,
        opponent_active_pokemon=opponent_active,
        available_switches=available_switches,
        player_prev_move=player_prev_move,
        opponent_prev_move=opponent_prev_move,
        opponents_remaining=opponents_remaining,
        player_conditions="noconditions",
        opponent_conditions="noconditions",
        weather="noweather",
        battle_field="nofield",
        forced_switch=False,
        battle_won=False,
        battle_lost=False,
        can_tera=not any(spec.terastallized for spec in player_specs),
        opponent_teampreview=opponent_teampreview,
    )


def get_legal_actions(state: UniversalState) -> List[int]:
    """Return legal action indices for the inference helpers."""

    actions: List[int] = []

    # Base moves (indices 0-3). We include them regardless of PP for reporting.
    for move_idx in range(4):
        actions.append(move_idx)

    # Switches (indices 4-8). Mask out fainted or empty slots.
    max_switch_slots = 5
    for slot_idx in range(max_switch_slots):
        if slot_idx >= len(state.available_switches):
            continue

        candidate = state.available_switches[slot_idx]
        hp_pct = getattr(candidate, "hp_pct", 0.0)
        status = getattr(candidate, "status", "")

        if hp_pct is None:
            hp_pct = 0.0

        is_fainted = status == "fnt" or hp_pct <= 0.0
        if not is_fainted:
            actions.append(4 + slot_idx)

    if state.can_tera:
        num_moves = len(state.player_active_pokemon.moves)
        actions.extend(range(9, 9 + num_moves))

    return sorted(set(actions))


def get_action_name(action_idx: int, state: UniversalState) -> str:
    """
    Get the human-readable name for an action index.

    Action indices map to:
    - 0-3: Moves (in alphabetical order)
    - 4-8: Switches (5 switches)
    - 9-12: Tera moves (in alphabetical order)

    Args:
        action_idx: The action index (0-12)
        state: The UniversalState containing move information

    Returns:
        Human-readable action name
    """
    sorted_moves = consistent_move_order(state.player_active_pokemon.moves)

    if action_idx < 4:
        # Regular move
        if action_idx < len(sorted_moves):
            return sorted_moves[action_idx].name
        else:
            return f"move-{action_idx}"
    elif action_idx < 9:
        # Switch
        switch_idx = action_idx - 4
        return f"switch-{switch_idx}"
    else:
        # Tera move
        tera_move_idx = action_idx - 9
        if tera_move_idx < len(sorted_moves):
            return f"{sorted_moves[tera_move_idx].name}-tera"
        else:
            return f"tera+{tera_move_idx}"


def get_action_index_by_name(action_name: str, state: UniversalState) -> int:
    """Map an action name back to its index for the current state."""

    for idx in range(13):
        if get_action_name(idx, state) == action_name:
            return idx
    raise ValueError(f"Action '{action_name}' not available in current state")


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
        move_ids=["thundershock", "nuzzle", "nightslash", "moonblast"],
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
        hp_pct=1.0,
        item="leftovers",
        status="nostatus",
        stat_boosts={
            "atk_boost": 0,
            "spa_boost": 0,
            "def_boost": 6,  # +2 Defense from Iron Defense
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
    player_prev_move = create_universal_move_from_dex(dex, "swordsdance")
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
        opponent_teampreview=[
            "zamazenta",
            "toxapex",
            "toxapex",
            "toxapex",
            "toxapex",
            "toxapex",
        ],
    )

    return state


def create_case_1_state() -> UniversalState:
    """Magmortar vs Scizor - Flamethrower should be favored."""

    dex = Dex.from_gen(9)

    magmortar = create_universal_pokemon_from_dex(
        dex=dex,
        species="Magmortar",
        move_ids=["flamethrower", "thunderbolt", "focusblast", "solarbeam"],
        move_pps=[15, 15, 8, 10],
        ability="Flame Body",
        tera_type="Fire",
        hp_pct=1.0,
        item="choicespecs",
        status="nostatus",
    )

    scizor = create_universal_pokemon_from_dex(
        dex=dex,
        species="Scizor",
        move_ids=["bulletpunch", "uturn", "knockoff", "swordsdance"],
        move_pps=[30, 20, 24, 20],
        ability="Technician",
        tera_type="Steel",
        hp_pct=1.0,
        item="leftovers",
        status="nostatus",
    )

    fainted_pokemon = create_fainted_switches(dex)

    player_prev_move = create_universal_move_from_dex(dex, "flamethrower")
    opponent_prev_move = create_universal_move_from_dex(dex, "uturn")

    return UniversalState(
        format="gen9ou",
        player_active_pokemon=magmortar,
        opponent_active_pokemon=scizor,
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
        can_tera=False,
        opponent_teampreview=[
            "scizor",
            "blissey",
            "blissey",
            "blissey",
            "blissey",
            "blissey",
        ],
    )


def create_case_2_state() -> UniversalState:
    """Tera Water Torterra should choose Tera Earthquake vs Magmortar."""

    dex = Dex.from_gen(9)

    torterra = create_universal_pokemon_from_dex(
        dex=dex,
        species="Torterra",
        move_ids=["earthquake", "woodhammer", "stoneedge", "synthesis"],
        move_pps=[16, 16, 8, 16],
        ability="Overgrow",
        tera_type="Water",
        hp_pct=0.8,
        item="leftovers",
        status="nostatus",
        terastallized=False,
    )

    magmortar = create_universal_pokemon_from_dex(
        dex=dex,
        species="Magmortar",
        move_ids=["fireblast", "thunderbolt", "focusblast", "hiddenpowerice"],
        move_pps=[8, 15, 8, 24],
        ability="Flame Body",
        tera_type="Fire",
        hp_pct=1.0,
        item="lifeorb",
        status="nostatus",
    )

    fainted_pokemon = create_fainted_switches(dex)

    player_prev_move = create_universal_move_from_dex(dex, "synthesis")
    opponent_prev_move = create_universal_move_from_dex(dex, "fireblast")

    return UniversalState(
        format="gen9ou",
        player_active_pokemon=torterra,
        opponent_active_pokemon=magmortar,
        available_switches=fainted_pokemon,
        player_prev_move=player_prev_move,
        opponent_prev_move=opponent_prev_move,
        opponents_remaining=2,
        player_conditions="noconditions",
        opponent_conditions="stealthrock",
        weather="noweather",
        battle_field="grassyterrain",
        forced_switch=False,
        battle_won=False,
        battle_lost=False,
        can_tera=True,
        opponent_teampreview=[
            "magmortar",
            "blissey",
            "blissey",
            "blissey",
            "blissey",
            "blissey",
        ],
    )


def create_case_3_state() -> UniversalState:
    """Tera Ghost Tyranitar should mix Rock Slide options vs Weavile."""

    dex = Dex.from_gen(9)

    tyranitar = create_universal_pokemon_from_dex(
        dex=dex,
        species="Tyranitar",
        move_ids=["rockslide", "crunch", "earthquake", "stoneedge"],
        move_pps=[16, 24, 16, 8],
        ability="Sand Stream",
        tera_type="Ghost",
        hp_pct=0.75,
        item="assaultvest",
        status="nostatus",
        terastallized=False,
    )

    weavile = create_universal_pokemon_from_dex(
        dex=dex,
        species="Weavile",
        move_ids=["knockoff", "lowkick", "iciclecrash", "iceshard"],
        move_pps=[24, 32, 16, 30],
        ability="Pressure",
        tera_type="Fighting",
        hp_pct=1.0,
        item="lifeorb",
        status="nostatus",
    )

    fainted_pokemon = create_fainted_switches(dex)

    player_prev_move = create_universal_move_from_dex(dex, "rockslide")
    opponent_prev_move = create_universal_move_from_dex(dex, "knockoff")

    return UniversalState(
        format="gen9ou",
        player_active_pokemon=tyranitar,
        opponent_active_pokemon=weavile,
        available_switches=fainted_pokemon,
        player_prev_move=player_prev_move,
        opponent_prev_move=opponent_prev_move,
        opponents_remaining=2,
        player_conditions="sandstorm",
        opponent_conditions="spikes",
        weather="sandstorm",
        battle_field="nofield",
        forced_switch=False,
        battle_won=False,
        battle_lost=False,
        can_tera=True,
        opponent_teampreview=[
            "weavile",
            "blissey",
            "blissey",
            "blissey",
            "blissey",
            "blissey",
        ],
    )


def create_case_4_state() -> UniversalState:
    """Charizard should Terastallize into Fighting and use Low Kick vs Tyranitar."""

    dex = Dex.from_gen(9)

    charizard = create_universal_pokemon_from_dex(
        dex=dex,
        species="Charizard",
        move_ids=["lowkick", "flareblitz", "dragonclaw", "roost"],
        move_pps=[32, 24, 24, 16],
        ability="Blaze",
        tera_type="Fighting",
        hp_pct=0.85,
        item="lifeorb",
        status="nostatus",
        terastallized=False,
    )

    tyranitar = create_universal_pokemon_from_dex(
        dex=dex,
        species="Tyranitar",
        move_ids=["stoneedge", "crunch", "earthquake", "dragondance"],
        move_pps=[8, 24, 16, 20],
        ability="Sand Stream",
        tera_type="Rock",
        hp_pct=1.0,
        item="choicescarf",
        status="nostatus",
    )

    bench = create_fainted_switches(dex)

    return UniversalState(
        format="gen9ou",
        player_active_pokemon=charizard,
        opponent_active_pokemon=tyranitar,
        available_switches=bench,
        player_prev_move=create_universal_move_from_dex(dex, "roost"),
        opponent_prev_move=create_universal_move_from_dex(dex, "stoneedge"),
        opponents_remaining=2,
        player_conditions="sunnyday",
        opponent_conditions="sandstorm",
        weather="sandstorm",
        battle_field="nofield",
        forced_switch=False,
        battle_won=False,
        battle_lost=False,
        can_tera=True,
        opponent_teampreview=[
            "tyranitar",
            "garchomp",
            "dragonite",
            "excadrill",
            "hydreigon",
            "amoonguss",
        ],
    )


def create_case_5_state() -> UniversalState:
    """Darkrai should pivot out into Pecharunt against Zamazenta."""

    dex = Dex.from_gen(9)

    darkrai = create_universal_pokemon_from_dex(
        dex=dex,
        species="Darkrai",
        move_ids=["darkpulse", "sludgebomb", "focusblast", "nastyplot"],
        move_pps=[16, 16, 8, 20],
        ability="Bad Dreams",
        tera_type="Ghost",
        hp_pct=0.6,
        item="choicescarf",
        status="nostatus",
        terastallized=False,
    )

    zamazenta_opp = create_universal_pokemon_from_dex(
        dex=dex,
        species="Zamazenta",
        move_ids=["closecombat", "crunch", "bodypress", "irondefense"],
        move_pps=[8, 16, 16, 24],
        ability="Dauntless Shield",
        tera_type="Fighting",
        hp_pct=0.9,
        item="leftovers",
        status="nostatus",
    )

    pecharunt = create_universal_pokemon_from_dex(
        dex=dex,
        species="Pecharunt",
        move_ids=["malignantchain", "recover", "partingshot", "sludgebomb"],
        move_pps=[8, 16, 32, 16],
        ability="Poison Puppeteer",
        tera_type="Poison",
        hp_pct=0.95,
        item="leftovers",
        status="nostatus",
    )

    kingambit = create_universal_pokemon_from_dex(
        dex=dex,
        species="Kingambit",
        move_ids=["kowtowcleave", "suckerpunch", "ironhead", "swordsdance"],
        move_pps=[16, 8, 15, 20],
        ability="Supreme Overlord",
        tera_type="Dark",
        hp_pct=0.7,
        item="leftovers",
        status="nostatus",
    )

    bench = [kingambit, pecharunt] + create_fainted_switches(dex, count=3)

    return UniversalState(
        format="gen9ou",
        player_active_pokemon=darkrai,
        opponent_active_pokemon=zamazenta_opp,
        available_switches=bench,
        player_prev_move=create_universal_move_from_dex(dex, "darkvoid"),
        opponent_prev_move=create_universal_move_from_dex(dex, "closecombat"),
        opponents_remaining=2,
        player_conditions="noconditions",
        opponent_conditions="stealthrock",
        weather="noweather",
        battle_field="nofield",
        forced_switch=False,
        battle_won=False,
        battle_lost=False,
        can_tera=False,
        opponent_teampreview=[
            "zamazenta",
            "pecharunt",
            "garganacl",
            "tinglu",
            "gliscor",
            "rotomwash",
        ],
    )


def create_case_6_state() -> UniversalState:
    """Hatterene should use Healing Wish to revive low HP Zamazenta vs Kingambit."""

    dex = Dex.from_gen(9)

    hatterene = create_universal_pokemon_from_dex(
        dex=dex,
        species="Hatterene",
        move_ids=["healingwish", "psychic", "mysticalfire", "dazzlinggleam"],
        move_pps=[10, 16, 16, 16],
        ability="Magic Bounce",
        tera_type="Fairy",
        hp_pct=0.1,
        item="custapberry",
        status="nostatus",
        terastallized=False,
    )

    zamazenta_partner = create_universal_pokemon_from_dex(
        dex=dex,
        species="Zamazenta",
        move_ids=["closecombat", "crunch", "bodypress", "irondefense"],
        move_pps=[8, 16, 16, 24],
        ability="Dauntless Shield",
        tera_type="Fighting",
        hp_pct=0.05,
        item="leftovers",
        status="burn",
    )

    kingambit_opp = create_universal_pokemon_from_dex(
        dex=dex,
        species="Kingambit",
        move_ids=["kowtowcleave", "suckerpunch", "ironhead", "swordsdance"],
        move_pps=[16, 8, 15, 20],
        ability="Supreme Overlord",
        tera_type="Dark",
        hp_pct=0.9,
        item="blackglasses",
        status="nostatus",
    )

    bench = [zamazenta_partner] + create_fainted_switches(dex, count=4)

    return UniversalState(
        format="gen9ou",
        player_active_pokemon=hatterene,
        opponent_active_pokemon=kingambit_opp,
        available_switches=bench,
        player_prev_move=create_universal_move_from_dex(dex, "mysticalfire"),
        opponent_prev_move=create_universal_move_from_dex(dex, "suckerpunch"),
        opponents_remaining=1,
        player_conditions="screens",
        opponent_conditions="stealthrock",
        weather="noweather",
        battle_field="psychicterrain",
        forced_switch=False,
        battle_won=False,
        battle_lost=False,
        can_tera=False,
        opponent_teampreview=[
            "kingambit",
            "dragonite",
            "tinglu",
            "rotomwash",
            "garchomp",
            "corviknight",
        ],
    )


STATE_FACTORY_REGISTRY: Dict[str, Callable[[], UniversalState]] = {
    "create_case_1_state": create_case_1_state,
    "create_case_2_state": create_case_2_state,
    "create_case_3_state": create_case_3_state,
    "create_case_4_state": create_case_4_state,
    "create_case_5_state": create_case_5_state,
    "create_case_6_state": create_case_6_state,
}


def test_basic_inference():
    """Test basic policy and value inference."""
    print("=" * 70)
    print("TEST 1: BASIC POLICY AND VALUE INFERENCE")
    print("=" * 70)

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
    print(
        f"   Our Pokemon: {state.player_active_pokemon.name} ({state.player_active_pokemon.hp_pct * 100:.0f}% HP)"
    )
    print(
        f"   Opponent: {state.opponent_active_pokemon.name} ({state.opponent_active_pokemon.hp_pct * 100:.0f}% HP)"
    )
    print(f"   Our moves: {[m.name for m in state.player_active_pokemon.moves]}")

    # Convert state to observation
    print("\n3. Converting state to observation...")
    print(
        f"   Original move order: {[m.name for m in state.player_active_pokemon.moves]}"
    )
    sorted_moves = consistent_move_order(state.player_active_pokemon.moves)
    print(f"   Alphabetical order: {[m.name for m in sorted_moves]}")
    print(f"   Action 0 -> {sorted_moves[0].name}")
    print(f"   Action 1 -> {sorted_moves[1].name}")
    print(f"   Action 2 -> {sorted_moves[2].name}")
    print(f"   Action 3 -> {sorted_moves[3].name}")

    obs = abra.observation_space.state_to_obs(
        state
    )  # Use full observation space (includes tokenization)
    legal_actions = get_legal_actions(state)

    obs_torch = prepare_observation(
        obs, legal_actions, abra.action_space.gym_space.n, device
    )
    print(f"\n   Observation keys: {list(obs_torch.keys())}")

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

    print("\n   ALL 13 Actions (sorted by probability):")
    print(f"   {'Rank':<6} {'Action':<8} {'Prob':<10} {'Q-Value':<10} {'Move'}")
    print(f"   {'-' * 70}")
    for i in range(13):  # Always show all 13 actions
        action_idx = sorted_actions[i].item()
        prob = sorted_probs[i].item()
        q_val = q_values[action_idx].item()
        move_name = get_action_name(action_idx, state)
        print(f"   {i + 1:<6} {action_idx:<8} {prob:<10.4f} {q_val:<10.4f} {move_name}")

    # Sample and greedy actions
    print("\n6. Action selection:")
    sampled = sample_action(action_probs, temperature=1.0, legal_actions=legal_actions)
    greedy = get_best_action(action_probs, legal_actions=legal_actions)

    print(f"   Sampled action: {sampled} ({get_action_name(sampled, state)})")
    print(f"   Greedy action: {greedy} ({get_action_name(greedy, state)})")

    # Assertions for test scenario
    print("\n7. Test Assertions:")
    best_action = torch.argmax(action_probs).item()
    print(f"   Best action: {best_action}")
    print(f"      Best move: {get_action_name(best_action, state)}")
    print(f"   State value: {state_value.item():.4f}")

    # Basic sanity checks (informational only)
    prob_sum = action_probs.sum().item()
    if prob_sum > 0.99:
        print("   ✓ Action probabilities sum to ~1")
    else:
        print(f"   ✗ Probabilities sum to {prob_sum:.4f} (expected ~1.0)")

    if len(q_values) == 13:
        print("   ✓ Received Q-values for all 13 actions")
    else:
        print(f"   ✗ Expected 13 Q-values, got {len(q_values)}")


def test_stateful_inference():
    """Test stateful inference wrapper."""
    print("\n" + "=" * 70)
    print("TEST 2: STATEFUL INFERENCE WITH PolicyValueInference")
    print("=" * 70)

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
    legal_actions = get_legal_actions(state)

    # Get policy and value
    print("\n3. Getting policy and value (step 0)...")
    action_probs, q_values, state_value = inference(obs, legal_actions)

    print(f"   State value: {state_value.item():.4f}")
    best_action = get_best_action(action_probs, legal_actions)
    print(f"   Best action: {best_action} ({get_action_name(best_action, state)})")
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
    print("\n" + "=" * 70)
    print("TEST 3: MULTI-GAMMA ANALYSIS")
    print("=" * 70)

    # Load model
    abra = get_pretrained_model("Abra")
    agent = abra.initialize_agent(checkpoint=40, log=False)
    policy = agent.policy
    policy.eval()
    device = agent.DEVICE

    # Create test state
    state = create_test_state()
    obs = abra.observation_space.state_to_obs(state)  # Use full observation space
    legal_actions = get_legal_actions(state)

    obs_torch = prepare_observation(
        obs, legal_actions, abra.action_space.gym_space.n, device
    )
    rl2s, time_idxs, hidden_state = init_inference_inputs(1, device, policy)

    print("\nHow different discount factors affect action selection:")
    print(f"\n{'Gamma':<10} {'Top Action':<12} {'Prob':<10} {'Q-Value':<10} {'Move'}")
    print("-" * 60)

    for gamma_idx, gamma in enumerate(policy.gammas):
        action_probs, q_values, state_value, _, _ = get_policy_and_value(
            policy, obs_torch, rl2s, time_idxs, hidden_state, gamma_idx=gamma_idx
        )

        top_action = torch.argmax(action_probs).item()
        top_prob = action_probs[top_action].item()
        top_q = q_values[top_action].item()
        move_name = get_action_name(top_action, state)[:20]  # Truncate for display

        print(
            f"{gamma.item():<10.3f} {top_action:<12} {top_prob:<10.4f} {top_q:<10.4f} {move_name}"
        )


def test_markdown_cases():
    """Execute scenarios defined in test_cases.md."""

    print("\n" + "=" * 70)
    print("TEST 4: MARKDOWN-DEFINED SCENARIOS")
    print("=" * 70)

    cases = load_test_cases()
    if not cases:
        print("\nNo markdown test cases found.")
        return

    abra = get_pretrained_model("Abra")
    agent = abra.initialize_agent(checkpoint=40, log=False)
    policy = agent.policy
    policy.eval()
    device = agent.DEVICE

    for case in cases:
        print("\n" + "-" * 70)
        print(f"Case {case['id']}: {case['name']}")
        print(case["description"])

        factory_name = case["state_factory"]
        if factory_name not in STATE_FACTORY_REGISTRY:
            raise KeyError(f"Unknown state factory '{factory_name}' in test_cases.md")

        state_factory = STATE_FACTORY_REGISTRY[factory_name]
        state = state_factory()

        legal_actions = get_legal_actions(state)
        print(f"   Legal actions: {legal_actions}")

        obs = abra.observation_space.state_to_obs(state)
        obs_torch = prepare_observation(
            obs, legal_actions, abra.action_space.gym_space.n, device
        )
        rl2s, time_idxs, hidden_state = init_inference_inputs(1, device, policy)

        action_probs, q_values, state_value, _, _ = get_policy_and_value(
            policy, obs_torch, rl2s, time_idxs, hidden_state, gamma_idx=-1
        )

        best_action_idx = get_best_action(action_probs, legal_actions=legal_actions)
        best_action_name = get_action_name(best_action_idx, state)
        print(f"   State value: {state_value.item():.4f}")
        print(f"   Best action index: {best_action_idx}")
        print(f"   Best action name: {best_action_name}")
        print("   Probabilities (top 4):")

        top_probs, top_indices = torch.topk(action_probs, 4)
        for rank in range(top_probs.shape[0]):
            idx = top_indices[rank].item()
            name = get_action_name(idx, state)
            print(
                f"      #{rank + 1}: idx={idx:<2} name={name:<20} prob={top_probs[rank].item():.4f}"
            )

        expected = case["expected"]
        expectation_type = expected.get("type")
        passed = True

        if expectation_type == "best_action":
            expected_name = expected["action_name"]
            if best_action_name == expected_name:
                print(f"   ✓ Best action matches expectation: {expected_name}")
            else:
                passed = False
                print(
                    f"   ✗ Expected best action '{expected_name}' but model preferred '{best_action_name}'"
                )
        elif expectation_type == "probability_mix":
            action_names = expected["actions"]
            indices = []
            for action_name in action_names:
                try:
                    idx = get_action_index_by_name(action_name, state)
                    indices.append(idx)
                except ValueError as err:
                    passed = False
                    print(f"   ✗ {err}")
            illegal = [idx for idx in indices if idx not in legal_actions]
            if illegal:
                passed = False
                print(f"   ✗ Actions {illegal} not legal in this state")

            probs = [action_probs[idx].item() for idx in indices] if indices else []
            if probs:
                max_prob = max(probs)
                min_prob = min(probs)
                diff = max_prob - min_prob
                threshold = expected.get("max_difference", 0.1)

                if best_action_name in action_names:
                    print(
                        f"   ✓ Best action '{best_action_name}' lies inside desired mix {action_names}"
                    )
                else:
                    passed = False
                    print(
                        f"   ✗ Best action '{best_action_name}' not in desired mix {action_names}"
                    )

                within_threshold = diff <= threshold + 1e-6
                status = "✓" if within_threshold else "✗"
                print(
                    f"   {status} Probability gap {diff:.4f} vs allowed {threshold:.4f}: {action_names} -> {probs}"
                )
                passed = passed and within_threshold
        else:
            print(f"   ✗ Unsupported expectation type '{expectation_type}'")
            passed = False

        print(f"   Result: {'PASS' if passed else 'FAIL'}")

    print("\n   Scenarios processed; inspect logs above for pass/fail details")


def test_poke_engine_state_conversion():
    """Ensure poke-engine state conversion matches direct observation generation."""

    dex = Dex.from_gen(9)

    player_specs = [
        PokemonSpec(
            species="Iron Valiant",
            move_ids=["moonblast", "nightslash", "nuzzle", "thunderbolt"],
            move_pps=[16, 15, 20, 24],
            ability="Quark Drive",
            tera_type="Fairy",
            hp_pct=0.75,
            item="choicespecs",
        ),
        PokemonSpec(
            species="Corviknight",
            move_ids=["roost", "bravebird", "uturn", "bodypress"],
            move_pps=[16, 16, 32, 16],
            ability="Pressure",
            tera_type="Flying",
            hp_pct=1.0,
            item="leftovers",
        ),
        PokemonSpec(
            species="Toxapex",
            move_ids=["scald", "recover", "haze", "toxicspikes"],
            move_pps=[16, 16, 48, 24],
            ability="Regenerator",
            tera_type="Water",
            hp_pct=0.0,
            item="blacksludge",
            status="fnt",
        ),
        PokemonSpec(
            species="Dragapult",
            move_ids=["dragondarts", "shadowball", "uturn", "thunderbolt"],
            move_pps=[16, 16, 32, 24],
            ability="Infiltrator",
            tera_type="Ghost",
            hp_pct=1.0,
            item="choicespecs",
        ),
        PokemonSpec(
            species="Great Tusk",
            move_ids=["headlongrush", "closecombat", "rapidspin", "stealthrock"],
            move_pps=[16, 8, 40, 32],
            ability="Protosynthesis",
            tera_type="Ground",
            hp_pct=0.5,
            item="leftovers",
        ),
        PokemonSpec(
            species="Gholdengo",
            move_ids=["makeitrain", "shadowball", "focusblast", "nastyplot"],
            move_pps=[8, 24, 8, 32],
            ability="Good as Gold",
            tera_type="Steel",
            hp_pct=0.0,
            item="airballoon",
            status="fnt",
        ),
    ]

    opponent_specs = [
        PokemonSpec(
            species="Garchomp",
            move_ids=["earthquake", "swordsdance", "stoneedge", "dragonclaw"],
            move_pps=[16, 32, 8, 15],
            ability="Rough Skin",
            tera_type="Dragon",
            hp_pct=0.6,
            item="lumberry",
        ),
        PokemonSpec(
            species="Rotom-Wash",
            move_ids=["hydropump", "voltswitch", "willowisp", "protect"],
            move_pps=[8, 16, 24, 16],
            ability="Levitate",
            tera_type="Water",
            hp_pct=0.5,
            item="leftovers",
        ),
        PokemonSpec(
            species="Kingambit",
            move_ids=["kowtowcleave", "suckerpunch", "ironhead", "swordsdance"],
            move_pps=[15, 8, 15, 20],
            ability="Supreme Overlord",
            tera_type="Dark",
            hp_pct=1.0,
            item="blackglasses",
        ),
        PokemonSpec(
            species="Amoonguss",
            move_ids=["spore", "sludgebomb", "gigaDrain", "pollenpuff"],
            move_pps=[15, 16, 16, 24],
            ability="Regenerator",
            tera_type="Water",
            hp_pct=0.7,
            item="rockyhelmet",
        ),
        PokemonSpec(
            species="Ting-Lu",
            move_ids=["ruination", "stompingtantrum", "spikes", "whirlwind"],
            move_pps=[10, 10, 32, 32],
            ability="Vessel of Ruin",
            tera_type="Ground",
            hp_pct=0.3,
            item="leftovers",
        ),
        PokemonSpec(
            species="Blissey",
            move_ids=["softboiled", "seismictoss", "thunderwave", "teleport"],
            move_pps=[16, 32, 32, 32],
            ability="Natural Cure",
            tera_type="Normal",
            hp_pct=0.0,
            item="leftovers",
            status="fnt",
        ),
    ]

    universal_state = build_universal_state_from_specs(
        dex,
        player_specs,
        opponent_specs,
        player_prev_index=1,
        opponent_prev_index=2,
    )
    print(universal_state)

    obs_space_direct = get_observation_space("OpponentMoveObservationSpace")
    obs_space_direct.reset()
    direct_obs = obs_space_direct.state_to_obs(universal_state)

    pe_state = build_poke_engine_state_from_specs(
        dex,
        player_specs,
        opponent_specs,
        player_prev_index=1,
        opponent_prev_index=2,
    )

    obs_space_pe = get_observation_space("OpponentMoveObservationSpace")
    obs_space_pe.reset()
    converted_obs = poke_engine_state_to_observation(
        pe_state,
        obs_space_pe,
        battle_format="gen9ou",
        perspective="side_one",
    )

    assert np.allclose(direct_obs["numbers"], converted_obs["numbers"])
    assert direct_obs["text"].item() == converted_obs["text"].item()

    converted_state = poke_engine_state_to_universal_state(
        pe_state,
        battle_format="gen9ou",
        perspective="side_one",
    )

    assert (
        converted_state.player_active_pokemon.name
        == universal_state.player_active_pokemon.name
    )
    assert np.isclose(
        converted_state.player_active_pokemon.hp_pct,
        universal_state.player_active_pokemon.hp_pct,
    )
    assert (
        converted_state.opponent_active_pokemon.name
        == universal_state.opponent_active_pokemon.name
    )


def main():
    """Run all tests."""
    # test_basic_inference()
    # test_stateful_inference()
    # test_multi_gamma()
    # test_markdown_cases()
    test_poke_engine_state_conversion()

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("- Use get_policy_and_value() for one-off inference")
    print("- Use PolicyValueInference() for stateful sequential inference")
    print("- action_probs gives you π(a|s) for all actions")
    print("- q_values gives you Q(s,a) for all actions")
    print("- state_value is V(s) = E_a[Q(s,a)] under current policy")
    print("- Perfect for MCTS: policy prior + value estimate in one pass!")


if __name__ == "__main__":
    main()
