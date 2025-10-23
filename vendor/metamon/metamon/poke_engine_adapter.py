"""Utilities for converting poke-engine states into Metamon-friendly formats."""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

from poke_engine import Pokemon as PEPokemon
from poke_engine import Move as PEMove
from poke_engine import Side as PESide
from poke_engine import State as PEState
from poke_env.environment.effect import Effect as ShowdownEffect
from poke_env.environment.side_condition import (
    SideCondition as ShowdownSideCondition,
)

from metamon.backend.showdown_dex import Dex
from metamon.backend.replay_parser.str_parsing import (
    clean_name,
    clean_no_numbers,
    pokemon_name,
)
from metamon.interface import UniversalMove, UniversalPokemon, UniversalState

_STATUS_MAP = {
    "none": "nostatus",
    "burn": "brn",
    "sleep": "slp",
    "freeze": "frz",
    "paralyze": "par",
    "poison": "psn",
    "toxic": "tox",
}

_KNOWN_EFFECTS = {
    clean_no_numbers(effect.name) for effect in ShowdownEffect
}

_EFFECT_PRIORITY = [
    "substitute",
    "destinybond",
    "perish",
    "yawn",
    "encore",
    "taunt",
    "lockedmove",
    "confusion",
    "partiallytrapped",
    "uproar",
    "disable",
    "torment",
    "curse",
    "leechseed",
]

_KNOWN_SIDE_CONDITIONS = {
    clean_no_numbers(condition.name) for condition in ShowdownSideCondition
}


def _normalize_status(status: str) -> str:
    status = status.lower()
    return _STATUS_MAP.get(status, clean_no_numbers(status))


def _normalize_type(type_name: str) -> str:
    if not type_name or type_name.lower() in {"typeless", "none"}:
        return "notype"
    return clean_name(type_name)


def _normalize_item(item: str) -> str:
    item = (item or "").strip().lower()
    if item in {"", "none", "noitem"}:
        return "noitem"
    return clean_name(item)


def _normalize_ability(ability: str) -> str:
    ability = (ability or "").strip().lower()
    if ability in {"", "none", "noability"}:
        return "noability"
    return clean_name(ability)


def _dex_for_format(battle_format: str) -> Dex:
    return Dex.from_format(battle_format)


def _move_lookup_key(move_id: str) -> str:
    return clean_name(move_id).lower()


def _universal_move_from_pe(move: PEMove, dex: Dex) -> UniversalMove:
    if move is None or move.id in {"", "none"}:
        raise ValueError(
            "poke-engine move missing identifier; cannot build observation"
        )

    move_key = _move_lookup_key(move.id)
    move_entry = dex.moves.get(move_key)
    if move_entry is None:
        raise KeyError(f"Move '{move.id}' not found in Dex; cannot build observation")

    accuracy = move_entry.get("accuracy", 100)
    if accuracy is True:
        accuracy = 100

    base_power = move_entry.get("basePower", 0)
    category = clean_name(move_entry.get("category", "status"))
    move_type = _normalize_type(move_entry.get("type", "typeless"))
    priority = move_entry.get("priority", 0)
    max_pp = int(move_entry.get("pp", 0) * 1.6)

    return UniversalMove(
        name=clean_name(move_entry.get("name", move.id)),
        move_type=move_type,
        category=category,
        base_power=base_power,
        accuracy=float(accuracy) / 100.0 if isinstance(accuracy, (int, float)) else 1.0,
        priority=priority,
        current_pp=move.pp,
        max_pp=max_pp if max_pp else move.pp,
    )


def _pokemon_base_stats(dex: Dex, species: str) -> Tuple[int, int, int, int, int, int]:
    dex_entry = dex.get_pokedex_entry(species)
    stats = dex_entry["baseStats"]
    return (
        stats["atk"],
        stats["spa"],
        stats["def"],
        stats["spd"],
        stats["spe"],
        stats["hp"],
    )


def _pokemon_base_species(dex: Dex, species: str) -> str:
    dex_entry = dex.get_pokedex_entry(species)
    return clean_name(dex_entry.get("baseSpecies", species))


def _active_effect_from_side(side: PESide) -> str:
    effect_candidates = set()

    volatile_statuses = getattr(side, "volatile_statuses", set()) or set()
    for status in volatile_statuses:
        sanitized = clean_no_numbers(status)
        if sanitized:
            effect_candidates.add(sanitized)

    durations = getattr(side, "volatile_status_durations", None)
    if durations is not None:
        duration_fields = {
            "confusion": getattr(durations, "confusion", 0),
            "encore": getattr(durations, "encore", 0),
            "lockedmove": getattr(durations, "lockedmove", 0),
            "slowstart": getattr(durations, "slowstart", 0),
            "taunt": getattr(durations, "taunt", 0),
            "yawn": getattr(durations, "yawn", 0),
        }
        for name, value in duration_fields.items():
            if value and value > 0:
                effect_candidates.add(clean_no_numbers(name))

    if getattr(side, "substitute_health", 0) > 0:
        effect_candidates.add("substitute")

    if getattr(side, "force_trapped", False):
        effect_candidates.add("partiallytrapped")

    future_sight_turns, _ = getattr(side, "future_sight", (0, "0"))
    if future_sight_turns and future_sight_turns > 0:
        effect_candidates.add("futuresight")

    # Keep only effects that exist in the showdown enum vocabulary
    normalized_effects = {
        effect
        for effect in (
            clean_no_numbers(candidate) for candidate in effect_candidates
        )
        if effect in _KNOWN_EFFECTS
    }

    if not normalized_effects:
        return "noeffect"

    for preferred in _EFFECT_PRIORITY:
        if preferred in normalized_effects:
            return preferred

    return sorted(normalized_effects)[0]


def _universal_pokemon_from_pe(
    pokemon: PEPokemon,
    *,
    dex: Dex,
    is_active: bool,
    boost_tuple: Tuple[int, int, int, int, int, int, int],
    side: Optional[PESide] = None,
) -> UniversalPokemon:
    atk_boost, spa_boost, def_boost, spd_boost, spe_boost, acc_boost, eva_boost = (
        boost_tuple
    )

    if not pokemon.moves:
        raise ValueError(
            f"Pokemon '{pokemon.id}' is missing move data; cannot build observation"
        )

    if len(pokemon.moves) < 4:
        raise ValueError(
            f"Pokemon '{pokemon.id}' has fewer than 4 moves ({len(pokemon.moves)}); cannot build observation"
        )

    move_objs = [_universal_move_from_pe(move, dex) for move in pokemon.moves[:4]]

    atk, spa, df, sd, sp, hp_base = _pokemon_base_stats(dex, pokemon.id)
    tera_type = _normalize_type(getattr(pokemon, "tera_type", "typeless"))
    base_species = _pokemon_base_species(dex, pokemon.id)

    if not pokemon.maxhp or pokemon.maxhp <= 0:
        raise ValueError(
            f"Pokemon '{pokemon.id}' has invalid max HP ({pokemon.maxhp}); cannot build observation"
        )

    hp_pct = max(0.0, min(1.0, pokemon.hp / pokemon.maxhp))

    effect = "noeffect"
    if is_active and side is not None:
        effect = _active_effect_from_side(side)

    return UniversalPokemon(
        name=pokemon_name(pokemon.id),
        base_species=base_species,
        hp_pct=hp_pct,
        types=" ".join(sorted(_normalize_type(t) for t in pokemon.types)),
        item=_normalize_item(pokemon.item),
        ability=_normalize_ability(pokemon.ability),
        lvl=pokemon.level,
        status=_normalize_status(pokemon.status),
        effect=effect,
        moves=move_objs,
        atk_boost=atk_boost if is_active else 0,
        spa_boost=spa_boost if is_active else 0,
        def_boost=def_boost if is_active else 0,
        spd_boost=spd_boost if is_active else 0,
        spe_boost=spe_boost if is_active else 0,
        accuracy_boost=acc_boost if is_active else 0,
        evasion_boost=eva_boost if is_active else 0,
        base_atk=atk,
        base_spa=spa,
        base_def=df,
        base_spd=sd,
        base_spe=sp,
        base_hp=hp_base,
        tera_type=tera_type,
    )


def _side_conditions_to_str(side: PESide) -> str:
    sc = side.side_conditions
    mapping = {
        "aurora_veil": sc.aurora_veil,
        "crafty_shield": sc.crafty_shield,
        "healing_wish": sc.healing_wish,
        "light_screen": sc.light_screen,
        "lucky_chant": sc.lucky_chant,
        "lunar_dance": sc.lunar_dance,
        "mat_block": sc.mat_block,
        "mist": sc.mist,
        "protect": sc.protect,
        "quick_guard": sc.quick_guard,
        "reflect": sc.reflect,
        "safeguard": sc.safeguard,
        "spikes": sc.spikes,
        "stealth_rock": sc.stealth_rock,
        "sticky_web": sc.sticky_web,
        "tailwind": sc.tailwind,
        "toxic_count": sc.toxic_count,
        "toxic_spikes": sc.toxic_spikes,
        "wide_guard": sc.wide_guard,
    }
    active_conditions = {}
    for name, value in mapping.items():
        if not value or value <= 0:
            continue
        sanitized = clean_no_numbers(name)
        if sanitized in _KNOWN_SIDE_CONDITIONS:
            active_conditions[sanitized] = value

    if not active_conditions:
        return "noconditions"

    if {"reflect", "lightscreen"}.issubset(active_conditions.keys()):
        return "screens"

    chosen = max(active_conditions.items(), key=lambda item: (item[1], item[0]))
    return chosen[0]


def _side_boosts(side: PESide) -> Tuple[int, int, int, int, int, int, int]:
    return (
        side.attack_boost,
        side.special_attack_boost,
        side.defense_boost,
        side.special_defense_boost,
        side.speed_boost,
        side.accuracy_boost,
        side.evasion_boost,
    )


def _last_used_move_to_universal(
    side: PESide,
    active_pokemon: PEPokemon,
    dex: Dex,
) -> UniversalMove:
    raw = side.last_used_move or "move:none"
    if not raw.startswith("move:"):
        return UniversalMove.blank_move()
    _, _, index = raw.partition(":")
    try:
        move_index = int(index)
    except ValueError:
        return UniversalMove.blank_move()
    moves = active_pokemon.moves or []
    if 0 <= move_index < len(moves):
        return _universal_move_from_pe(moves[move_index], dex)
    return UniversalMove.blank_move()


def _collect_available_switches(
    side: PESide,
    active_index: int,
    dex: Dex,
) -> List[UniversalPokemon]:
    boost_defaults = (0, 0, 0, 0, 0, 0, 0)
    switches: List[UniversalPokemon] = []

    if getattr(side, "force_trapped", False):
        return switches

    for idx, pokemon in enumerate(side.pokemon):
        if idx == active_index or pokemon.hp <= 0:
            continue
        switches.append(
            _universal_pokemon_from_pe(
                pokemon,
                dex=dex,
                is_active=False,
                boost_tuple=boost_defaults,
            )
        )
    return switches


def _count_remaining(pokemon_list: Iterable[PEPokemon]) -> int:
    return sum(1 for pokemon in pokemon_list if pokemon.hp > 0)


def _extract_teampreview(side: PESide) -> List[str]:
    names = []
    for pokemon in side.pokemon:
        identifier = (getattr(pokemon, "id", "") or "").strip()
        if not identifier or identifier.lower() == "none":
            raise ValueError(
                "Encountered placeholder Pokemon without identifier in poke-engine state"
            )
        names.append(pokemon_name(identifier))
    while len(names) < 6:
        names.append("<blank>")
    return names


def poke_engine_state_to_universal_state(
    state: PEState,
    *,
    battle_format: str,
    perspective: str = "side_one",
) -> UniversalState:
    player_side, opponent_side = (
        (state.side_one, state.side_two)
        if perspective == "side_one"
        else (state.side_two, state.side_one)
    )

    dex = _dex_for_format(battle_format)

    if len(player_side.pokemon) != 6 or len(opponent_side.pokemon) != 6:
        raise ValueError("poke-engine state must contain exactly 6 pokemon per side")

    try:
        active_index = int(player_side.active_index)
        opponent_active_index = int(opponent_side.active_index)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid active index in poke-engine state") from exc

    player_active = player_side.pokemon[active_index]
    player_boosts = _side_boosts(player_side)

    opponent_active = opponent_side.pokemon[opponent_active_index]
    opponent_boosts = _side_boosts(opponent_side)

    player_universal = _universal_pokemon_from_pe(
        player_active,
        dex=dex,
        is_active=True,
        boost_tuple=player_boosts,
        side=player_side,
    )
    opponent_universal = _universal_pokemon_from_pe(
        opponent_active,
        dex=dex,
        is_active=True,
        boost_tuple=opponent_boosts,
        side=opponent_side,
    )

    available_switches = _collect_available_switches(player_side, active_index, dex)

    player_prev_move = _last_used_move_to_universal(player_side, player_active, dex)
    opponent_prev_move = _last_used_move_to_universal(
        opponent_side, opponent_active, dex
    )

    opponents_remaining = _count_remaining(opponent_side.pokemon)

    player_conditions = _side_conditions_to_str(player_side)
    opponent_conditions = _side_conditions_to_str(opponent_side)

    weather = clean_no_numbers(state.weather or "none")
    if weather in {"none", ""}:
        weather = "noweather"

    terrain = clean_no_numbers(state.terrain or "none")
    trick_room_active = bool(getattr(state, "trick_room", False))

    if trick_room_active:
        battle_field = "trickroom"
    elif terrain in {"none", ""}:
        battle_field = "nofield"
    else:
        battle_field = terrain

    forced_switch = bool(player_side.force_switch)

    player_alive = any(pokemon.hp > 0 for pokemon in player_side.pokemon)
    opponent_alive = any(pokemon.hp > 0 for pokemon in opponent_side.pokemon)
    battle_won = player_alive and not opponent_alive
    battle_lost = opponent_alive and not player_alive

    can_tera = not any(p.terastallized for p in player_side.pokemon)

    opponent_teampreview = _extract_teampreview(opponent_side)

    return UniversalState(
        format=battle_format.lower(),
        player_active_pokemon=player_universal,
        opponent_active_pokemon=opponent_universal,
        available_switches=available_switches,
        player_prev_move=player_prev_move,
        opponent_prev_move=opponent_prev_move,
        opponents_remaining=opponents_remaining,
        player_conditions=player_conditions,
        opponent_conditions=opponent_conditions,
        weather=weather,
        battle_field=battle_field,
        forced_switch=forced_switch,
        battle_won=battle_won,
        battle_lost=battle_lost,
        can_tera=can_tera,
        opponent_teampreview=opponent_teampreview,
    )


def poke_engine_state_to_observation(
    state: PEState,
    observation_space,
    *,
    battle_format: str,
    perspective: str = "side_one",
):
    observation_space.reset()
    universal_state = poke_engine_state_to_universal_state(
        state,
        battle_format=battle_format,
        perspective=perspective,
    )
    return observation_space.state_to_obs(universal_state)
