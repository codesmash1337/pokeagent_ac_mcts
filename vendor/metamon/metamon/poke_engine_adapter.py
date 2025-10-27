"""Utilities for converting poke-engine states into Metamon-friendly formats."""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Tuple
import time

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

_KNOWN_EFFECTS = {clean_no_numbers(effect.name) for effect in ShowdownEffect}

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


@lru_cache(maxsize=None)
def _dex_for_format(battle_format: str) -> Dex:
    return Dex.from_format(battle_format)


def _move_lookup_key(move_id: str) -> str:
    return clean_name(move_id).lower()


@lru_cache(maxsize=4096)
def _get_move_static_data(
    move_id: str, dex_gen: int
) -> Tuple[str, str, str, int, float, int, int]:
    """Cache static move data (everything except current PP)."""
    dex = Dex.from_gen(dex_gen)
    move_key = _move_lookup_key(move_id)
    move_entry = dex.moves.get(move_key)
    if move_entry is None:
        raise KeyError(f"Move '{move_id}' not found in Dex; cannot build observation")

    accuracy = move_entry.get("accuracy", 100)
    if accuracy is True:
        accuracy = 100

    base_power = move_entry.get("basePower", 0)
    category = clean_name(move_entry.get("category", "status"))
    move_type = _normalize_type(move_entry.get("type", "typeless"))
    priority = move_entry.get("priority", 0)
    max_pp = int(move_entry.get("pp", 0) * 1.6)
    name = clean_name(move_entry.get("name", move_id))
    accuracy_float = (
        float(accuracy) / 100.0 if isinstance(accuracy, (int, float)) else 1.0
    )

    return (name, move_type, category, base_power, accuracy_float, priority, max_pp)


@lru_cache(maxsize=16384)
def _cached_universal_move(
    move_id: str, current_pp: int, dex_gen: int
) -> UniversalMove:
    name, move_type, category, base_power, accuracy, priority, max_pp = (
        _get_move_static_data(move_id, dex_gen)
    )
    return UniversalMove(
        name=name,
        move_type=move_type,
        category=category,
        base_power=base_power,
        accuracy=accuracy,
        priority=priority,
        current_pp=current_pp,
        max_pp=max_pp if max_pp else current_pp,
    )


def _universal_move_from_pe(move: PEMove, dex: Dex) -> UniversalMove:
    if move is None or move.id in {"", "none"}:
        raise ValueError(
            "poke-engine move missing identifier; cannot build observation"
        )

    return _cached_universal_move(move.id, int(move.pp), dex.gen)


@lru_cache(maxsize=1024)
def _pokemon_base_stats(
    species: str, dex_format: str
) -> Tuple[int, int, int, int, int, int]:
    """Cache pokemon base stats lookups."""
    dex = _dex_for_format(dex_format)
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


@lru_cache(maxsize=1024)
def _pokemon_base_species(species: str, dex_format: str) -> str:
    """Cache pokemon base species lookups."""
    dex = _dex_for_format(dex_format)
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
        for effect in (clean_no_numbers(candidate) for candidate in effect_candidates)
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
    dex_format: str,
    is_active: bool,
    boost_tuple: Tuple[int, int, int, int, int, int, int],
    dex: Dex,
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

    move_objs = [
        _universal_move_from_pe(move, dex) for move in pokemon.moves[:4]
    ]

    atk, spa, df, sd, sp, hp_base = _pokemon_base_stats(pokemon.id, dex_format)
    # Only use tera type if Pokemon is actually terastallized, otherwise use notype
    if getattr(pokemon, "terastallized", False):
        tera_type = _normalize_type(getattr(pokemon, "tera_type", "typeless"))
    else:
        tera_type = "notype"
    base_species = _pokemon_base_species(pokemon.id, dex_format)

    if not pokemon.maxhp or pokemon.maxhp <= 0:
        raise ValueError(
            f"Pokemon '{pokemon.id}' has invalid max HP ({pokemon.maxhp}); cannot build observation"
        )

    hp_pct = max(0.0, min(1.0, pokemon.hp / pokemon.maxhp))

    # Prepare string fields
    pkmn_name = pokemon_name(pokemon.id)
    types_str = " ".join(sorted(_normalize_type(t) for t in pokemon.types))
    item_str = _normalize_item(pokemon.item)
    ability_str = _normalize_ability(pokemon.ability)
    status_str = _normalize_status(pokemon.status)

    effect = "noeffect"
    if is_active and side is not None:
        effect = _active_effect_from_side(side)

    result = UniversalPokemon(
        name=pkmn_name,
        base_species=base_species,
        hp_pct=hp_pct,
        types=types_str,
        item=item_str,
        ability=ability_str,
        lvl=pokemon.level,
        status=status_str,
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

    return result


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
    dex_format: str,
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
    dex_format: str,
    timings: Optional[Dict[str, float]] = None,
) -> List[UniversalPokemon]:
    boost_defaults = (0, 0, 0, 0, 0, 0, 0)
    switches: List[UniversalPokemon] = []

    if getattr(side, "force_trapped", False):
        return switches

    for idx, pokemon in enumerate(side.pokemon):
        if idx == active_index or pokemon.hp <= 0:
            continue
        t_total_start = time.perf_counter() if timings is not None else 0.0
        bench = _bench_universal_pokemon(
            pokemon,
            dex_format=dex_format,
            timings=timings,
        )
        if timings is not None:
            t_total_end = time.perf_counter()
            timings["switch_total"] = timings.get("switch_total", 0.0) + (
                t_total_end - t_total_start
            ) * 1000
        switches.append(bench)
    return switches


SwitchCacheKey = Tuple[
    str,
    str,
    int,
    int,
    int,
    Tuple[str, ...],
    str,
    str,
    str,
    bool,
    str,
    Tuple[Tuple[str, int], ...],
]


@lru_cache(maxsize=16384)
def _sanitize_bench_static(
    species_id: str,
    raw_types: Tuple[str, ...],
    raw_item: str,
    raw_ability: str,
    raw_status: str,
    tera_flag: bool,
    raw_tera_type: str,
) -> Tuple[Tuple[str, ...], str, str, str, str]:
    normalized_types = tuple(
        sorted(_normalize_type(type_name) for type_name in raw_types if type_name)
    )
    if not normalized_types:
        normalized_types = ("notype",)

    normalized_item = _normalize_item(raw_item)
    normalized_ability = _normalize_ability(raw_ability)
    normalized_status = _normalize_status(raw_status)
    normalized_tera = (
        _normalize_type(raw_tera_type) if tera_flag else "notype"
    )

    return (
        normalized_types,
        normalized_item,
        normalized_ability,
        normalized_status,
        normalized_tera,
    )


def _bench_cache_key(pokemon: PEPokemon, dex_format: str) -> SwitchCacheKey:
    moves = getattr(pokemon, "moves", None) or []
    if len(moves) < 4:
        raise ValueError(
            f"Bench pokemon '{pokemon.id}' has fewer than 4 moves ({len(moves)}); cannot build observation"
        )
    move_entries = []
    for move in moves[:4]:
        if move is None or move.id in {"", "none"}:
            raise ValueError(
                f"Bench pokemon '{pokemon.id}' contains an invalid move entry"
            )
        move_entries.append((move.id, int(move.pp)))

    raw_types = tuple(getattr(pokemon, "types", []) or [])
    raw_item = getattr(pokemon, "item", "")
    raw_ability = getattr(pokemon, "ability", "")
    raw_status = getattr(pokemon, "status", "")
    tera_flag = bool(getattr(pokemon, "terastallized", False))
    raw_tera_type = getattr(pokemon, "tera_type", "typeless")

    (
        norm_types,
        norm_item,
        norm_ability,
        norm_status,
        norm_tera,
    ) = _sanitize_bench_static(
        pokemon.id,
        raw_types,
        raw_item,
        raw_ability,
        raw_status,
        tera_flag,
        raw_tera_type,
    )

    return (
        dex_format,
        pokemon.id,
        int(getattr(pokemon, "hp", 0)),
        int(getattr(pokemon, "maxhp", 1)),
        int(getattr(pokemon, "level", 0)),
        norm_types,
        norm_item,
        norm_ability,
        norm_status,
        tera_flag,
        norm_tera,
        tuple(move_entries),
    )


@lru_cache(maxsize=8192)
def _bench_universal_from_key(key: SwitchCacheKey) -> UniversalPokemon:
    (
        dex_format,
        species_id,
        hp,
        maxhp,
        level,
        types,
        item,
        ability,
        status,
        tera_flag,
        tera_type,
        moves,
    ) = key

    dex = _dex_for_format(dex_format)
    atk, spa, df, sd, sp, hp_base = _pokemon_base_stats(species_id, dex_format)
    base_species = _pokemon_base_species(species_id, dex_format)
    name = pokemon_name(species_id)

    hp_pct = 0.0 if maxhp <= 0 else max(0.0, min(1.0, hp / maxhp))
    types_str = "notype" if not types else " ".join(types)

    move_objs = [
        _cached_universal_move(move_id, pp, dex.gen)
        for move_id, pp in moves
    ]

    return UniversalPokemon(
        name=name,
        base_species=base_species,
        hp_pct=hp_pct,
        types=types_str,
        item=item,
        ability=ability,
        lvl=level,
        status=status,
        effect="noeffect",
        moves=move_objs,
        atk_boost=0,
        spa_boost=0,
        def_boost=0,
        spd_boost=0,
        spe_boost=0,
        accuracy_boost=0,
        evasion_boost=0,
        base_atk=atk,
        base_spa=spa,
        base_def=df,
        base_spd=sd,
        base_spe=sp,
        base_hp=hp_base,
        tera_type=tera_type if tera_flag else "notype",
    )


def _bench_universal_pokemon(
    pokemon: PEPokemon,
    *,
    dex_format: str,
    timings: Optional[Dict[str, float]] = None,
) -> UniversalPokemon:
    t_key_start = time.perf_counter() if timings is not None else 0.0
    key = _bench_cache_key(pokemon, dex_format)
    if timings is not None:
        t_key_end = time.perf_counter()
        timings["switch_key"] = timings.get("switch_key", 0.0) + (
            t_key_end - t_key_start
        ) * 1000
    else:
        t_key_end = 0.0

    cached = _bench_universal_from_key(key)
    if timings is not None:
        t_lookup_end = time.perf_counter()
        timings["switch_lookup"] = timings.get("switch_lookup", 0.0) + (
            t_lookup_end - t_key_end
        ) * 1000
    else:
        t_lookup_end = 0.0

    result = replace(
        cached,
        moves=[replace(move) for move in cached.moves],
    )
    if timings is not None:
        t_clone_end = time.perf_counter()
        timings["switch_clone"] = timings.get("switch_clone", 0.0) + (
            t_clone_end - t_lookup_end
        ) * 1000
    return result


def _count_remaining(pokemon_list: Iterable[PEPokemon]) -> int:
    return sum(1 for pokemon in pokemon_list if pokemon.hp > 0)


_TeampreviewKey = Tuple[str, ...]


def _teampreview_key(side: PESide) -> _TeampreviewKey:
    identifiers = []
    for pokemon in side.pokemon:
        identifier = (getattr(pokemon, "id", "") or "").strip()
        if not identifier or identifier.lower() == "none":
            raise ValueError(
                "Encountered placeholder Pokemon without identifier in poke-engine state"
            )
        identifiers.append(identifier)
    return tuple(identifiers)


@lru_cache(maxsize=4096)
def _cached_teampreview(battle_format: str, key: _TeampreviewKey) -> Tuple[str, ...]:
    names = [pokemon_name(identifier) for identifier in key]
    while len(names) < 6:
        names.append("<blank>")
    return tuple(names)


def _extract_teampreview(side: PESide, battle_format: str) -> List[str]:
    key = _teampreview_key(side)
    cached = _cached_teampreview(battle_format, key)
    return list(cached)


def poke_engine_state_to_universal_state(
    state: PEState,
    *,
    battle_format: str,
    perspective: str = "side_one",
    timings: Optional[Dict[str, float]] = None,
) -> UniversalState:
    t_start = time.perf_counter()
    player_side, opponent_side = (
        (state.side_one, state.side_two)
        if perspective == "side_one"
        else (state.side_two, state.side_one)
    )

    dex = _dex_for_format(battle_format)
    t_dex = time.perf_counter()
    if timings is not None:
        timings["dex"] = timings.get("dex", 0.0) + (t_dex - t_start) * 1000

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
    t_boosts = time.perf_counter()
    if timings is not None:
        timings["boosts"] = timings.get("boosts", 0.0) + (t_boosts - t_dex) * 1000

    player_universal = _universal_pokemon_from_pe(
        player_active,
        dex_format=battle_format,
        is_active=True,
        boost_tuple=player_boosts,
        dex=dex,
        side=player_side,
    )
    opponent_universal = _universal_pokemon_from_pe(
        opponent_active,
        dex_format=battle_format,
        is_active=True,
        boost_tuple=opponent_boosts,
        dex=dex,
        side=opponent_side,
    )
    t_pokemon = time.perf_counter()
    if timings is not None:
        timings["pokemon"] = timings.get("pokemon", 0.0) + (t_pokemon - t_boosts) * 1000

    available_switches = _collect_available_switches(
        player_side,
        active_index,
        battle_format,
        timings=timings,
    )
    t_switches = time.perf_counter()
    if timings is not None:
        timings["switches"] = timings.get("switches", 0.0) + (t_switches - t_pokemon) * 1000

    player_prev_move = _last_used_move_to_universal(
        player_side, player_active, battle_format, dex
    )
    opponent_prev_move = _last_used_move_to_universal(
        opponent_side, opponent_active, battle_format, dex
    )
    t_moves = time.perf_counter()
    if timings is not None:
        timings["moves"] = timings.get("moves", 0.0) + (t_moves - t_switches) * 1000

    opponents_remaining = _count_remaining(opponent_side.pokemon)

    player_conditions = _side_conditions_to_str(player_side)
    opponent_conditions = _side_conditions_to_str(opponent_side)
    t_conditions = time.perf_counter()
    if timings is not None:
        timings["conditions"] = timings.get("conditions", 0.0) + (t_conditions - t_moves) * 1000

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
    t_field = time.perf_counter()
    if timings is not None:
        timings["field"] = timings.get("field", 0.0) + (t_field - t_conditions) * 1000

    forced_switch = bool(player_side.force_switch)

    player_alive = any(pokemon.hp > 0 for pokemon in player_side.pokemon)
    opponent_alive = any(pokemon.hp > 0 for pokemon in opponent_side.pokemon)
    battle_won = player_alive and not opponent_alive
    battle_lost = opponent_alive and not player_alive

    can_tera = not any(p.terastallized for p in player_side.pokemon)
    t_status = time.perf_counter()
    if timings is not None:
        timings["status"] = timings.get("status", 0.0) + (t_status - t_field) * 1000

    opponent_teampreview = _extract_teampreview(opponent_side, battle_format)
    t_end = time.perf_counter()
    if timings is not None:
        timings["preview"] = timings.get("preview", 0.0) + (t_end - t_status) * 1000
        timings["total"] = timings.get("total", 0.0) + (t_end - t_start) * 1000

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
