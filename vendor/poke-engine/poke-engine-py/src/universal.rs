use std::collections::{BTreeMap, HashMap, HashSet};
use std::time::Instant;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::PyState;
use poke_engine::choices::{Choice, MoveCategory};
use poke_engine::engine::abilities::Abilities;
use poke_engine::engine::items::Items;
use poke_engine::state::{Pokemon, PokemonIndex, PokemonMoves, PokemonStatus, Side, State};

#[derive(Debug)]
pub struct MoveData {
    pub name: String,
    pub sort_name: String,
    pub move_type: String,
    pub category: String,
    pub base_power: f32,
    pub accuracy: f32,
    pub priority: i8,
    pub current_pp: i8,
    pub max_pp: i8,
}

#[derive(Debug)]
struct OldMoveData {
    name: String,
    sort_name: String,
    move_type: String,
    category: String,
    base_power: f32,
    accuracy: f32,
    priority: i8,
    current_pp: i8,
    max_pp: i8, // NOTE: equal to current_pp because we can't reconstruct boosted PP without extra metadata
}

#[derive(Debug)]
struct PokemonData {
    name: String,
    sort_name: String,
    base_species: String, // NOTE: derived from the cleaned id; does not resolve base species overrides like the dex-backed Python helper
    hp_pct: f32,
    types_display: String,
    item: String,
    ability: String,
    level: i8,
    status: String, // NOTE: simple sanitize of enum string; no extra alias handling
    effect: String, // NOTE: only captures a subset of volatile statuses (matches current Rust logic)
    tera_type: String, // NOTE: represents only the tera typing; doesn't expose tera availability flags
    base_stats: (i16, i16, i16, i16, i16, i16),
    moves: Vec<MoveData>,
}

#[derive(Debug)]
pub struct PreparedStateData {
    pub universal_state: PyObject,
    pub legal_actions: Vec<usize>,
    pub move_mapping: BTreeMap<usize, usize>,
    pub switch_mapping: BTreeMap<usize, usize>,
}

#[derive(Debug, Clone, Copy)]
enum Perspective {
    SideOne,
    SideTwo,
}

impl Perspective {
    fn from_str(raw: &str) -> PyResult<Self> {
        match raw {
            "side_one" => Ok(Self::SideOne),
            "side_two" => Ok(Self::SideTwo),
            other => Err(PyValueError::new_err(format!(
                "perspective must be 'side_one' or 'side_two', got {}",
                other
            ))),
        }
    }
}

pub fn clean_name(input: &str) -> String {
    input
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .map(|c| c.to_ascii_lowercase())
        .collect()
}

pub fn clean_no_numbers(input: &str) -> String {
    input
        .chars()
        .filter(|c| c.is_ascii_alphabetic())
        .map(|c| c.to_ascii_lowercase())
        .collect()
}

pub fn pokemon_name_str(raw: &str) -> String {
    let cleaned = clean_name(raw).trim().to_string();
    // Map forms to base species (e.g., landorustherian -> landorus)
    crate::base_species_map::to_base_species(&cleaned).to_string()
}

pub fn normalize_type(raw: &str) -> String {
    let lowered = raw.to_ascii_lowercase();
    if lowered.is_empty() || lowered == "typeless" || lowered == "none" {
        "notype".to_string()
    } else {
        clean_name(raw)
    }
}

pub fn normalize_item(item: Items) -> String {
    let raw = item.to_string();
    let lowered = raw.to_ascii_lowercase();
    if lowered.is_empty() || lowered == "none" || lowered == "noitem" {
        "noitem".to_string()
    } else {
        clean_name(&raw) // NOTE: this is a straight enum string cleanup; no dex alias lookups like the Python dex helper
    }
}

pub fn normalize_ability(ability: Abilities) -> String {
    let raw = ability.to_string();
    let lowered = raw.to_ascii_lowercase();
    if lowered.is_empty() || lowered == "none" || lowered == "noability" {
        "noability".to_string()
    } else {
        clean_name(&raw) // NOTE: simple sanitization; doesn't remap abilities with special punctuation beyond lowercase stripping
    }
}

pub fn normalize_status(status: PokemonStatus) -> String {
    let raw = status.to_string();
    match raw.to_ascii_lowercase().as_str() {
        "none" => "nostatus".to_string(),
        "burn" => "brn".to_string(),
        "sleep" => "slp".to_string(),
        "freeze" => "frz".to_string(),
        "paralyze" | "paralysis" => "par".to_string(),  // Handle both PARALYZE enum and legacy paralysis
        "poison" => "psn".to_string(),
        "badly_poison" | "badly poison" | "tox" => "tox".to_string(),
        other => clean_no_numbers(other), // NOTE: catches anything else generically; python had a few extra aliases we skip here
    }
}

pub fn pokemon_types_display(pokemon: &Pokemon) -> String {
    // If terastallized, surface the tera type in the first slot and "notype" in the second
    if pokemon.terastallized {
        let raw = pokemon.tera_type.to_string();
        let tera = normalize_type(&raw);
        return format!("{} notype", tera);
    }
    // Otherwise show natural typing (sorted for stability)
    let mut types: Vec<String> = Vec::new();
    for t in [pokemon.types.0, pokemon.types.1] {
        let raw = t.to_string();
        let normalized = normalize_type(&raw);
        types.push(normalized);
    }
    types.sort();
    types.join(" ")
}

pub fn tera_type(pokemon: &Pokemon) -> String {
    if pokemon.terastallized {
        let raw = pokemon.tera_type.to_string();
        normalize_type(&raw) // NOTE: we only surface the active form; the Python adapter also tracked whether tera was still available
    } else {
        "notype".to_string()
    }
}

pub fn base_stats(pokemon: &Pokemon) -> (i16, i16, i16, i16, i16, i16) {
    pokemon.id.base_stats()
}

fn move_category_name(category: MoveCategory) -> &'static str {
    match category {
        MoveCategory::Physical => "physical",
        MoveCategory::Special => "special",
        MoveCategory::Status => "status",
        MoveCategory::Switch => "status", // NOTE: treat switches as status for downstream token ordering (matches prior python simplification)
    }
}

pub fn pokemon_index_to_usize(index: PokemonIndex) -> usize {
    match index {
        PokemonIndex::P0 => 0,
        PokemonIndex::P1 => 1,
        PokemonIndex::P2 => 2,
        PokemonIndex::P3 => 3,
        PokemonIndex::P4 => 4,
        PokemonIndex::P5 => 5,
    }
}

pub fn build_move_data(move_obj: &Choice, move_struct: &poke_engine::state::Move) -> MoveData {
    let move_name_raw = move_struct.id.to_string();
    let name = pokemon_name_str(&move_name_raw);
    let sort_name = clean_no_numbers(&move_name_raw);
    let move_type_string = normalize_type(&move_obj.move_type.to_string());
    let category = clean_name(move_category_name(move_obj.category));
    let base_power = move_obj.base_power;
    let mut accuracy = move_obj.accuracy;
    if accuracy > 1.0 {
        accuracy /= 100.0;
    }
    let priority = move_obj.priority;
    let current_pp = move_struct.pp;
    let max_pp = ((current_pp + 7) / 8) * 8; // Approximation because I don't want to store all max PPs

    MoveData {
        name,
        sort_name,
        move_type: move_type_string,
        category,
        base_power,
        accuracy,
        priority,
        current_pp,
        max_pp,
    }
}

pub fn pokemon_moves_to_vec(moves: &PokemonMoves) -> Vec<poke_engine::state::Move> {
    vec![moves.m0.clone(), moves.m1.clone(), moves.m2.clone(), moves.m3.clone()]
}

pub fn consistent_move_order_indices(moves: &[MoveData]) -> Vec<usize> {
    let mut pairs: Vec<(usize, &MoveData)> = moves.iter().enumerate().collect();
    pairs.sort_by(|a, b| a.1.sort_name.cmp(&b.1.sort_name).then(a.0.cmp(&b.0)));
    pairs.into_iter().map(|(idx, _)| idx).collect()
}

pub fn consistent_switch_order_indices(pokemon: &[PokemonData]) -> Vec<usize> {
    let mut pairs: Vec<(usize, &PokemonData)> = pokemon.iter().enumerate().collect();
    pairs.sort_by(|a, b| a.1.sort_name.cmp(&b.1.sort_name).then(a.0.cmp(&b.0)));
    pairs.into_iter().map(|(idx, _)| idx).collect()
}

pub fn active_effect_from_side(side: &Side) -> String {
    let mut effects: HashSet<String> = HashSet::new();

    for status in side.volatile_statuses.iter() {
        let normalized = clean_no_numbers(&status.to_string());
        if !normalized.is_empty() {
            effects.insert(normalized);
        }
    }

    let durations = &side.volatile_status_durations;
    let duration_fields = [
        ("confusion", durations.confusion),
        ("encore", durations.encore),
        ("lockedmove", durations.lockedmove),
        ("slowstart", durations.slowstart),
        ("taunt", durations.taunt),
        ("yawn", durations.yawn),
    ];
    for (name, value) in duration_fields {
        if value > 0 {
            effects.insert(clean_no_numbers(name));
        }
    }

    if side.substitute_health > 0 {
        effects.insert("substitute".to_string());
    }

    if side.force_trapped {
        effects.insert("partiallytrapped".to_string());
    }

    if side.future_sight.0 > 0 {
        effects.insert("futuresight".to_string());
    }

    if effects.is_empty() {
        return "noeffect".to_string();
    }

    // Prefer ability-driven boost effects with stat suffixes (matches Abra expectations), e.g.:
    // quarkdriveatk/def/spa/spd/spe, protosynthesisatk/.../spe
    if let Some(specific) = effects
        .iter()
        .find(|e| e.starts_with("quarkdrive"))
        .cloned()
    {
        return specific;
    }
    if let Some(specific) = effects
        .iter()
        .find(|e| e.starts_with("protosynthesis"))
        .cloned()
    {
        return specific;
    }

    let priority = [
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
    ];

    for pref in priority {
        if effects.contains(pref) {
            return pref.to_string();
        }
    }

    let mut sorted: Vec<String> = effects.into_iter().collect();
    sorted.sort();
    sorted[0].clone()
}

pub fn side_conditions_to_str(side: &Side) -> String {
    let sc = &side.side_conditions;
    let mapping = [
        ("aurora_veil", sc.aurora_veil),
        ("crafty_shield", sc.crafty_shield),
        ("healing_wish", sc.healing_wish),
        ("light_screen", sc.light_screen),
        ("lucky_chant", sc.lucky_chant),
        ("lunar_dance", sc.lunar_dance),
        ("mat_block", sc.mat_block),
        ("mist", sc.mist),
        ("protect", sc.protect),
        ("quick_guard", sc.quick_guard),
        ("reflect", sc.reflect),
        ("safeguard", sc.safeguard),
        ("spikes", sc.spikes),
        ("stealth_rock", sc.stealth_rock),
        ("sticky_web", sc.sticky_web),
        ("tailwind", sc.tailwind),
        ("toxic_count", sc.toxic_count),
        ("toxic_spikes", sc.toxic_spikes),
        ("wide_guard", sc.wide_guard),
    ];

    let mut active = HashMap::new();
    for (name, value) in mapping {
        if value > 0 {
            active.insert(clean_no_numbers(name), value);
        }
    }

    if active.is_empty() {
        return "noconditions".to_string();
    }

    if active.contains_key("reflect") && active.contains_key("lightscreen") {
        return "screens".to_string();
    }

    let mut items: Vec<(String, i8)> = active.into_iter().collect();
    items.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    items
        .into_iter()
        .next()
        .map(|(name, _)| name)
        .unwrap_or_else(|| "noconditions".to_string())
}

pub fn count_remaining(pokemon: &[Pokemon]) -> i32 {
    pokemon.iter().filter(|p| p.hp > 0).count() as i32
}

pub fn extract_teampreview(side: &Side) -> Vec<String> {
    let mut preview: Vec<String> = side
        .pokemon
        .pkmn
        .iter()
        .map(|pokemon| pokemon_name_str(&pokemon.id.to_string())) // NOTE: just cleaned IDs; does not mirror the dex base-species normalization from Python
        .collect();
    preview.sort();
    preview
}

pub fn build_move_mapping(order: &[usize]) -> BTreeMap<usize, usize> {
    let mut map = BTreeMap::new();
    for (consistent_idx, original_idx) in order.iter().enumerate() {
        map.insert(consistent_idx, *original_idx);
    }
    map
}

pub fn prepare_state_data(
    py: Python<'_>,
    state: &State,
    perspective: Perspective,
    battle_format: &str,
) -> PyResult<PreparedStateData> {
    let t_start = Instant::now();
    
    // select player/opponent sides
    let (player_side, opponent_side) = match perspective {
        Perspective::SideOne => (&state.side_one, &state.side_two),
        Perspective::SideTwo => (&state.side_two, &state.side_one),
    };

    // active pokemon indices
    let active_index = pokemon_index_to_usize(player_side.active_index);
    let opponent_active_index = pokemon_index_to_usize(opponent_side.active_index);

    let player_pokemon_slice: Vec<Pokemon> = player_side
        .pokemon
        .pkmn
        .iter()
        .cloned()
        .collect();
    let opponent_pokemon_slice: Vec<Pokemon> = opponent_side
        .pokemon
        .pkmn
        .iter()
        .cloned()
        .collect();
    
    let t_after_setup = Instant::now();

    if active_index >= player_pokemon_slice.len() {
        return Err(PyValueError::new_err("active_index out of range for player side"));
    }
    if opponent_active_index >= opponent_pokemon_slice.len() {
        return Err(PyValueError::new_err(
            "active_index out of range for opponent side",
        ));
    }

    let active_pokemon = &player_pokemon_slice[active_index];
    let opponent_pokemon = &opponent_pokemon_slice[opponent_active_index];

    // Build move data for active pokemon
    let active_moves_raw = pokemon_moves_to_vec(&active_pokemon.moves);
    let mut active_moves: Vec<MoveData> = Vec::with_capacity(active_moves_raw.len());
    for mv in active_moves_raw.iter() {
        let data = build_move_data(&mv.choice, mv);
        active_moves.push(data);
    }

    // bench pokemon data
    let mut bench_data: Vec<PokemonData> = Vec::new();
    for (idx, pokemon) in player_pokemon_slice.iter().enumerate() {
        if idx == active_index || pokemon.hp <= 0 {
            continue;
        }

        let tera_type = tera_type(pokemon);

        let moves_vec = pokemon_moves_to_vec(&pokemon.moves);
        let mut bench_moves = Vec::with_capacity(moves_vec.len());
        for mv in moves_vec.iter() {
            bench_moves.push(build_move_data(&mv.choice, mv));
        }

        let base_stats = base_stats(pokemon);

        let name = pokemon_name_str(&pokemon.id.to_string());
        let sort_name = clean_name(&pokemon.id.to_string());
        let base_species = name.clone();
        let hp_pct = if pokemon.maxhp > 0 {
            (pokemon.hp.max(0) as f32 / pokemon.maxhp as f32).clamp(0.0, 1.0)
        } else {
            0.0
        };

        bench_data.push(PokemonData {
            name,
            sort_name,
            base_species,
            hp_pct,
            types_display: pokemon_types_display(pokemon),
            item: normalize_item(pokemon.item),
            ability: normalize_ability(pokemon.ability),
            level: pokemon.level,
            status: normalize_status(pokemon.status),
            effect: "noeffect".to_string(), // NOTE: bench mons ignore side effects; python did the same but it's still an assumption
            tera_type,
            base_stats,
            moves: bench_moves,
        });
    }

    let t_after_bench = Instant::now();
    
    let switch_order = consistent_switch_order_indices(&bench_data);
    let switch_mapping = build_move_mapping(&switch_order);

    let move_order = consistent_move_order_indices(&active_moves);
    let move_mapping = build_move_mapping(&move_order);

    let can_tera = !player_pokemon_slice.iter().any(|p| p.terastallized); // NOTE: assumes tera is available iff no one on the side has terastallized yet
    let forced_switch = player_side.force_switch;

    // Legal actions
    let mut legal_actions: Vec<usize> = Vec::new(); // NOTE: purely index-based; does not validate against actual battle legality beyond force-switch/tera heuristics
    let moves_len = active_moves.len();
    if !forced_switch {
        for i in 0..moves_len {
            legal_actions.push(i);
        }
        if can_tera {
            for i in 0..moves_len {
                legal_actions.push(9 + i);
            }
        }
    }
    let switch_len = bench_data.len();
    for i in 0..switch_len {
        legal_actions.push(4 + i);
    }
    legal_actions.sort_unstable();
    
    let t_after_mappings = Instant::now();

    // Helper to create move dict from MoveData (no Python object creation)
    let build_move_dict = |move_data: &MoveData| -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        dict.set_item("name", &move_data.name)?;
        dict.set_item("move_type", &move_data.move_type)?;
        dict.set_item("category", &move_data.category)?;
        dict.set_item("base_power", move_data.base_power as i32)?;
        dict.set_item("accuracy", move_data.accuracy)?;
        dict.set_item("priority", move_data.priority)?;
        dict.set_item("current_pp", move_data.current_pp)?;
        dict.set_item("max_pp", move_data.max_pp)?;
        Ok(dict.into())
    };

    let t_after_py_imports = Instant::now();
    
    let active_moves_py: Vec<PyObject> = active_moves
        .iter()
        .map(|m| build_move_dict(m))
        .collect::<PyResult<Vec<_>>>()?;
    
    let t_after_active_moves = Instant::now();

    // Build dicts for bench pokemon (no Python object creation)
    let mut bench_py_list: Vec<PyObject> = Vec::new();
    for data in bench_data.iter() {
        let bench_moves_py: Vec<PyObject> = data
            .moves
            .iter()
            .map(|m| build_move_dict(m))
            .collect::<PyResult<Vec<_>>>()?;

        let dict = PyDict::new(py);
        dict.set_item("name", &data.name)?;
        dict.set_item("base_species", &data.base_species)?;
        dict.set_item("hp_pct", data.hp_pct)?;
        dict.set_item("types", &data.types_display)?;
        dict.set_item("item", &data.item)?;
        dict.set_item("ability", &data.ability)?;
        dict.set_item("lvl", data.level)?;
        dict.set_item("status", &data.status)?;
        dict.set_item("effect", &data.effect)?;
        dict.set_item("moves", PyList::new(py, bench_moves_py))?;
        dict.set_item("atk_boost", 0)?;
        dict.set_item("spa_boost", 0)?;
        dict.set_item("def_boost", 0)?;
        dict.set_item("spd_boost", 0)?;
        dict.set_item("spe_boost", 0)?;
        dict.set_item("accuracy_boost", 0)?;
        dict.set_item("evasion_boost", 0)?;
        dict.set_item("base_atk", data.base_stats.1)?;
        dict.set_item("base_spa", data.base_stats.3)?;
        dict.set_item("base_def", data.base_stats.2)?;
        dict.set_item("base_spd", data.base_stats.4)?;
        dict.set_item("base_spe", data.base_stats.5)?;
        dict.set_item("base_hp", data.base_stats.0)?;
        dict.set_item("tera_type", &data.tera_type)?;
        bench_py_list.push(dict.into());
    }
    
    let t_after_bench_py = Instant::now();

    let (active_base_hp, active_base_atk, active_base_def, active_base_spa, active_base_spd, active_base_spe) =
        base_stats(active_pokemon);
    let active_boosts = (
        player_side.attack_boost,
        player_side.special_attack_boost,
        player_side.defense_boost,
        player_side.special_defense_boost,
        player_side.speed_boost,
        player_side.accuracy_boost,
        player_side.evasion_boost,
    );

    let active_tera_type = tera_type(active_pokemon);
    let active_effect = active_effect_from_side(player_side);

    let active_dict = PyDict::new(py);
    active_dict.set_item("name", pokemon_name_str(&active_pokemon.id.to_string()))?;
    active_dict.set_item("base_species", pokemon_name_str(&active_pokemon.id.to_string()))?;
    let hp_pct = if active_pokemon.maxhp > 0 {
        (active_pokemon.hp.max(0) as f32 / active_pokemon.maxhp as f32).clamp(0.0, 1.0)
    } else {
        0.0
    };
    active_dict.set_item("hp_pct", hp_pct)?;
    active_dict.set_item("types", pokemon_types_display(active_pokemon))?;
    active_dict.set_item("item", normalize_item(active_pokemon.item))?;
    active_dict.set_item("ability", normalize_ability(active_pokemon.ability))?;
    active_dict.set_item("lvl", active_pokemon.level)?;
    active_dict.set_item("status", normalize_status(active_pokemon.status))?;
    active_dict.set_item("effect", active_effect)?;
    active_dict.set_item("moves", PyList::new(py, &active_moves_py))?;
    active_dict.set_item("atk_boost", active_boosts.0)?;
    active_dict.set_item("spa_boost", active_boosts.1)?;
    active_dict.set_item("def_boost", active_boosts.2)?;
    active_dict.set_item("spd_boost", active_boosts.3)?;
    active_dict.set_item("spe_boost", active_boosts.4)?;
    active_dict.set_item("accuracy_boost", active_boosts.5)?;
    active_dict.set_item("evasion_boost", active_boosts.6)?;
    active_dict.set_item("base_atk", active_base_atk)?;
    active_dict.set_item("base_spa", active_base_spa)?;
    active_dict.set_item("base_def", active_base_def)?;
    active_dict.set_item("base_spd", active_base_spd)?;
    active_dict.set_item("base_spe", active_base_spe)?;
    active_dict.set_item("base_hp", active_base_hp)?;
    active_dict.set_item("tera_type", active_tera_type)?;

    // Opponent active dict
    let opponent_moves_raw = pokemon_moves_to_vec(&opponent_pokemon.moves);
    let opponent_moves_py: Vec<PyObject> = opponent_moves_raw
        .iter()
        .map(|mv| build_move_dict(&build_move_data(&mv.choice, mv)))
        .collect::<PyResult<Vec<_>>>()?;

    let opponent_dict = PyDict::new(py);
    opponent_dict.set_item(
        "name",
        pokemon_name_str(&opponent_pokemon.id.to_string()),
    )?;
    opponent_dict.set_item(
        "base_species",
        pokemon_name_str(&opponent_pokemon.id.to_string()),
    )?;
    let opponent_hp_pct = if opponent_pokemon.maxhp > 0 {
        (opponent_pokemon.hp.max(0) as f32 / opponent_pokemon.maxhp as f32).clamp(0.0, 1.0)
    } else {
        0.0
    };
    opponent_dict.set_item("hp_pct", opponent_hp_pct)?;
    opponent_dict.set_item("types", pokemon_types_display(opponent_pokemon))?;
    opponent_dict.set_item("item", normalize_item(opponent_pokemon.item))?;
    opponent_dict.set_item("ability", normalize_ability(opponent_pokemon.ability))?;
    opponent_dict.set_item("lvl", opponent_pokemon.level)?;
    opponent_dict.set_item("status", normalize_status(opponent_pokemon.status))?;
    opponent_dict.set_item("effect", active_effect_from_side(opponent_side))?;
    opponent_dict.set_item("moves", PyList::new(py, &opponent_moves_py))?;
    opponent_dict.set_item("atk_boost", opponent_side.attack_boost)?;
    opponent_dict.set_item("spa_boost", opponent_side.special_attack_boost)?;
    opponent_dict.set_item("def_boost", opponent_side.defense_boost)?;
    opponent_dict.set_item("spd_boost", opponent_side.special_defense_boost)?;
    opponent_dict.set_item("spe_boost", opponent_side.speed_boost)?;
    opponent_dict.set_item("accuracy_boost", opponent_side.accuracy_boost)?;
    opponent_dict.set_item("evasion_boost", opponent_side.evasion_boost)?;
    let (opponent_base_hp, opponent_base_atk, opponent_base_def, opponent_base_spa, opponent_base_spd, opponent_base_spe) =
        base_stats(opponent_pokemon);
    opponent_dict.set_item("base_atk", opponent_base_atk)?;
    opponent_dict.set_item("base_spa", opponent_base_spa)?;
    opponent_dict.set_item("base_def", opponent_base_def)?;
    opponent_dict.set_item("base_spd", opponent_base_spd)?;
    opponent_dict.set_item("base_spe", opponent_base_spe)?;
    opponent_dict.set_item("base_hp", opponent_base_hp)?;
    opponent_dict.set_item("tera_type", tera_type(opponent_pokemon))?;

    // Previous moves (simplified)
    let blank_move_dict = PyDict::new(py);
    blank_move_dict.set_item("name", "nomove")?;
    blank_move_dict.set_item("move_type", "nomove")?;
    blank_move_dict.set_item("category", "nomove")?;
    blank_move_dict.set_item("base_power", 0)?;
    blank_move_dict.set_item("accuracy", 1.0)?;
    blank_move_dict.set_item("priority", 0)?;
    blank_move_dict.set_item("current_pp", 0)?;
    blank_move_dict.set_item("max_pp", 0)?; // NOTE: we always emit "nomove" placeholders; last_used_move data is ignored for now

    let state_dict = PyDict::new(py);
    state_dict.set_item("format", battle_format.to_lowercase())?;
    state_dict.set_item("player_active_pokemon", active_dict)?;
    state_dict.set_item("opponent_active_pokemon", opponent_dict)?;
    state_dict.set_item("available_switches", PyList::new(py, &bench_py_list))?;
    state_dict.set_item("player_prev_move", blank_move_dict)?;
    state_dict.set_item("opponent_prev_move", blank_move_dict)?;
    state_dict.set_item("opponents_remaining", count_remaining(&opponent_pokemon_slice))?;
    state_dict.set_item("player_conditions", side_conditions_to_str(player_side))?;
    state_dict.set_item("opponent_conditions", side_conditions_to_str(opponent_side))?;
    let weather = {
        let raw = state.weather.weather_type.to_string();
        let normalized = clean_no_numbers(&raw); // NOTE: mirrors python's simplification but still ignores duration/turns remaining
        if normalized.is_empty() || normalized == "none" {
            "noweather".to_string()
        } else {
            normalized
        }
    };
    state_dict.set_item("weather", weather)?;
    let terrain = {
        let raw = state.terrain.terrain_type.to_string();
        let normalized = clean_no_numbers(&raw); // NOTE: same assumption: only type string is surfaced, no turn counts
        if normalized.is_empty() || normalized == "none" {
            "nofield".to_string()
        } else {
            normalized
        }
    };
    state_dict.set_item("battle_field", terrain)?;
    state_dict.set_item("forced_switch", forced_switch)?;
    let player_alive = player_pokemon_slice.iter().any(|p| p.hp > 0);
    let opponent_alive = opponent_pokemon_slice.iter().any(|p| p.hp > 0);
    state_dict.set_item("battle_won", player_alive && !opponent_alive)?;
    state_dict.set_item("battle_lost", opponent_alive && !player_alive)?;
    state_dict.set_item("can_tera", can_tera)?;
    state_dict.set_item("opponent_teampreview", PyList::new(py, extract_teampreview(opponent_side)))?;
    
    let t_end = Instant::now();
    
    // Print timing breakdown
    let setup_ms = (t_after_setup - t_start).as_secs_f64() * 1000.0;
    let bench_ms = (t_after_bench - t_after_setup).as_secs_f64() * 1000.0;
    let mappings_ms = (t_after_mappings - t_after_bench).as_secs_f64() * 1000.0;
    let py_dict_setup_ms = (t_after_py_imports - t_after_mappings).as_secs_f64() * 1000.0;
    let active_moves_ms = (t_after_active_moves - t_after_py_imports).as_secs_f64() * 1000.0;
    let bench_py_ms = (t_after_bench_py - t_after_active_moves).as_secs_f64() * 1000.0;
    let final_ms = (t_end - t_after_bench_py).as_secs_f64() * 1000.0;
    let total_ms = (t_end - t_start).as_secs_f64() * 1000.0;
    
    println!(
        "[RUST_BREAKDOWN] setup={:.2}ms bench_data={:.2}ms mappings={:.2}ms dict_setup={:.2}ms active_moves_dicts={:.2}ms bench_dicts={:.2}ms final_dict={:.2}ms total={:.2}ms",
        setup_ms, bench_ms, mappings_ms, py_dict_setup_ms, active_moves_ms, bench_py_ms, final_ms, total_ms
    );

    Ok(PreparedStateData {
        universal_state: state_dict.into(),
        legal_actions,
        move_mapping,
        switch_mapping,
    })
}

#[pyfunction(name = "_prepare_state_fast")]
pub fn py_prepare_state_fast(
    py: Python<'_>,
    py_state: &PyState,
    perspective: &str,
    battle_format: &str,
) -> PyResult<PyObject> {
    let rust_state: State = py_state.clone().into();
    let perspective_enum = Perspective::from_str(perspective)?;
    let prepared = prepare_state_data(py, &rust_state, perspective_enum, battle_format)?;
    let result_dict = PyDict::new(py);
    result_dict.set_item("universal_state", prepared.universal_state)?;
    result_dict.set_item("legal_actions", PyList::new(py, prepared.legal_actions))?;

    let move_mapping = PyDict::new(py);
    for (k, v) in prepared.move_mapping {
        move_mapping.set_item(k, v)?;
    }
    result_dict.set_item("move_mapping", move_mapping)?;

    let switch_mapping = PyDict::new(py);
    for (k, v) in prepared.switch_mapping {
        switch_mapping.set_item(k, v)?;
    }
    result_dict.set_item("switch_mapping", switch_mapping)?;

    Ok(result_dict.into())
}
