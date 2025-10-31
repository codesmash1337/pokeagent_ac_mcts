use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::{Mutex, OnceLock};
use serde::Deserialize;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use numpy::{PyArray1, PyArray2};
use rayon::prelude::*;

use poke_engine::choices::Choices;
use poke_engine::state::{LastUsedMove, Pokemon, PokemonMoveIndex, PokemonStatus, Side, State};
use crate::universal::{
    pokemon_name_str, clean_no_numbers, normalize_item, normalize_ability, normalize_status,
    pokemon_types_display, tera_type, active_effect_from_side, active_effect_from_side_with_pokemon,
    side_conditions_to_str, pokemon_index_to_usize, pokemon_moves_to_vec,
    build_move_data, base_stats, count_remaining, extract_teampreview, MoveData,
};

const UNKNOWN_TOKEN: i32 = -1;
static TOKENIZER: OnceLock<Tokenizer> = OnceLock::new();
static OBSERVATION_TRACKERS: OnceLock<Mutex<HashMap<String, BattleTracker>>> = OnceLock::new();

fn observation_trackers() -> &'static Mutex<HashMap<String, BattleTracker>> {
    OBSERVATION_TRACKERS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Clear all observation trackers (useful for testing/benchmarking)
pub fn clear_observation_trackers() {
    let mut trackers = observation_trackers().lock().unwrap();
    trackers.clear();
}

#[derive(Default, Debug)]
struct BattleTracker {
    revealed_opponents: BTreeSet<String>,
    any_opponent_asleep: bool,
    any_opponent_frozen: bool,
    steps_seen: u64,
    /// Map effect name -> turn number when first seen (lower = more recent)
    player_effects_seen: BTreeMap<String, u64>,
    opponent_effects_seen: BTreeMap<String, u64>,
}

impl BattleTracker {
    fn reset(&mut self) {
        self.revealed_opponents.clear();
        self.any_opponent_asleep = false;
        self.any_opponent_frozen = false;
        self.steps_seen = 0;
        self.player_effects_seen.clear();
        self.opponent_effects_seen.clear();
    }
}

#[derive(Debug)]
struct ObservationContext {
    revealed_opponents: Vec<String>,
    player_prev_move: String,
    opponent_prev_move: String,
    any_opponent_asleep: bool,
    any_opponent_frozen: bool,
    can_tera: bool,
    player_effects_seen: BTreeMap<String, u64>,
    opponent_effects_seen: BTreeMap<String, u64>,
}

#[derive(Debug)]
pub struct InferencePayload {
    pub text_tokens: Vec<i32>,
    pub numbers: Vec<f32>,
    pub legal_actions: Vec<usize>,
    pub move_mapping: BTreeMap<usize, usize>,
    pub switch_mapping: BTreeMap<usize, usize>,
}

#[derive(Debug)]
pub struct BatchPayload {
    pub text_tokens: Vec<i32>,
    pub numbers: Vec<f32>,
    pub legal_actions: Vec<Vec<usize>>,
    pub move_mappings: Vec<BTreeMap<usize, usize>>,
    pub switch_mappings: Vec<BTreeMap<usize, usize>>,
    pub token_len: usize,
    pub number_len: usize,
}

#[derive(Debug, Deserialize)]
struct Tokenizer {
    vocab: HashMap<String, i32>,
}

impl Tokenizer {
    fn load(json_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let json_str = std::fs::read_to_string(json_path)?;
        let vocab: HashMap<String, i32> = serde_json::from_str(&json_str)?;
        Ok(Tokenizer { vocab })
    }

    fn tokenize(&self, text: &str) -> Vec<i32> {
        text.split_whitespace()
            .map(|word| *self.vocab.get(word).unwrap_or(&UNKNOWN_TOKEN))
            .collect()
    }
}

pub fn get_tokenizer() -> &'static Tokenizer {
    TOKENIZER.get_or_init(|| {
        // Use DefaultObservationSpace-v1 tokenizer to match pretrained models
        Tokenizer::load("vendor/metamon/metamon/tokenizer/DefaultObservationSpace-v1.json")
            .expect("Failed to load tokenizer")
    })
}

fn move_name(move_name: &str) -> String {
    clean_no_numbers(move_name)
}

fn move_index_to_usize(index: PokemonMoveIndex) -> usize {
    match index {
        PokemonMoveIndex::M0 => 0,
        PokemonMoveIndex::M1 => 1,
        PokemonMoveIndex::M2 => 2,
        PokemonMoveIndex::M3 => 3,
    }
}

fn move_ref<'a>(pokemon: &'a Pokemon, index: PokemonMoveIndex) -> &'a poke_engine::state::Move {
    match index {
        PokemonMoveIndex::M0 => &pokemon.moves.m0,
        PokemonMoveIndex::M1 => &pokemon.moves.m1,
        PokemonMoveIndex::M2 => &pokemon.moves.m2,
        PokemonMoveIndex::M3 => &pokemon.moves.m3,
    }
}

fn sorted_active_moves(pokemon: &Pokemon) -> Vec<(PokemonMoveIndex, MoveData)> {
    let mut entries = vec![
        (PokemonMoveIndex::M0, build_move_data(&pokemon.moves.m0.choice, &pokemon.moves.m0)),
        (PokemonMoveIndex::M1, build_move_data(&pokemon.moves.m1.choice, &pokemon.moves.m1)),
        (PokemonMoveIndex::M2, build_move_data(&pokemon.moves.m2.choice, &pokemon.moves.m2)),
        (PokemonMoveIndex::M3, build_move_data(&pokemon.moves.m3.choice, &pokemon.moves.m3)),
    ];

    entries.sort_by(|a, b| {
        let sort_cmp = a.1.sort_name.cmp(&b.1.sort_name);
        if sort_cmp == std::cmp::Ordering::Equal {
            move_index_to_usize(a.0).cmp(&move_index_to_usize(b.0))
        } else {
            sort_cmp
        }
    });

    entries
}

fn build_move_mapping_from_sorted(sorted_moves: &[(PokemonMoveIndex, MoveData)]) -> BTreeMap<usize, usize> {
    let mut mapping = BTreeMap::new();
    for (idx, (original_index, _)) in sorted_moves.iter().enumerate() {
        mapping.insert(idx, move_index_to_usize(*original_index));
    }
    mapping
}

fn build_switch_mapping(sorted_switches: &[(usize, String)]) -> BTreeMap<usize, usize> {
    let mut mapping = BTreeMap::new();
    for (idx, (original_index, _)) in sorted_switches.iter().enumerate() {
        mapping.insert(idx, *original_index);
    }
    mapping
}

/// Build the observation text string in the exact format expected by TeamPreviewObservationSpace
fn build_observation_text(
    state: &State,
    player_side: &Side,
    opponent_side: &Side,
    player_pokemon: &[Pokemon],
    opponent_pokemon: &[Pokemon],
    battle_format: &str,
    context: &ObservationContext,
) -> String {
    let active_index = pokemon_index_to_usize(player_side.active_index);
    let opponent_active_index = pokemon_index_to_usize(opponent_side.active_index);
    
    let active_pokemon = &player_pokemon[active_index];
    let opponent_pokemon_active = &opponent_pokemon[opponent_active_index];
    
    let mut parts = Vec::new();
    
    // Format tag + force switch
    parts.push(format!("<{}>", battle_format.to_lowercase()));
    let force_switch = if player_side.force_switch { "<forcedswitch>" } else { "<anychoice>" };
    parts.push(force_switch.to_string());
    
    // Player active pokemon
    parts.push("<player>".to_string());
    parts.push(pokemon_name_str(&active_pokemon.id.to_string()));
    parts.push(normalize_item(active_pokemon.item));
    parts.push(normalize_ability(active_pokemon.ability));
    parts.push(pokemon_types_display(active_pokemon));
    parts.push(select_most_recent_effect(player_side, Some(active_pokemon), &context.player_effects_seen));
    parts.push(normalize_status(active_pokemon.status));
    parts.push(tera_type(active_pokemon));
    
    // Player active moves (sorted)
    let sorted_active_moves = sorted_active_moves(active_pokemon);
    
    for (_, move_data) in &sorted_active_moves {
        parts.push("<move>".to_string());
        parts.push(move_data.name.clone());
        parts.push(move_data.move_type.clone());
        parts.push(move_data.category.clone());
    }
    
    // Pad to 4 moves
    for _ in sorted_active_moves.len()..4 {
        parts.push("<move>".to_string());
        parts.push("<blank>".to_string());
        parts.push("<blank>".to_string());
        parts.push("<blank>".to_string());
    }
    
    // Player switches (sorted: alive first alphabetically, then fainted alphabetically)
    let mut switches: Vec<_> = player_pokemon
        .iter()
        .enumerate()
        .filter(|(idx, _p)| *idx != active_index)  // Include ALL non-active pokemon
        .collect();
    switches.sort_by(|(idx_a, p_a), (idx_b, p_b)| {
        let alive_a = p_a.hp > 0;
        let alive_b = p_b.hp > 0;
        
        // Sort by alive status first (alive = true comes before fainted = false)
        match alive_b.cmp(&alive_a) {
            std::cmp::Ordering::Equal => {
                // Within same alive status, sort alphabetically by name
                let name_a = pokemon_name_str(&p_a.id.to_string());
                let name_b = pokemon_name_str(&p_b.id.to_string());
                let cmp = name_a.cmp(&name_b);
                if cmp == std::cmp::Ordering::Equal {
                    idx_a.cmp(idx_b)
                } else {
                    cmp
                }
            }
            other => other,
        }
    });
    
    for (_, pokemon) in &switches {
        parts.push("<switch>".to_string());
        
        // Fainted pokemon: output 9 <blank> tokens (name, item, ability, <moveset>, 4 moves, tera_type)
        if pokemon.hp <= 0 {
            for _ in 0..9 {
                parts.push("<blank>".to_string());
            }
        } else {
            // Alive pokemon: output actual data
            parts.push(pokemon_name_str(&pokemon.id.to_string()));
            parts.push(normalize_item(pokemon.item));
            parts.push(normalize_ability(pokemon.ability));
            parts.push("<moveset>".to_string());
            
            let switch_moves_raw = pokemon_moves_to_vec(&pokemon.moves);
            let mut switch_moves: Vec<_> = switch_moves_raw
                .iter()
                .map(|mv| move_name(&build_move_data(&mv.choice, mv).name))
                .collect();
            switch_moves.sort_unstable();
            for move_name in &switch_moves {
                parts.push(move_name.clone());
            }
            for _ in switch_moves.len()..4 {
                parts.push("<blank>".to_string());
            }
            parts.push(tera_type(pokemon));
        }
    }
    
    // Pad to 5 switches
    for _ in switches.len()..5 {
        parts.push("<switch>".to_string());
        for _ in 0..9 {
            parts.push("<blank>".to_string());
        }
    }
    
    // Opponent active
    parts.push("<opponent>".to_string());
    parts.push(pokemon_name_str(&opponent_pokemon_active.id.to_string()));
    parts.push(normalize_item(opponent_pokemon_active.item));
    parts.push(normalize_ability(opponent_pokemon_active.ability));
    parts.push(pokemon_types_display(opponent_pokemon_active));
    parts.push(select_most_recent_effect(opponent_side, Some(opponent_pokemon_active), &context.opponent_effects_seen));
    parts.push(normalize_status(opponent_pokemon_active.status));
    parts.push(tera_type(opponent_pokemon_active));
    
    // Conditions
    parts.push("<conditions>".to_string());
    let weather = {
        let raw = state.weather.weather_type.to_string();
        let normalized = clean_no_numbers(&raw);
        if normalized.is_empty() || normalized == "none" {
            "noweather".to_string()
        } else {
            normalized
        }
    };
    parts.push(weather);
    parts.push(side_conditions_to_str(player_side));
    parts.push(side_conditions_to_str(opponent_side));
    
    // Previous moves
    parts.push("<player_prev>".to_string());
    parts.push(context.player_prev_move.clone());
    parts.push("<opp_prev>".to_string());
    parts.push(context.opponent_prev_move.clone());
    
    // Revealed opponents (persistent across battle)
    let mut revealed_added = 0usize;
    for name in context.revealed_opponents.iter().take(6) {
        parts.push(name.clone());
        revealed_added += 1;
    }
    for _ in revealed_added..6 {
        parts.push("<blank>".to_string());
    }
    
    // Team preview
    let mut teampreview = extract_teampreview(opponent_side);
    teampreview.sort_unstable();
    for name in &teampreview {
        parts.push(name.clone());
    }
    for _ in teampreview.len()..6 {
        parts.push("<blank>".to_string());
    }
    let mut teampreview_names: BTreeSet<String> = extract_teampreview(opponent_side)
    .into_iter()
    .collect();
    // *NOTE: This is a proposed method that also does not work in my testing
    // for revealed in &context.revealed_opponents {
    //     teampreview_names.insert(revealed.clone());
    // }
    // let mut teampreview: Vec<String> = teampreview_names.into_iter().collect();
    // while teampreview.len() < 6 {
    //     teampreview.push("<blank>".to_string());
    // }
    // for name in teampreview.iter().take(6) {
    //     parts.push(name.clone());
    // }
    parts.join(" ")
}

/// Build the observation numbers array (55 floats)
fn build_observation_numbers(
    player_side: &Side,
    opponent_side: &Side,
    player_pokemon: &[Pokemon],
    opponent_pokemon: &[Pokemon],
    context: &ObservationContext,
) -> Vec<f32> {
    let active_index = pokemon_index_to_usize(player_side.active_index);
    let opponent_active_index = pokemon_index_to_usize(opponent_side.active_index);
    
    let active_pokemon = &player_pokemon[active_index];
    let opponent_pokemon_active = &opponent_pokemon[opponent_active_index];
    
    let mut numbers = Vec::new();
    
    // Opponents remaining
    numbers.push(count_remaining(&opponent_pokemon) as f32 / 6.0);
    
    // Player active pokemon numerical features
    let hp_pct = if active_pokemon.maxhp > 0 {
        (active_pokemon.hp.max(0) as f32 / active_pokemon.maxhp as f32).clamp(0.0, 1.0)
    } else {
        0.0
    };
    numbers.push(hp_pct);
    numbers.push(active_pokemon.level as f32 / 100.0);
    
    let (base_hp, base_atk, base_def, base_spa, base_spd, base_spe) = base_stats(active_pokemon);
    numbers.push(base_atk as f32 / 255.0);
    numbers.push(base_spa as f32 / 255.0);
    numbers.push(base_def as f32 / 255.0);
    numbers.push(base_spd as f32 / 255.0);
    numbers.push(base_spe as f32 / 255.0);
    numbers.push(base_hp as f32 / 255.0);
    
    numbers.push(player_side.attack_boost as f32 / 6.0);
    numbers.push(player_side.special_attack_boost as f32 / 6.0);
    numbers.push(player_side.defense_boost as f32 / 6.0);
    numbers.push(player_side.special_defense_boost as f32 / 6.0);
    numbers.push(player_side.speed_boost as f32 / 6.0);
    numbers.push(player_side.accuracy_boost as f32 / 6.0);
    numbers.push(player_side.evasion_boost as f32 / 6.0);
    
    // Player active moves numerical features (4 moves × 4 features each = 16)
    let sorted_moves = sorted_active_moves(active_pokemon);
    for (_, move_data) in &sorted_moves {
        numbers.push(move_data.base_power as f32 / 200.0);
        numbers.push(move_data.accuracy);
        numbers.push(move_data.priority as f32 / 5.0);
        let max_pp = move_data.max_pp.max(1) as f32;
        let current_pp = move_data.current_pp.max(0) as f32;
        let pp_ratio = (current_pp / max_pp).clamp(0.0, 1.0);
        let pp_warning = ((pp_ratio >= 0.5) as i32
            + (pp_ratio >= 0.25) as i32
            + (pp_ratio > 0.0) as i32) as f32;
        numbers.push(pp_warning);
    }
    for _ in sorted_moves.len()..4 {
        numbers.push(-2.0);
        numbers.push(-2.0);
        numbers.push(-2.0);
        numbers.push(-2.0);
    }
    
    // Player switches numerical features (5 switches × 1 feature = 5)
    // Must use same sorting as text: alive first (alphabetically), then fainted (alphabetically)
    let mut switches: Vec<_> = player_pokemon.iter().enumerate()
        .filter(|(idx, _p)| *idx != active_index)  // Include ALL non-active pokemon
        .collect();
    switches.sort_by(|(idx_a, p_a), (idx_b, p_b)| {
        let alive_a = p_a.hp > 0;
        let alive_b = p_b.hp > 0;
        
        // Sort by alive status first (alive = true comes before fainted = false)
        match alive_b.cmp(&alive_a) {
            std::cmp::Ordering::Equal => {
                // Within same alive status, sort alphabetically by name
                let name_a = pokemon_name_str(&p_a.id.to_string());
                let name_b = pokemon_name_str(&p_b.id.to_string());
                let cmp = name_a.cmp(&name_b);
                if cmp == std::cmp::Ordering::Equal {
                    idx_a.cmp(idx_b)
                } else {
                    cmp
                }
            }
            other => other,
        }
    });
    
    for (_, pokemon) in &switches {
        // Fainted pokemon use -2.0 sentinel value (matching Python's _get_pokemon_pad_numerical)
        let switch_hp_pct = if pokemon.hp <= 0 {
            -2.0
        } else if pokemon.maxhp > 0 {
            (pokemon.hp.max(0) as f32 / pokemon.maxhp as f32).clamp(0.0, 1.0)
        } else {
            0.0
        };
        numbers.push(switch_hp_pct);
    }
    for _ in switches.len()..5 {
        numbers.push(-2.0);
    }
    
    // Opponent active pokemon numerical features
    let opp_hp_pct = if opponent_pokemon_active.maxhp > 0 {
        (opponent_pokemon_active.hp.max(0) as f32 / opponent_pokemon_active.maxhp as f32).clamp(0.0, 1.0)
    } else {
        0.0
    };
    numbers.push(opp_hp_pct);
    numbers.push(opponent_pokemon_active.level as f32 / 100.0);
    
    let (opp_base_hp, opp_base_atk, opp_base_def, opp_base_spa, opp_base_spd, opp_base_spe) =
        base_stats(opponent_pokemon_active);
    numbers.push(opp_base_atk as f32 / 255.0);
    numbers.push(opp_base_spa as f32 / 255.0);
    numbers.push(opp_base_def as f32 / 255.0);
    numbers.push(opp_base_spd as f32 / 255.0);
    numbers.push(opp_base_spe as f32 / 255.0);
    numbers.push(opp_base_hp as f32 / 255.0);
    
    numbers.push(opponent_side.attack_boost as f32 / 6.0);
    numbers.push(opponent_side.special_attack_boost as f32 / 6.0);
    numbers.push(opponent_side.defense_boost as f32 / 6.0);
    numbers.push(opponent_side.special_defense_boost as f32 / 6.0);
    numbers.push(opponent_side.speed_boost as f32 / 6.0);
    numbers.push(opponent_side.accuracy_boost as f32 / 6.0);
    numbers.push(opponent_side.evasion_boost as f32 / 6.0);
    
    // Additional ExpandedObservationSpace features
    // For now, we'll use false for sleep/freeze since we don't have state tracking
    numbers.push(if context.any_opponent_asleep { 1.0 } else { 0.0 });
    numbers.push(if context.any_opponent_frozen { 1.0 } else { 0.0 });
    numbers.push(if context.can_tera { 1.0 } else { 0.0 });
    
    numbers
}

fn last_used_move_token(side: &Side, active_pokemon: &Pokemon) -> String {
    match side.last_used_move {
        LastUsedMove::Move(index) => {
            let move_struct = move_ref(active_pokemon, index);

            if move_struct.id == Choices::NONE {
                return "nomove".to_string();
            }

            let sorted_moves = sorted_active_moves(active_pokemon);
            if let Some((_, data)) = sorted_moves.into_iter().find(|(idx, _)| *idx == index) {
                if data.name.is_empty() {
                    "nomove".to_string()
                } else {
                    data.name
                }
            } else {
                "nomove".to_string()
            }
        }
        _ => "nomove".to_string(),
    }
}

fn build_side_signature(pokemon: &[Pokemon]) -> String {
    // Build a stable signature that doesn't change during battle
    // Use species name, level, and maxhp (which are stable)
    // Don't use item (can be consumed) or current HP (changes every turn)
    let mut entries: Vec<String> = pokemon
        .iter()
        .map(|p| {
            format!(
                "{}-{}-{}",
                pokemon_name_str(&p.id.to_string()),
                p.level,
                p.maxhp
            )
        })
        .collect();
    entries.sort();
    entries.join("|")
}

fn build_battle_key(
    player_pokemon: &[Pokemon],
    opponent_pokemon: &[Pokemon],
    battle_format: &str,
    perspective: &str,
) -> String {
    format!(
        "{}|{}|{}|{}",
        battle_format.to_lowercase(),
        perspective.to_lowercase(),
        build_side_signature(player_pokemon),
        build_side_signature(opponent_pokemon)
    )
}

fn team_fully_reset(pokemon: &[Pokemon]) -> bool {
    pokemon.iter().all(|p| {
        p.hp >= p.maxhp && matches!(p.status, PokemonStatus::NONE) && !p.terastallized
    })
}

fn should_reset_tracker(tracker: &BattleTracker, player: &[Pokemon], opponent: &[Pokemon]) -> bool {
    if tracker.steps_seen == 0 {
        return false;
    }
    // Only reset if BOTH teams are fully reset (indicates a new battle)
    // Don't reset on terminal states (all fainted) - those occur during MCTS simulation
    // and shouldn't clear the revealed opponents for the actual ongoing battle
    if team_fully_reset(player) && team_fully_reset(opponent) {
        return true;
    }
    false
}

/// Select the most recent effect from tracked effects, matching interface.py's behavior
/// of picking the effect with the lowest turn number (most recently added)
fn select_most_recent_effect(
    side: &Side,
    pokemon: Option<&Pokemon>,
    effects_seen: &BTreeMap<String, u64>,
) -> String {
    let mut current_effects = HashSet::new();
    
    // Collect all currently active effects
    for status in side.volatile_statuses.iter() {
        let sanitized = clean_no_numbers(&status.to_string());
        if !sanitized.is_empty() && sanitized != "none" {
            current_effects.insert(sanitized);
        }
    }
    
    let durations = &side.volatile_status_durations;
    for (name, value) in [
        ("confusion", durations.confusion),
        ("encore", durations.encore),
        ("lockedmove", durations.lockedmove),
        ("slowstart", durations.slowstart),
        ("taunt", durations.taunt),
        ("yawn", durations.yawn),
    ] {
        if value > 0 {
            current_effects.insert(clean_no_numbers(name));
        }
    }
    
    if side.substitute_health > 0 {
        current_effects.insert("substitute".to_string());
    }
    if side.force_trapped {
        current_effects.insert("partiallytrapped".to_string());
    }
    if side.future_sight.0 > 0 {
        current_effects.insert("futuresight".to_string());
    }
    
    // Check for Supreme Overlord ability (Kingambit) - generates "fallen" effect
    // Only apply if the pokemon itself is not fainted
    if let Some(pokemon) = pokemon {
        if pokemon.hp > 0 {
            let ability_str = pokemon.ability.to_string().to_lowercase();
            if ability_str == "supremeoverlord" || ability_str == "supreme overlord" {
                let fainted_count = side.pokemon.pkmn.iter().filter(|p| p.hp == 0).count();
                if fainted_count > 0 {
                    current_effects.insert("fallen".to_string());
                }
            }
        }
    }
    
    if current_effects.is_empty() {
        return "noeffect".to_string();
    }
    
    // Find the effect with the lowest turn number (most recent) from those currently active
    let most_recent = current_effects
        .iter()
        .filter_map(|effect| effects_seen.get(effect).map(|turn| (effect, turn)))
        .min_by_key(|(_, turn)| *turn);
    
    if let Some((effect, _turn)) = most_recent {
        effect.to_string()
    } else {
        // Fallback: if none are tracked, just pick first alphabetically
        let mut sorted: Vec<String> = current_effects.into_iter().collect();
        sorted.sort();
        sorted[0].clone()
    }
}

fn collect_observation_context(
    player_side: &Side,
    opponent_side: &Side,
    player_pokemon: &[Pokemon],
    opponent_pokemon: &[Pokemon],
    battle_format: &str,
    perspective: &str,
) -> ObservationContext {
    let active_index = pokemon_index_to_usize(player_side.active_index);
    let opponent_active_index = pokemon_index_to_usize(opponent_side.active_index);

    let active_pokemon = &player_pokemon[active_index];
    let opponent_pokemon_active = &opponent_pokemon[opponent_active_index];

    let key = build_battle_key(player_pokemon, opponent_pokemon, battle_format, perspective);

    let mut trackers = observation_trackers().lock().unwrap();
    let tracker = trackers.entry(key).or_default();

    if should_reset_tracker(tracker, player_pokemon, opponent_pokemon) {
        tracker.reset();
    }

    tracker.steps_seen = tracker.steps_seen.saturating_add(1);

    if opponent_pokemon_active.status == PokemonStatus::SLEEP {
        tracker.any_opponent_asleep = true;
    }
    if opponent_pokemon_active.status == PokemonStatus::FREEZE {
        tracker.any_opponent_frozen = true;
    }

    tracker
        .revealed_opponents
        .insert(pokemon_name_str(&opponent_pokemon_active.id.to_string()));

    // Track player effects (volatile statuses and durations)
    for status in player_side.volatile_statuses.iter() {
        let sanitized = clean_no_numbers(&status.to_string());
        if !sanitized.is_empty() && sanitized != "none" {
            tracker.player_effects_seen.entry(sanitized).or_insert(tracker.steps_seen);
        }
    }
    let durations = &player_side.volatile_status_durations;
    for (name, value) in [
        ("confusion", durations.confusion),
        ("encore", durations.encore),
        ("lockedmove", durations.lockedmove),
        ("slowstart", durations.slowstart),
        ("taunt", durations.taunt),
        ("yawn", durations.yawn),
    ] {
        if value > 0 {
            let sanitized = clean_no_numbers(name);
            tracker.player_effects_seen.entry(sanitized).or_insert(tracker.steps_seen);
        }
    }
    if player_side.substitute_health > 0 {
        tracker.player_effects_seen.entry("substitute".to_string()).or_insert(tracker.steps_seen);
    }
    if player_side.force_trapped {
        tracker.player_effects_seen.entry("partiallytrapped".to_string()).or_insert(tracker.steps_seen);
    }
    if player_side.future_sight.0 > 0 {
        tracker.player_effects_seen.entry("futuresight".to_string()).or_insert(tracker.steps_seen);
    }

    // Track opponent effects
    for status in opponent_side.volatile_statuses.iter() {
        let sanitized = clean_no_numbers(&status.to_string());
        if !sanitized.is_empty() && sanitized != "none" {
            tracker.opponent_effects_seen.entry(sanitized).or_insert(tracker.steps_seen);
        }
    }
    let opp_durations = &opponent_side.volatile_status_durations;
    for (name, value) in [
        ("confusion", opp_durations.confusion),
        ("encore", opp_durations.encore),
        ("lockedmove", opp_durations.lockedmove),
        ("slowstart", opp_durations.slowstart),
        ("taunt", opp_durations.taunt),
        ("yawn", opp_durations.yawn),
    ] {
        if value > 0 {
            let sanitized = clean_no_numbers(name);
            tracker.opponent_effects_seen.entry(sanitized).or_insert(tracker.steps_seen);
        }
    }
    if opponent_side.substitute_health > 0 {
        tracker.opponent_effects_seen.entry("substitute".to_string()).or_insert(tracker.steps_seen);
    }
    if opponent_side.force_trapped {
        tracker.opponent_effects_seen.entry("partiallytrapped".to_string()).or_insert(tracker.steps_seen);
    }
    if opponent_side.future_sight.0 > 0 {
        tracker.opponent_effects_seen.entry("futuresight".to_string()).or_insert(tracker.steps_seen);
    }

    let any_opponent_asleep = tracker.any_opponent_asleep;
    let any_opponent_frozen = tracker.any_opponent_frozen;
    let mut revealed_opponents: Vec<String> = tracker.revealed_opponents.iter().cloned().collect();
    revealed_opponents.sort_unstable();
    
    let player_effects_seen = tracker.player_effects_seen.clone();
    let opponent_effects_seen = tracker.opponent_effects_seen.clone();

    let can_tera = !player_side
        .pokemon
        .pkmn
        .iter()
        .any(|p| p.terastallized);

    drop(trackers);

    ObservationContext {
        revealed_opponents,
        player_prev_move: last_used_move_token(player_side, active_pokemon),
        opponent_prev_move: last_used_move_token(opponent_side, opponent_pokemon_active),
        any_opponent_asleep,
        any_opponent_frozen,
        can_tera,
        player_effects_seen,
        opponent_effects_seen,
    }
}

pub fn build_inference_payload(
    state: &State,
    perspective: &str,
    battle_format: &str,
) -> InferencePayload {
    let (player_side, opponent_side) = match perspective {
        "side_one" => (&state.side_one, &state.side_two),
        "side_two" => (&state.side_two, &state.side_one),
        other => panic!("Invalid perspective: {}", other),
    };

    let player_pokemon: Vec<Pokemon> = player_side.pokemon.pkmn.iter().cloned().collect();
    let opponent_pokemon: Vec<Pokemon> = opponent_side.pokemon.pkmn.iter().cloned().collect();

    let context = collect_observation_context(
        player_side,
        opponent_side,
        &player_pokemon,
        &opponent_pokemon,
        battle_format,
        perspective,
    );

    let text = build_observation_text(
        state,
        player_side,
        opponent_side,
        &player_pokemon,
        &opponent_pokemon,
        battle_format,
        &context,
    );

    let tokenizer = get_tokenizer();
    let mut tokens = tokenizer.tokenize(&text);
    let numbers = build_observation_numbers(
        player_side,
        opponent_side,
        &player_pokemon,
        &opponent_pokemon,
        &context,
    );

    let active_index = pokemon_index_to_usize(player_side.active_index);
    let sorted_moves = sorted_active_moves(&player_pokemon[active_index]);
    let move_mapping = build_move_mapping_from_sorted(&sorted_moves);

    // Switches: alive first (alphabetically), then fainted (alphabetically)
    // This must match the sorting in text/numbers observation
    let mut switches: Vec<(usize, String, bool)> = player_pokemon
        .iter()
        .enumerate()
        .filter(|(idx, _p)| *idx != active_index)  // Include ALL non-active pokemon
        .map(|(idx, pokemon)| (idx, pokemon_name_str(&pokemon.id.to_string()), pokemon.hp > 0))
        .collect();
    switches.sort_by(|a, b| {
        // Sort by alive status first (alive = true comes before fainted = false)
        match b.2.cmp(&a.2) {
            std::cmp::Ordering::Equal => {
                // Within same alive status, sort alphabetically by name
                a.1.cmp(&b.1).then(a.0.cmp(&b.0))
            }
            other => other,
        }
    });
    
    // Switch mapping includes all switches (for observation alignment)
    let switch_mapping = build_switch_mapping(&switches.iter().map(|(idx, name, _)| (*idx, name.clone())).collect::<Vec<_>>());

    let mut legal_actions: Vec<usize> = Vec::new();
    if !player_side.force_switch {
        let moves_len = sorted_moves.len();
        for i in 0..moves_len {
            legal_actions.push(i);
        }
        if context.can_tera {
            for i in 0..moves_len {
                legal_actions.push(9 + i);
            }
        }
    }
    // Legal actions: only include ALIVE switches (can't switch to fainted pokemon)
    for (sorted_idx, (_, _, alive)) in switches.iter().enumerate() {
        if *alive {
            legal_actions.push(4 + sorted_idx);
        }
    }
    legal_actions.sort_unstable();

    InferencePayload {
        text_tokens: tokens,
        numbers,
        legal_actions,
        move_mapping,
        switch_mapping,
    }
}

fn build_inference_payload_batch_inner(
    states: &[&State],
    perspective: &str,
    battle_format: &str,
) -> BatchPayload {
    if states.is_empty() {
        return BatchPayload {
            text_tokens: Vec::new(),
            numbers: Vec::new(),
            legal_actions: Vec::new(),
            move_mappings: Vec::new(),
            switch_mappings: Vec::new(),
            token_len: 0,
            number_len: 0,
        };
    }

    let payloads: Vec<InferencePayload> = states
        .par_iter()
        .map(|state| build_inference_payload(*state, perspective, battle_format))
        .collect();

    let first = &payloads[0];
    let token_len = first.text_tokens.len();
    let number_len = first.numbers.len();

    let mut text_tokens: Vec<i32> = Vec::with_capacity(states.len() * token_len);
    let mut numbers: Vec<f32> = Vec::with_capacity(states.len() * number_len);
    let mut legal_actions: Vec<Vec<usize>> = Vec::with_capacity(states.len());
    let mut move_mappings: Vec<BTreeMap<usize, usize>> = Vec::with_capacity(states.len());
    let mut switch_mappings: Vec<BTreeMap<usize, usize>> = Vec::with_capacity(states.len());

    for payload in payloads {
        debug_assert_eq!(token_len, payload.text_tokens.len());
        debug_assert_eq!(number_len, payload.numbers.len());
        text_tokens.extend(payload.text_tokens);
        numbers.extend(payload.numbers);
        legal_actions.push(payload.legal_actions);
        move_mappings.push(payload.move_mapping);
        switch_mappings.push(payload.switch_mapping);
    }

    BatchPayload {
        text_tokens,
        numbers,
        legal_actions,
        move_mappings,
        switch_mappings,
        token_len,
        number_len,
    }
}

pub fn build_inference_payload_batch(
    states: &[State],
    perspective: &str,
    battle_format: &str,
) -> BatchPayload {
    let state_refs: Vec<&State> = states.iter().collect();
    build_inference_payload_batch_inner(&state_refs, perspective, battle_format)
}

pub fn build_inference_payload_batch_from_refs(
    states: &[&State],
    perspective: &str,
    battle_format: &str,
) -> BatchPayload {
    build_inference_payload_batch_inner(states, perspective, battle_format)
}

pub fn build_observation_direct(
    py: Python<'_>,
    state: &State,
    perspective: &str,
    battle_format: &str,
) -> PyResult<PyObject> {
    let tokenizer = get_tokenizer();
    
    let (player_side, opponent_side) = match perspective {
        "side_one" => (&state.side_one, &state.side_two),
        "side_two" => (&state.side_two, &state.side_one),
        _ => return Err(pyo3::exceptions::PyValueError::new_err("Invalid perspective")),
    };
    
    let player_pokemon: Vec<Pokemon> = player_side.pokemon.pkmn.iter().cloned().collect();
    let opponent_pokemon: Vec<Pokemon> = opponent_side.pokemon.pkmn.iter().cloned().collect();
    let context = collect_observation_context(
        player_side,
        opponent_side,
        &player_pokemon,
        &opponent_pokemon,
        battle_format,
        perspective,
    );
    
    // Build text string and tokenize
    let text = build_observation_text(
        state,
        player_side,
        opponent_side,
        &player_pokemon,
        &opponent_pokemon,
        battle_format,
        &context,
    );
    let mut tokens = tokenizer.tokenize(&text);
    
    
    let tokens_array = PyArray1::from_vec(py, tokens).to_owned();
    
    // Build numbers array
    let numbers = build_observation_numbers(
        player_side,
        opponent_side,
        &player_pokemon,
        &opponent_pokemon,
        &context,
    );
    let numbers_array = PyArray1::from_vec(py, numbers).to_owned();
    
    // Return as dict
    let result = PyDict::new(py);
    result.set_item("text_tokens", tokens_array)?;
    result.set_item("numbers", numbers_array)?;
    
    Ok(result.into())
}

#[pyfunction(name = "build_observation_direct")]
pub fn py_build_observation_direct(
    py: Python<'_>,
    py_state: crate::PyState,
    perspective: String,
    battle_format: String,
) -> PyResult<PyObject> {
    let state: State = py_state.into();
    build_observation_direct(py, &state, &perspective, &battle_format)
}

#[pyfunction(name = "build_observation_with_text")]
pub fn py_build_observation_with_text(
    py: Python<'_>,
    py_state: crate::PyState,
    perspective: String,
    battle_format: String,
) -> PyResult<PyObject> {
    let state: State = py_state.into();
    let tokenizer = get_tokenizer();
    
    let (player_side, opponent_side) = match perspective.as_str() {
        "side_one" => (&state.side_one, &state.side_two),
        "side_two" => (&state.side_two, &state.side_one),
        _ => return Err(pyo3::exceptions::PyValueError::new_err("Invalid perspective")),
    };
    let player_pokemon: Vec<Pokemon> = player_side.pokemon.pkmn.iter().cloned().collect();
    let opponent_pokemon: Vec<Pokemon> = opponent_side.pokemon.pkmn.iter().cloned().collect();
    let context = collect_observation_context(
        player_side,
        opponent_side,
        &player_pokemon,
        &opponent_pokemon,
        &battle_format,
        &perspective,
    );
    
    // Build text string
    let text = build_observation_text(
        &state,
        player_side,
        opponent_side,
        &player_pokemon,
        &opponent_pokemon,
        &battle_format,
        &context,
    );
    // Tokenize
    let mut tokens = tokenizer.tokenize(&text);
    
    
    let tokens_array = PyArray1::from_vec(py, tokens).to_owned();
    
    // Build numbers array
    let numbers = build_observation_numbers(
        player_side,
        opponent_side,
        &player_pokemon,
        &opponent_pokemon,
        &context,
    );
    let numbers_array = PyArray1::from_vec(py, numbers).to_owned();
    
    // Return as dict with text included
    let result = PyDict::new(py);
    result.set_item("text", text)?;
    result.set_item("text_tokens", tokens_array)?;
    result.set_item("numbers", numbers_array)?;
    
    Ok(result.into())
}

#[pyfunction(name = "prepare_inference_payload")]
pub fn py_prepare_inference_payload(
    py: Python<'_>,
    py_state: crate::PyState,
    perspective: String,
    battle_format: String,
) -> PyResult<PyObject> {
    let state: State = py_state.into();

    if perspective != "side_one" && perspective != "side_two" {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Invalid perspective: {}",
            perspective
        )));
    }

    let payload = build_inference_payload(&state, &perspective, &battle_format);

    let text_tokens = PyArray1::from_vec(py, payload.text_tokens.clone()).to_owned();
    let numbers = PyArray1::from_vec(py, payload.numbers.clone()).to_owned();

    let result = PyDict::new(py);
    result.set_item("text_tokens", text_tokens)?;
    result.set_item("numbers", numbers)?;
    result.set_item("legal_actions", PyList::new(py, payload.legal_actions))?;

    let move_mapping_dict = PyDict::new(py);
    for (k, v) in payload.move_mapping {
        move_mapping_dict.set_item(k, v)?;
    }
    result.set_item("move_mapping", move_mapping_dict)?;

    let switch_mapping_dict = PyDict::new(py);
    for (k, v) in payload.switch_mapping {
        switch_mapping_dict.set_item(k, v)?;
    }
    result.set_item("switch_mapping", switch_mapping_dict)?;

    Ok(result.into())
}

#[pyfunction(name = "prepare_inference_payload_batch")]
pub fn py_prepare_inference_payload_batch(
    py: Python<'_>,
    py_states: Vec<crate::PyState>,
    perspective: String,
    battle_format: String,
) -> PyResult<PyObject> {
    if perspective != "side_one" && perspective != "side_two" {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Invalid perspective: {}",
            perspective
        )));
    }

    let states: Vec<State> = py_states.into_iter().map(Into::into).collect();
    let payload = build_inference_payload_batch(&states, &perspective, &battle_format);
    let batch_size = states.len();

    let text_tokens = unsafe {
        // Safety: we create a new contiguous array and immediately fill it with a
        // slice of matching length, so no aliasing occurs and the layout matches.
        let array = PyArray2::<i32>::new(py, [batch_size, payload.token_len], false);
        if !payload.text_tokens.is_empty() {
            array
                .as_slice_mut()
                .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?
                .copy_from_slice(&payload.text_tokens);
        }
        array.to_owned()
    };

    let numbers = unsafe {
        // Safety: identical reasoning as above, using a contiguous f32 buffer.
        let array = PyArray2::<f32>::new(py, [batch_size, payload.number_len], false);
        if !payload.numbers.is_empty() {
            array
                .as_slice_mut()
                .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?
                .copy_from_slice(&payload.numbers);
        }
        array.to_owned()
    };

    let legal_actions = PyList::empty(py);
    for actions in payload.legal_actions {
        legal_actions.append(PyList::new(py, actions))?;
    }

    let move_mappings = PyList::empty(py);
    for mapping in payload.move_mappings {
        let dict = PyDict::new(py);
        for (k, v) in mapping {
            dict.set_item(k, v)?;
        }
        move_mappings.append(dict)?;
    }

    let switch_mappings = PyList::empty(py);
    for mapping in payload.switch_mappings {
        let dict = PyDict::new(py);
        for (k, v) in mapping {
            dict.set_item(k, v)?;
        }
        switch_mappings.append(dict)?;
    }

    let result = PyDict::new(py);
    result.set_item("text_tokens", text_tokens)?;
    result.set_item("numbers", numbers)?;
    result.set_item("legal_actions", legal_actions)?;
    result.set_item("move_mappings", move_mappings)?;
    result.set_item("switch_mappings", switch_mappings)?;
    result.set_item("token_len", payload.token_len)?;
    result.set_item("number_len", payload.number_len)?;
    result.set_item("batch_size", batch_size)?;

    Ok(result.into())
}

#[pyfunction(name = "prepare_inference_payload_batch_from_pointers")]
pub fn py_prepare_inference_payload_batch_from_pointers(
    py: Python<'_>,
    ptrs: Vec<usize>,
    perspective: String,
    battle_format: String,
) -> PyResult<PyObject> {
    if perspective != "side_one" && perspective != "side_two" {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Invalid perspective: {}",
            perspective
        )));
    }

    let mut state_refs: Vec<&State> = Vec::with_capacity(ptrs.len());
    for ptr in ptrs {
        if ptr == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Null state pointer provided",
            ));
        }
        let state_ref = unsafe { &*(ptr as *const State) };
        state_refs.push(state_ref);
    }

    let payload = build_inference_payload_batch_from_refs(&state_refs, &perspective, &battle_format);
    let batch_size = state_refs.len();

    let text_tokens = unsafe {
        let array = PyArray2::<i32>::new(py, [batch_size, payload.token_len], false);
        if !payload.text_tokens.is_empty() {
            array
                .as_slice_mut()
                .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?
                .copy_from_slice(&payload.text_tokens);
        }
        array.to_owned()
    };

    let numbers = unsafe {
        let array = PyArray2::<f32>::new(py, [batch_size, payload.number_len], false);
        if !payload.numbers.is_empty() {
            array
                .as_slice_mut()
                .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?
                .copy_from_slice(&payload.numbers);
        }
        array.to_owned()
    };

    let legal_actions = PyList::empty(py);
    for actions in payload.legal_actions {
        legal_actions.append(PyList::new(py, actions))?;
    }

    let move_mappings = PyList::empty(py);
    for mapping in payload.move_mappings {
        let dict = PyDict::new(py);
        for (k, v) in mapping {
            dict.set_item(k, v)?;
        }
        move_mappings.append(dict)?;
    }

    let switch_mappings = PyList::empty(py);
    for mapping in payload.switch_mappings {
        let dict = PyDict::new(py);
        for (k, v) in mapping {
            dict.set_item(k, v)?;
        }
        switch_mappings.append(dict)?;
    }

    let result = PyDict::new(py);
    result.set_item("text_tokens", text_tokens)?;
    result.set_item("numbers", numbers)?;
    result.set_item("legal_actions", legal_actions)?;
    result.set_item("move_mappings", move_mappings)?;
    result.set_item("switch_mappings", switch_mappings)?;
    result.set_item("token_len", payload.token_len)?;
    result.set_item("number_len", payload.number_len)?;
    result.set_item("batch_size", batch_size)?;

    Ok(result.into())
}
