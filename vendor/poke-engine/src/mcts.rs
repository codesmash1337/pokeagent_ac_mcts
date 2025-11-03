use crate::engine::evaluate::evaluate as heuristic_evaluate;
use crate::engine::generate_instructions::generate_instructions_from_move_pair;
use crate::engine::state::MoveChoice;
use crate::instruction::StateInstructions;
use crate::neural_evaluate;
use crate::neural_evaluate::drain_raw_value_samples;
use crate::state::{PokemonIndex, PokemonMoveIndex, SideReference, State};
use rand::distr::{weighted::WeightedIndex, Distribution};
use rand::rng;
use rand_distr::Gamma;
use std::collections::HashMap;
use std::fs::{create_dir_all, File, OpenOptions};
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const PUCT_C1: f32 = 2.5;
const PUCT_C2: f32 = 5000.0;
const DIRICHLET_NOISE_EPSILON: f32 = 0.4;
const DIRICHLET_NOISE_ALPHA: f32 = 0.8; // 13 possible actions = 5/13
const VIRTUAL_PRIOR_STRENGTH: f32 = 8.0;
const VIRTUAL_PRIOR_BASELINE: f32 = 0.5;
const VERBOSE_LOGGING: bool = false;
const BATCH_SIZE: usize = 64;
// When true, skip MCTS rollouts and seed visits directly from policy priors (for debugging)
const SANITY_CHECK_POLICY_ONLY: bool = false;
// When true, use hand-crafted heuristic evaluation instead of neural network values
const USE_HEURISTIC_VALUE: bool = false;

macro_rules! verbose_eprintln {
    ($($arg:tt)*) => {
        if VERBOSE_LOGGING {
            eprintln!($($arg)*);
        }
    };
}

fn sigmoid(x: f32) -> f32 {
    // Tuned so that ~200 points is very close to 1.0
    1.0 / (1.0 + (-0.0125 * x).exp())
}

#[derive(Clone, Copy)]
enum ValueSource {
    Neural,
    Heuristic,
}

#[derive(Clone)]
struct EvalOutcome {
    value: f32,
    source: ValueSource,
    policy: Option<Vec<f32>>,
}

#[derive(Default)]
struct TreeStats {
    node_count: usize,
    total_visits: u64,
    never_visited: usize,
    min_avg: f32,
    max_avg: f32,
    sum_avg: f64,
}

impl TreeStats {
    fn update(&mut self, visits: u32, total_score: f32) {
        self.node_count += 1;
        self.total_visits += visits as u64;
        if visits == 0 {
            self.never_visited += 1;
            return;
        }
        let avg = total_score / visits as f32;
        if avg < self.min_avg {
            self.min_avg = avg;
        }
        if avg > self.max_avg {
            self.max_avg = avg;
        }
        self.sum_avg += avg as f64;
    }

    fn average_avg(&self) -> f32 {
        if self.node_count == 0 {
            0.0
        } else {
            (self.sum_avg / self.node_count as f64) as f32
        }
    }

    fn log_summary(&self) {
        eprintln!(
            "Tree stats | nodes: {} | visited nodes: {} | never visited: {} | avg(avg): {:.3} | min(avg): {:.3} | max(avg): {:.3}",
            self.node_count,
            self.node_count - self.never_visited,
            self.never_visited,
            self.average_avg(),
            self.min_avg,
            self.max_avg
        );
    }

    fn write_json(&self, path: &str) {
        if let Ok(mut file) = File::create(path) {
            let _ = writeln!(
                file,
                "{{\"node_count\":{},\"visited_nodes\":{},\"never_visited\":{},\"avg_avg\":{:.6},\"min_avg\":{:.6},\"max_avg\":{:.6}}}",
                self.node_count,
                self.node_count - self.never_visited,
                self.never_visited,
                self.average_avg(),
                self.min_avg,
                self.max_avg
            );
        }
    }
}

fn collect_root_stats(options: &[MoveNode]) -> TreeStats {
    let mut stats = TreeStats::default();
    for node in options {
        stats.update(node.visits, node.total_score);
    }
    stats
}

fn apply_dirichlet_noise_to_priors(nodes: &mut [MoveNode]) {
    if nodes.is_empty() {
        return;
    }

    let active_indices: Vec<usize> = nodes
        .iter()
        .enumerate()
        .filter(|(_, node)| !matches!(node.move_choice, MoveChoice::None))
        .map(|(idx, _)| idx)
        .collect();

    if active_indices.len() <= 1 {
        return;
    }

    if let Ok(gamma) = Gamma::new(DIRICHLET_NOISE_ALPHA, 1.0) {
        let mut rng = rng();
        let mut noise: Vec<f32> = active_indices
            .iter()
            .map(|_| gamma.sample(&mut rng) as f32)
            .collect();

        let sum_noise: f32 = noise.iter().sum();
        if sum_noise <= f32::EPSILON {
            return;
        }

        for value in noise.iter_mut() {
            *value /= sum_noise;
        }

        for (idx, noise_val) in active_indices.iter().zip(noise.iter()) {
            let node = &mut nodes[*idx];
            node.prior = (1.0 - DIRICHLET_NOISE_EPSILON) * node.prior
                + DIRICHLET_NOISE_EPSILON * noise_val;
        }

        let total: f32 = nodes.iter().map(|node| node.prior).sum();
        if total > f32::EPSILON {
            for node in nodes.iter_mut() {
                node.prior /= total;
            }
        }

        apply_virtual_priors(nodes);
    }
}

fn evaluate_with_fallback(state: &State) -> EvalOutcome {
    if USE_HEURISTIC_VALUE {
        let heuristic_value = heuristic_evaluate(state);
        // Still get policy from neural network for move selection
        let policy = neural_evaluate::neural_state_value(state)
            .map(|v| v.policy)
            .unwrap_or_else(|| {
                eprintln!("[WARN] Neural policy unavailable in heuristic mode, using uniform");
                vec![]
            });
        EvalOutcome {
            value: sigmoid(heuristic_value),
            source: ValueSource::Heuristic,
            policy: Some(policy),
        }
    } else if let Some(value) = neural_evaluate::neural_state_value(state) {
        EvalOutcome {
            value: value.value,
            source: ValueSource::Neural,
            policy: Some(value.policy),
        }
    } else {
        panic!("Evaluation FAILED what are you doing with your life");
    }
}

fn evaluate_with_fallback_for_side(state: &State, side: SideReference) -> EvalOutcome {
    if USE_HEURISTIC_VALUE {
        let heuristic_value = heuristic_evaluate(state);
        // Apply perspective: side_two sees inverted value
        let adjusted_value = match side {
            SideReference::SideOne => heuristic_value,
            SideReference::SideTwo => -heuristic_value,
        };
        // Still get policy from neural network for move selection
        let policy = neural_evaluate::neural_state_value_for_side(state, side)
            .map(|v| v.policy)
            .unwrap_or_else(|| {
                eprintln!("[WARN] Neural policy unavailable in heuristic mode, using uniform");
                vec![]
            });
        EvalOutcome {
            value: sigmoid(adjusted_value),
            source: ValueSource::Heuristic,
            policy: Some(policy),
        }
    } else if let Some(value) = neural_evaluate::neural_state_value_for_side(state, side) {
        EvalOutcome {
            value: value.value,
            source: ValueSource::Neural,
            policy: Some(value.policy),
        }
    } else {
        panic!("Evaluation FAILED what are you doing with your life");
    }
}

fn evaluate_with_fallback_batch(states: &[&State]) -> Vec<EvalOutcome> {
    if USE_HEURISTIC_VALUE {
        // Evaluate each state with heuristic, still get policies from neural network
        states
            .iter()
            .map(|state| {
                let heuristic_value = heuristic_evaluate(state);
                let policy = neural_evaluate::neural_state_value(state)
                    .map(|v| v.policy)
                    .unwrap_or_else(|| {
                        eprintln!("[WARN] Neural policy unavailable in heuristic mode, using uniform");
                        vec![]
                    });
                EvalOutcome {
                    value: sigmoid(heuristic_value),
                    source: ValueSource::Heuristic,
                    policy: Some(policy),
                }
            })
            .collect()
    } else {
    match neural_evaluate::neural_state_values_batch(states) {
        Some(values) => values
            .into_iter()
            .map(|val| EvalOutcome {
                value: val.value,
                source: ValueSource::Neural,
                policy: Some(val.policy),
            })
            .collect(),
        None => panic!("Batched evaluation FAILED: neural inference unavailable"),
        }
    }
}

fn evaluate_with_fallback_batch_for_side(states: &[&State], side: SideReference) -> Vec<EvalOutcome> {
    if USE_HEURISTIC_VALUE {
        // Evaluate each state with heuristic, still get policies from neural network
        states
            .iter()
            .map(|state| {
                let heuristic_value = heuristic_evaluate(state);
                // Apply perspective: side_two sees inverted value
                let adjusted_value = match side {
                    SideReference::SideOne => heuristic_value,
                    SideReference::SideTwo => -heuristic_value,
                };
                let policy = neural_evaluate::neural_state_value_for_side(state, side)
                    .map(|v| v.policy)
                    .unwrap_or_else(|| {
                        eprintln!("[WARN] Neural policy unavailable in heuristic mode, using uniform");
                        vec![]
                    });
                EvalOutcome {
                    value: sigmoid(adjusted_value),
                    source: ValueSource::Heuristic,
                    policy: Some(policy),
                }
            })
            .collect()
    } else {
    match neural_evaluate::neural_state_values_batch_for_side(states, side) {
        Some(values) => values
            .into_iter()
            .map(|val| EvalOutcome {
                value: val.value,
                source: ValueSource::Neural,
                policy: Some(val.policy),
            })
            .collect(),
        None => panic!("Batched evaluation FAILED: neural inference unavailable"),
        }
    }
}

fn ensure_logging_dir() -> &'static str {
    static mut DIR: Option<String> = None;
    unsafe {
        DIR.get_or_insert_with(|| {
            let ts = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            let dir = format!("mcts_logs/{}", ts);
            let _ = create_dir_all(&dir);
            dir
        });
        DIR.as_ref().unwrap()
    }
}

struct LoggingPaths {
    state_path: String,
    stats_path: String,
    tree_path: String,
    comparison_path: String,
    comparison_path_side2: String,
    q_u_values_path: String,
    value_scale_path: String,
}

static LOG_TURN: AtomicU32 = AtomicU32::new(0);
static SWITCH_DEBUG_PRINTED: AtomicBool = AtomicBool::new(true);
static SWITCH_DEBUG_TURN: AtomicU32 = AtomicU32::new(0);
static PRIOR_MAP_PRINTED: AtomicBool = AtomicBool::new(true);

fn logging_paths_for_turn(turn: u32) -> LoggingPaths {
    let dir = ensure_logging_dir();
    let turn_dir = format!("{}/turn_{:03}", dir, turn);
    let _ = create_dir_all(&turn_dir);
    LoggingPaths {
        state_path: format!("{}/state_value_log.tsv", turn_dir),
        stats_path: format!("{}/mcts_stats.json", turn_dir),
        tree_path: format!("{}/mcts_tree_root.json", turn_dir),
        comparison_path: format!("{}/policy_vs_mcts.txt", turn_dir),
        comparison_path_side2: format!("{}/policy_vs_mcts_side2.txt", turn_dir),
        q_u_values_path: format!("{}/q_u_values_by_batch.txt", turn_dir),
        value_scale_path: format!("{}/value_scale_samples.csv", turn_dir),
    }
}

fn format_state_readable(state: &State) -> String {
    let mut output = String::new();
    
    output.push_str("Side One (Player):\n");
    let s1_active_idx = pokemon_index_to_usize(state.side_one.active_index);
    let s1_active = &state.side_one.pokemon.pkmn[s1_active_idx];
    output.push_str(&format!("  Active: {} (HP: {}/{}, Status: {:?})\n", 
        s1_active.id, s1_active.hp, s1_active.maxhp, s1_active.status));
    output.push_str("  Team:\n");
    for (i, pkmn) in state.side_one.pokemon.pkmn.iter().enumerate() {
        let marker = if i == s1_active_idx { "ACTIVE" } else { "" };
        output.push_str(&format!("    [{}] {} - HP: {}/{}, Status: {:?}, Ability: {:?} {}\n",
            i, pkmn.id, pkmn.hp, pkmn.maxhp, pkmn.status, pkmn.ability, marker));
    }
    output.push_str(&format!("  Side Conditions: {:?}\n", state.side_one.side_conditions));
    output.push_str(&format!("  Last Used Move: {:?}\n", state.side_one.last_used_move));
    
    output.push_str("\nSide Two (Opponent):\n");
    let s2_active_idx = pokemon_index_to_usize(state.side_two.active_index);
    let s2_active = &state.side_two.pokemon.pkmn[s2_active_idx];
    output.push_str(&format!("  Active: {} (HP: {}/{}, Status: {:?})\n", 
        s2_active.id, s2_active.hp, s2_active.maxhp, s2_active.status));
    output.push_str("  Team:\n");
    for (i, pkmn) in state.side_two.pokemon.pkmn.iter().enumerate() {
        let marker = if i == s2_active_idx { "ACTIVE" } else { "" };
        output.push_str(&format!("    [{}] {} - HP: {}/{}, Status: {:?}, Ability: {:?} {}\n",
            i, pkmn.id, pkmn.hp, pkmn.maxhp, pkmn.status, pkmn.ability, marker));
    }
    output.push_str(&format!("  Side Conditions: {:?}\n", state.side_two.side_conditions));
    output.push_str(&format!("  Last Used Move: {:?}\n", state.side_two.last_used_move));
    
    output.push_str(&format!("\nWeather: {:?}\n", state.weather.weather_type));
    output.push_str(&format!("Terrain: {:?}\n", state.terrain.terrain_type));
    output.push_str(&format!("Trick Room: {:?}\n", state.trick_room));
    
    output
}

fn log_state_value(state: &State, eval: &EvalOutcome, paths: &LoggingPaths) {
    let serialized = state.serialize();
    let sanitized_state = serialized.replace('\n', "\\n");
    let source_label = match eval.source {
        ValueSource::Neural => "neural",
        ValueSource::Heuristic => "heuristic",
    };

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0);

    if let Ok(mut file) = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&paths.state_path)
    {
        let _ = writeln!(
            file,
            "{:.3}\t{}\t{:.6}\t{}",
            timestamp, source_label, eval.value, sanitized_state
        );
    }
}

fn log_policy_priors_comparison(
    state: &State,
    options: &[MoveNode],
    side_ref: SideReference,
    path: &str,
    policy_priors: Option<&[f32]>,
    root_node: Option<*const Node>,
) {
    if let Ok(mut file) = File::create(path) {
        let _ = writeln!(file, "=== POLICY PRIORS vs MCTS RESULTS ===\n");

        // Print the current state for debugging
        let _ = writeln!(file, "=== BATTLE STATE ===");
        let readable_state = format_state_readable(state);
        let _ = write!(file, "{}", readable_state);
        let _ = writeln!(file, "===================\n");

        // Also emit a compact per-option mapping file alongside policy_vs_mcts.txt
        // This clarifies, per option, the action kind and how it maps to policy indices
        if let Some(parent_dir) = std::path::Path::new(path).parent() {
            let side_suffix = match side_ref {
                SideReference::SideOne => "side1",
                SideReference::SideTwo => "side2",
            };
            let mapping_path = parent_dir.join(format!("action_mapping_{}.txt", side_suffix));
            if let Ok(mut mapf) = File::create(&mapping_path) {
                // Fetch raw policy from the network so we can show unmodified priors
                let raw_policy_owned: Option<Vec<f32>> = policy_priors.map(|s| s.to_vec());
                let _ = writeln!(mapf, "=== ACTION MAPPING (side {:?}) ===", side_ref);
                if let Some(ref rp) = raw_policy_owned {
                    let raw_fmt: Vec<String> = rp.iter().map(|v| format!("{:.4}", v)).collect();
                    let _ = writeln!(mapf, "RawPolicy (len {}): [{}]", rp.len(), raw_fmt.join(", "));
                } else {
                    let _ = writeln!(mapf, "RawPolicy: <unavailable>");
                }
                let _ = writeln!(mapf, "{:<4} {:<8} {:<8} {:<8} {:<8} {:<8} {:<20} {:>10} {:>10} {:>8} {:>8}",
                    "Idx", "Kind", "Team", "ObsSlot", "ActIdx", "MoveIdx", "Name", "RawPrior", "Prior", "Visits", "AvgVal");
                let _ = writeln!(mapf, "{}", "-".repeat(120));
                for (idx, node) in options.iter().enumerate() {
                    let (kind, team_str, obs_slot_str, act_idx_str, move_idx_str, name_str) = match &node.move_choice {
                        MoveChoice::Move(midx) => {
                            let m = move_index_to_usize(*midx);
                            let act_idx = action_index_for_choice(state, side_ref, &node.move_choice)
                                .unwrap_or(m);
                            let mv_name = match side_ref {
                                SideReference::SideOne => format!("{:?}", state.side_one.get_active_immutable().moves[midx].id).to_lowercase(),
                                SideReference::SideTwo => format!("{:?}", state.side_two.get_active_immutable().moves[midx].id).to_lowercase(),
                            };
                            ("MOVE", "-".to_string(), "-".to_string(), format!("{}", act_idx), format!("{}", m), mv_name)
                        }
                        #[cfg(not(any(feature = "gen1", feature = "gen2", feature = "gen3")))]
                        MoveChoice::MoveTera(midx) => {
                            let m = move_index_to_usize(*midx);
                            let act_idx = action_index_for_choice(state, side_ref, &node.move_choice)
                                .unwrap_or(9 + m);
                            let mv_name = match side_ref {
                                SideReference::SideOne => format!("{:?}", state.side_one.get_active_immutable().moves[midx].id).to_lowercase(),
                                SideReference::SideTwo => format!("{:?}", state.side_two.get_active_immutable().moves[midx].id).to_lowercase(),
                            };
                            ("TERA", "-".to_string(), "-".to_string(), format!("{}", act_idx), format!("{}", m), mv_name)
                        }
                        MoveChoice::Switch(pidx) => {
                            let team_idx = pokemon_index_to_usize(*pidx);
                            let slot_opt = switch_slot_index(state, side_ref, *pidx);
                            let act_idx_opt = slot_opt.map(|s| 4 + s);
                            let name = match side_ref {
                                SideReference::SideOne => format!("{:?}", state.side_one.pokemon.pkmn[team_idx].id).to_lowercase(),
                                SideReference::SideTwo => format!("{:?}", state.side_two.pokemon.pkmn[team_idx].id).to_lowercase(),
                            };
                            ("SWITCH",
                             format!("{}", team_idx),
                             slot_opt.map(|s| s.to_string()).unwrap_or_else(|| "-".to_string()),
                             act_idx_opt.map(|a| a.to_string()).unwrap_or_else(|| "-".to_string()),
                             "-".to_string(),
                             name)
                        }
                        _ => ("NONE", "-".to_string(), "-".to_string(), "-".to_string(), "-".to_string(), "-".to_string()),
                    };
                    let avg = if node.visits > 0 { node.total_score / node.visits as f32 } else { 0.0 };
                    // compute raw prior from raw_policy and act_idx if available
                    let raw_prior_val = match act_idx_str.parse::<usize>() {
                        Ok(ai) => raw_policy_owned.as_ref().and_then(|v| v.get(ai).copied()).unwrap_or(0.0),
                        Err(_) => 0.0,
                    };
                    let _ = writeln!(mapf,
                        "{:<4} {:<8} {:<8} {:<8} {:<8} {:<8} {:<20} {:>10.4} {:>10.4} {:>8} {:>8.4}",
                        idx, kind, team_str, obs_slot_str, act_idx_str, move_idx_str, name_str, raw_prior_val, node.prior, node.visits, avg);
                }
            }
        }

        // Print the observation and tokenization
        let _ = writeln!(file, "=== OBSERVATION & TOKENIZATION ===");
        match neural_evaluate::get_observation_for_logging_no_decode(state, side_ref) {
            Some((text_tokens, numbers)) => {
                let _ = writeln!(file, "Text tokens (length {}): {:?}", text_tokens.len(), text_tokens);
                let _ = writeln!(file, "Numerical features (length {}): {:?}", numbers.len(), numbers.iter().map(|x| format!("{:.3}", x)).collect::<Vec<_>>());
            }
            None => {
                let _ = writeln!(file, "Failed to generate observation - check stderr for [OBSERVATION ERROR] messages");
                eprintln!("Failed to generate observation for logging in turn file: {}", path);
            }
        }
        let _ = writeln!(file, "===================\n");

        // Switch debug section to clarify mapping between observation order, policy priors, and team indices
        let side_state = match side_ref {
            SideReference::SideOne => &state.side_one,
            SideReference::SideTwo => &state.side_two,
        };
        let active_idx = pokemon_index_to_usize(side_state.active_index);
        let force_switch = side_state.force_switch
            || side_state.pokemon.pkmn[active_idx].hp <= 0;
        let has_switch_option = options
            .iter()
            .any(|node| matches!(node.move_choice, MoveChoice::Switch(_)));
        if has_switch_option {
            // Build observation-style ordering (alive alphabetically, then fainted)
            let mut observation_switches: Vec<(usize, &crate::state::Pokemon)> = side_state
                .pokemon
                .pkmn
                .iter()
                .enumerate()
                .filter(|(idx, _)| *idx != active_idx)
                .collect();
            observation_switches.sort_by(|(idx_a, p_a), (idx_b, p_b)| {
                let alive_a = p_a.hp > 0;
                let alive_b = p_b.hp > 0;
                match alive_b.cmp(&alive_a) {
                    std::cmp::Ordering::Equal => {
                        let name_a = format!("{:?}", p_a.id).to_lowercase();
                        let name_b = format!("{:?}", p_b.id).to_lowercase();
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

            let _ = writeln!(file, "--- Switch Mapping Debug ---");
            let _ = writeln!(file, "Force switch turn: {}", force_switch);
            let _ = writeln!(file, "Observation slot order (what the policy sees):");
            let _ = writeln!(file, "{:<5} {:<7} {:<18} {:>7} {:>7}", "Slot", "Team", "Name", "HP", "Alive");
            for (slot, (team_idx, pokemon)) in observation_switches.iter().enumerate() {
                let name = format!("{:?}", pokemon.id).to_lowercase();
                let hp = if pokemon.maxhp > 0 {
                    let current_hp = if pokemon.hp > 0 { pokemon.hp as f32 } else { 0.0 };
                    (current_hp / (pokemon.maxhp as f32) * 100.0).clamp(0.0, 100.0)
                } else {
                    0.0
                };
                let alive = if pokemon.hp > 0 { "yes" } else { "no" };
                let _ = writeln!(
                    file,
                    "{:<5} {:<7} {:<18} {:>6.1}% {:>7}",
                    slot,
                    team_idx,
                    name,
                    hp,
                    alive
                );
            }

            let _ = writeln!(file, "\nMCTS switch options (policy prior ↔ team index mapping):");
            let _ = writeln!(
                file,
                "{:<5} {:<10} {:<7} {:<7} {:>8} {:>8} {:<18}",
                "Opt",
                "ActionIdx",
                "Team",
                "ObsSlot",
                "Prior",
                "Visits",
                "Name"
            );
            for (opt_idx, node) in options.iter().enumerate() {
                if let MoveChoice::Switch(pokemon_index) = node.move_choice {
                    let team_idx = pokemon_index_to_usize(pokemon_index);
                    let obs_slot = observation_switches
                        .iter()
                        .position(|(idx, _)| *idx == team_idx)
                        .map(|s| s as i32)
                        .unwrap_or(-1);
                    let name = format!(
                        "{:?}",
                        side_state.pokemon.pkmn[team_idx].id
                    )
                    .to_lowercase();
                    let hp = {
                        let pkmn = &side_state.pokemon.pkmn[team_idx];
                        if pkmn.maxhp > 0 {
                            let current_hp = if pkmn.hp > 0 { pkmn.hp as f32 } else { 0.0 };
                            (current_hp / (pkmn.maxhp as f32) * 100.0).clamp(0.0, 100.0)
                        } else {
                            0.0
                        }
                    };
                    let action_idx = action_index_for_choice(state, side_ref, &node.move_choice)
                        .map(|v| v as i32)
                        .unwrap_or(-1);
                    let _ = writeln!(
                        file,
                        "{:<5} {:<10} {:<7} {:<7} {:>8.4} {:>8} {:<18} ({:>5.1}%)",
                        opt_idx,
                        action_idx,
                        team_idx,
                        if obs_slot >= 0 { obs_slot.to_string() } else { "--".to_string() },
                        node.prior,
                        node.visits,
                        name,
                        hp
                    );
                }
            }
            let _ = writeln!(file);
        }

        // Sort by prior for initial policy view
        let mut by_prior: Vec<_> = options.iter().enumerate().collect();
        by_prior.sort_by(|a, b| b.1.prior.partial_cmp(&a.1.prior).unwrap_or(std::cmp::Ordering::Equal));

        let _ = writeln!(file, "--- Initial Policy (sorted by prior) ---");
        let _ = writeln!(file, "{:<4} {:<40} {:>8}", "Idx", "Move", "Prior");
        let _ = writeln!(file, "{}", "-".repeat(60));
        for (idx, node) in by_prior.iter().take(10) {
            let move_str = match side_ref {
                SideReference::SideOne => node.move_choice.to_string(&state.side_one),
                SideReference::SideTwo => node.move_choice.to_string(&state.side_two),
            };
            let _ = writeln!(file, "{:<4} {:<40} {:>8.4}", idx, move_str, node.prior);
        }

        // Sort by visits for MCTS results
        let mut by_visits: Vec<_> = options.iter().enumerate().collect();
        by_visits.sort_by(|a, b| b.1.visits.cmp(&a.1.visits));

        let _ = writeln!(file, "\n--- MCTS Results (sorted by visits) ---");
        let _ = writeln!(file, "{:<4} {:<40} {:>8} {:>8} {:>8}", "Idx", "Move", "Prior", "Visits", "AvgVal");
        let _ = writeln!(file, "{}", "-".repeat(80));
        for (idx, node) in by_visits.iter().take(10) {
            let move_str = match side_ref {
                SideReference::SideOne => node.move_choice.to_string(&state.side_one),
                SideReference::SideTwo => node.move_choice.to_string(&state.side_two),
            };
            let avg = if node.visits > 0 {
                node.total_score / node.visits as f32
            } else {
                0.0
            };
            let _ = writeln!(
                file,
                "{:<4} {:<40} {:>8.4} {:>8} {:>8.4}",
                idx, move_str, node.prior, node.visits, avg
            );
        }

        // Show divergence: moves with high prior but low visits, and vice versa
        let _ = writeln!(file, "\n--- Divergence Analysis ---");

        let total_visits: u32 = options.iter().map(|n| n.visits).sum();
        let _ = writeln!(file, "Total visits: {}", total_visits);

        let _ = writeln!(file, "\nMoves with HIGH PRIOR but LOW VISITS (policy disagreement):");
        let _ = writeln!(file, "{:<4} {:<40} {:>8} {:>10} {:>10}", "Idx", "Move", "Prior", "Visits", "Visit%");
        let _ = writeln!(file, "{}", "-".repeat(90));
        let mut high_prior_low_visit: Vec<_> = options.iter().enumerate()
            .filter(|(_, n)| n.prior > 0.05)
            .collect();
        high_prior_low_visit.sort_by(|a, b| {
            let a_ratio = if total_visits > 0 { a.1.visits as f32 / total_visits as f32 } else { 0.0 };
            let b_ratio = if total_visits > 0 { b.1.visits as f32 / total_visits as f32 } else { 0.0 };
            (a.1.prior - a_ratio).partial_cmp(&(b.1.prior - b_ratio)).unwrap_or(std::cmp::Ordering::Equal).reverse()
        });
        for (idx, node) in high_prior_low_visit.iter().take(5) {
            let move_str = match side_ref {
                SideReference::SideOne => node.move_choice.to_string(&state.side_one),
                SideReference::SideTwo => node.move_choice.to_string(&state.side_two),
            };
            let visit_pct = if total_visits > 0 {
                (node.visits as f32 / total_visits as f32) * 100.0
            } else {
                0.0
            };
            let _ = writeln!(
                file,
                "{:<4} {:<40} {:>8.4} {:>10} {:>9.2}%",
                idx, move_str, node.prior, node.visits, visit_pct
            );
        }

        let _ = writeln!(file, "\nMoves with LOW PRIOR but HIGH VISITS (MCTS found better):");
        let _ = writeln!(file, "{:<4} {:<40} {:>8} {:>10} {:>10}", "Idx", "Move", "Prior", "Visits", "Visit%");
        let _ = writeln!(file, "{}", "-".repeat(90));
        let mut low_prior_high_visit: Vec<_> = options.iter().enumerate()
            .filter(|(_, n)| n.visits > (total_visits / 20))  // At least 5% of visits
            .collect();
        low_prior_high_visit.sort_by(|a, b| {
            let a_ratio = if total_visits > 0 { a.1.visits as f32 / total_visits as f32 } else { 0.0 };
            let b_ratio = if total_visits > 0 { b.1.visits as f32 / total_visits as f32 } else { 0.0 };
            (b_ratio - b.1.prior).partial_cmp(&(a_ratio - a.1.prior)).unwrap_or(std::cmp::Ordering::Equal)
        });
        for (idx, node) in low_prior_high_visit.iter().take(5) {
            let move_str = match side_ref {
                SideReference::SideOne => node.move_choice.to_string(&state.side_one),
                SideReference::SideTwo => node.move_choice.to_string(&state.side_two),
            };
            let visit_pct = if total_visits > 0 {
                (node.visits as f32 / total_visits as f32) * 100.0
            } else {
                0.0
            };
            let _ = writeln!(
                file,
                "{:<4} {:<40} {:>8.4} {:>10} {:>9.2}%",
                idx, move_str, node.prior, node.visits, visit_pct
            );
        }
        
        // For side2, add joint statistics for the most visited action
        if side_ref == SideReference::SideTwo {
            if let Some(root_ptr) = root_node {
                unsafe {
                    let root = &*root_ptr;
                    // Find the most visited side2 action
                    if let Some((top_s2_idx, top_s2_node)) = options.iter().enumerate()
                        .max_by_key(|(_, node)| node.visits)
                        .filter(|(_, node)| node.visits > 0)
                    {
                        let top_s2_move_str = top_s2_node.move_choice.to_string(&state.side_two);
                        
                        // Get side1 options to iterate through
                        if let Some(s1_options) = root.s1_options.as_ref() {
                            let _ = writeln!(file, "\n--- Joint Statistics for Most Visited Side2 Action ---");
                            let _ = writeln!(file, "Side2 Action: {} (idx: {}, visits: {})", 
                                top_s2_move_str, top_s2_idx, top_s2_node.visits);
                            let _ = writeln!(file, "{:<4} {:<40} {:>10} {:>12}", 
                                "Idx", "Side1 Action", "JointVisits", "AvgValue");
                            let _ = writeln!(file, "{}", "-".repeat(70));
                            
                            // For each side1 action, compute joint statistics
                            for (s1_idx, s1_node) in s1_options.iter().enumerate() {
                                let mut joint_visits = 0u32;
                                let mut joint_total_score = 0.0f32;
                                
                                // Look up joint moves (s1_idx, top_s2_idx) in root children
                                if let Some(child_nodes) = root.children.get(&(s1_idx, top_s2_idx)) {
                                    for child in child_nodes.iter() {
                                        joint_visits += child.times_visited;
                                        joint_total_score += child.total_state_score;
                                    }
                                }
                                
                                let avg_value = if joint_visits > 0 {
                                    joint_total_score / joint_visits as f32
                                } else {
                                    0.0
                                };
                                
                                let s1_move_str = s1_node.move_choice.to_string(&state.side_one);
                                let _ = writeln!(file, "{:<4} {:<40} {:>10} {:>12.6}", 
                                    s1_idx, s1_move_str, joint_visits, avg_value);
                            }
                        }
                    }
                }
            }
        }
    }
}

fn transform_eval(eval: &EvalOutcome, root: &EvalOutcome) -> f32 {
    // Both heuristic and neural values are already normalized to [0, 1]
    // No additional transformation needed - use the value directly
    eval.value
}

#[derive(Debug)]
pub struct Node {
    pub root: bool,
    pub parent: *mut Node,
    pub children: HashMap<(usize, usize), Vec<Node>>,
    pub times_visited: u32,
    pub total_state_score: f32,

    // represents the instructions & s1/s2 moves that led to this node from the parent
    pub instructions: StateInstructions,
    pub s1_choice: u8,
    pub s2_choice: u8,

    // represents the total score and number of visits for this node
    // de-coupled for s1 and s2
    pub s1_options: Option<Vec<MoveNode>>,
    pub s2_options: Option<Vec<MoveNode>>,
    policy_priors: Option<Vec<f32>>,
    // Policy priors for side two (opponent) perspective
    s2_policy_priors: Option<Vec<f32>>,
    
    // The raw neural network evaluation when this state was first evaluated
    pub raw_state_value: Option<f32>,
}

#[derive(Clone)]
struct PathStep {
    node: *mut Node,
    s1_choice: usize,
    s2_choice: usize,
}

struct PendingEvaluation {
    path: Vec<PathStep>,
    leaf: *mut Node,
    state: State,
}

impl Node {
    fn new() -> Node {
        Node {
            root: false,
            parent: std::ptr::null_mut(),
            instructions: StateInstructions::default(),
            times_visited: 0,
            total_state_score: 0.0,
            children: HashMap::new(),
            s1_choice: 0,
            s2_choice: 0,
            s1_options: None,
            s2_options: None,
            policy_priors: None,
            s2_policy_priors: None,
            raw_state_value: None,
        }
    }
    unsafe fn populate(
        &mut self,
        state: &State,
        s1_options: Vec<MoveChoice>,
        s2_options: Vec<MoveChoice>,
    ) {
        let mut s1_options_vec: Vec<MoveNode> = s1_options
            .into_iter()
            .map(|x| MoveNode {
                move_choice: x,
                total_score: 0.0,
                visits: 0,
                prior: 0.0,
                virtual_visits: 0.0,
                virtual_score: 0.0,
            })
            .collect();
        let mut s2_options_vec: Vec<MoveNode> = s2_options
            .into_iter()
            .map(|x| MoveNode {
                move_choice: x,
                total_score: 0.0,
                visits: 0,
                prior: 0.0,
                virtual_visits: 0.0,
                virtual_score: 0.0,
            })
            .collect();

        // Reorder options to mirror observation ordering for clearer mapping:
        //  - Moves: alphabetical (by move name) order via active_move_slot_index
        //  - Tera moves: same alphabetical order after normal moves
        //  - Switches: observation slot order via switch_slot_index
        //  - Others/None: last
        fn sort_nodes_for_side(nodes: &mut [MoveNode], state: &State, side_ref: SideReference) {
            nodes.sort_by(|a, b| {
                let rank_a = match a.move_choice {
                    MoveChoice::Move(_) => 0,
                    #[cfg(not(any(feature = "gen1", feature = "gen2", feature = "gen3")))]
                    MoveChoice::MoveTera(_) => 1,
                    #[cfg(not(any(feature = "gen1", feature = "gen2", feature = "gen3")))]
                    MoveChoice::MoveMega(_) => 0,
                    MoveChoice::Switch(_) => 2,
                    MoveChoice::None => 3,
                };
                let rank_b = match b.move_choice {
                    MoveChoice::Move(_) => 0,
                    #[cfg(not(any(feature = "gen1", feature = "gen2", feature = "gen3")))]
                    MoveChoice::MoveTera(_) => 1,
                    #[cfg(not(any(feature = "gen1", feature = "gen2", feature = "gen3")))]
                    MoveChoice::MoveMega(_) => 0,
                    MoveChoice::Switch(_) => 2,
                    MoveChoice::None => 3,
                };
                if rank_a != rank_b {
                    return rank_a.cmp(&rank_b);
                }
                let key_a = match a.move_choice {
                    MoveChoice::Move(mi) => active_move_slot_index(state, side_ref, mi),
                    #[cfg(not(any(feature = "gen1", feature = "gen2", feature = "gen3")))]
                    MoveChoice::MoveTera(mi) => 100 + active_move_slot_index(state, side_ref, mi),
                    #[cfg(not(any(feature = "gen1", feature = "gen2", feature = "gen3")))]
                    MoveChoice::MoveMega(mi) => active_move_slot_index(state, side_ref, mi),
                    MoveChoice::Switch(pi) => switch_slot_index(state, side_ref, pi).unwrap_or(usize::MAX),
                    MoveChoice::None => usize::MAX - 1,
                };
                let key_b = match b.move_choice {
                    MoveChoice::Move(mi) => active_move_slot_index(state, side_ref, mi),
                    #[cfg(not(any(feature = "gen1", feature = "gen2", feature = "gen3")))]
                    MoveChoice::MoveTera(mi) => 100 + active_move_slot_index(state, side_ref, mi),
                    #[cfg(not(any(feature = "gen1", feature = "gen2", feature = "gen3")))]
                    MoveChoice::MoveMega(mi) => active_move_slot_index(state, side_ref, mi),
                    MoveChoice::Switch(pi) => switch_slot_index(state, side_ref, pi).unwrap_or(usize::MAX),
                    MoveChoice::None => usize::MAX - 1,
                };
                key_a.cmp(&key_b)
            });
        }
        sort_nodes_for_side(&mut s1_options_vec, state, SideReference::SideOne);
        sort_nodes_for_side(&mut s2_options_vec, state, SideReference::SideTwo);

        assign_priors_to_move_nodes(
            &mut s1_options_vec,
            state,
            SideReference::SideOne,
            self.policy_priors.as_deref(),
        );
        assign_priors_to_move_nodes(
            &mut s2_options_vec,
            state,
            SideReference::SideTwo,
            self.s2_policy_priors.as_deref(),
        );

        self.s1_options = Some(s1_options_vec);
        self.s2_options = Some(s2_options_vec);
    }

    fn refresh_priors(&mut self, state: &State) {
        if let Some(options) = self.s1_options.as_mut() {
            assign_priors_to_move_nodes(
                options,
                state,
                SideReference::SideOne,
                self.policy_priors.as_deref(),
            );
        }
        if let Some(options) = self.s2_options.as_mut() {
            assign_priors_to_move_nodes(
                options,
                state,
                SideReference::SideTwo,
                self.s2_policy_priors.as_deref(),
            );
        }
    }

    pub fn maximize_ucb_for_side(&self, side_map: &[MoveNode]) -> usize {
        let (mut min_q, mut max_q) = min_max_q(side_map);
        let mut choice = 0;
        let mut best_ucb1 = f32::MIN;
        for (index, node) in side_map.iter().enumerate() {
            let this_ucb1 = node.ucb_value(self.times_visited, min_q, max_q);
            if this_ucb1 > best_ucb1 {
                best_ucb1 = this_ucb1;
                choice = index;
            }
        }
        choice
    }

    pub unsafe fn selection(
        &mut self,
        state: &mut State,
        path: &mut Vec<PathStep>,
    ) -> (*mut Node, usize, usize) {
        let return_node = self as *mut Node;
        if self.s1_options.is_none() {
            let (s1_options, s2_options) = state.get_all_options();
            self.populate(&*state, s1_options, s2_options);
        }

        let s1_mc_index = self.maximize_ucb_for_side(&self.s1_options.as_ref().unwrap());
        let s2_mc_index = self.maximize_ucb_for_side(&self.s2_options.as_ref().unwrap());
        path.push(PathStep {
            node: self as *mut Node,
            s1_choice: s1_mc_index,
            s2_choice: s2_mc_index,
        });
        let child_vector = self.children.get_mut(&(s1_mc_index, s2_mc_index));
        match child_vector {
            Some(child_vector) => {
                let child_vec_ptr = child_vector as *mut Vec<Node>;
                let chosen_child = self.sample_node(child_vec_ptr);
                state.apply_instructions(&(*chosen_child).instructions.instruction_list);
                (*chosen_child).selection(state, path)
            }
            None => (return_node, s1_mc_index, s2_mc_index),
        }
    }

    unsafe fn sample_node(&self, move_vector: *mut Vec<Node>) -> *mut Node {
        let mut rng = rng();
        let weights: Vec<f64> = (*move_vector)
            .iter()
            .map(|x| x.instructions.percentage as f64)
            .collect();
        let dist = WeightedIndex::new(weights).unwrap();
        let chosen_node = &mut (&mut *move_vector)[dist.sample(&mut rng)];
        let chosen_node_ptr = chosen_node as *mut Node;
        chosen_node_ptr
    }

    pub unsafe fn expand(
        &mut self,
        state: &mut State,
        s1_move_index: usize,
        s2_move_index: usize,
    ) -> *mut Node {
        let s1_move = &self.s1_options.as_ref().unwrap()[s1_move_index].move_choice;
        let s2_move = &self.s2_options.as_ref().unwrap()[s2_move_index].move_choice;
        // if the battle is over or both moves are none there is no need to expand
        if (state.battle_is_over() != 0.0 && !self.root)
            || (s1_move == &MoveChoice::None && s2_move == &MoveChoice::None)
        {
            return self as *mut Node;
        }
        let should_branch_on_damage = self.root || (*self.parent).root;
        let mut new_instructions =
            generate_instructions_from_move_pair(state, s1_move, s2_move, should_branch_on_damage);
        let mut this_pair_vec = Vec::with_capacity(new_instructions.len());
        for state_instructions in new_instructions.drain(..) {
            let mut new_node = Node::new();
            new_node.parent = self;
            new_node.instructions = state_instructions;
            new_node.s1_choice = s1_move_index as u8;
            new_node.s2_choice = s2_move_index as u8;

            this_pair_vec.push(new_node);
        }

        // sample a node from the new instruction list.
        // this is the node that the rollout will be done on
        let new_node_ptr = self.sample_node(&mut this_pair_vec);
        state.apply_instructions(&(*new_node_ptr).instructions.instruction_list);
        self.children
            .insert((s1_move_index, s2_move_index), this_pair_vec);
        new_node_ptr
    }

    pub unsafe fn backpropagate(&mut self, score_s1: f32, score_s2: f32, state: &mut State) {
        // State score represents advantage: score_s1 - score_s2
        // Normalize advantage from [-1, 1] to [0, 1] for consistent scaling
        let advantage = score_s1 - score_s2;
        let normalized_advantage = (advantage + 1.0) * 0.5;
        
        self.times_visited += 1;
        self.total_state_score += normalized_advantage;
        if self.root {
            return;
        }

        // Side1 move nodes get normalized advantage (0=side2 wins, 0.5=equal, 1=side1 wins)
        let parent_s1_movenode =
            &mut (*self.parent).s1_options.as_mut().unwrap()[self.s1_choice as usize];
        parent_s1_movenode.total_score += normalized_advantage;
        parent_s1_movenode.visits += 1;

        // Side2 move nodes get inverted normalized advantage (1=side2 wins, 0.5=equal, 0=side1 wins)
        let parent_s2_movenode =
            &mut (*self.parent).s2_options.as_mut().unwrap()[self.s2_choice as usize];
        
        // Depends on how you want to model your opponent
        // parent_s2_movenode.total_score += 1 - normalized_advantage;
        
        parent_s2_movenode.total_score += score_s2;
        parent_s2_movenode.visits += 1;

        state.reverse_instructions(&self.instructions.instruction_list);
        (*self.parent).backpropagate(score_s1, score_s2, state);
    }

    pub fn rollout(&mut self, state: &mut State, root_eval: &EvalOutcome) -> f32 {
        let battle_is_over = state.battle_is_over();
        if battle_is_over == 0.0 {
            // Evaluate from both perspectives so both sides get policy priors
            let eval_s1 = evaluate_with_fallback_for_side(&*state, SideReference::SideOne);
            let eval_s2 = evaluate_with_fallback_for_side(&*state, SideReference::SideTwo);

            self.policy_priors = eval_s1.policy.clone();
            self.s2_policy_priors = eval_s2.policy.clone();
            self.refresh_priors(&*state);

            // Use side-one value for scoring/backprop; store raw for logging
            let transformed = transform_eval(&eval_s1, root_eval);
            self.raw_state_value = Some(eval_s1.value);
            transformed
        } else {
            // Terminal state
            let terminal_score = if battle_is_over == -1.0 { 0.0 } else { battle_is_over };
            self.raw_state_value = Some(terminal_score);
            terminal_score
        }
    }
}

#[derive(Debug)]
pub struct MoveNode {
    pub move_choice: MoveChoice,
    pub total_score: f32,
    pub visits: u32,
    pub prior: f32,
    pub virtual_visits: f32,
    pub virtual_score: f32,
}

#[derive(Clone, Copy)]
struct UcbStats {
    raw_q: Option<f32>,
    scaled_q: Option<f32>,
    u: f32,
    total: f32,
}

impl MoveNode {
    fn ucb_stats(&self, parent_visits: u32, min_q: f32, max_q: f32) -> UcbStats {
        let parent = (parent_visits.max(1)) as f32;
        let c = c_puct(parent);
        let u = c * self.prior * parent.sqrt() / (1.0 + self.visits as f32);
        let effective_visits = self.visits as f32 + self.virtual_visits;
        if effective_visits <= 0.0 {
            return UcbStats {
                raw_q: None,
                scaled_q: None,
                u,
                total: u,
            };
        }

        let raw_q = (self.total_score + self.virtual_score) / effective_visits;
        let diff = (max_q - min_q).abs();
        let mut scaled_q = if diff > 1e-9 {
            (raw_q - min_q) / diff
        } else {
            0.5
        };
        if !scaled_q.is_finite() {
            scaled_q = 0.5;
        } else {
            scaled_q = scaled_q.clamp(0.0, 1.0);
        }
        UcbStats {
            raw_q: Some(raw_q),
            scaled_q: Some(scaled_q),
            u,
            total: scaled_q + u,
        }
    }

    pub fn ucb_value(&self, parent_visits: u32, min_q: f32, max_q: f32) -> f32 {
        self.ucb_stats(parent_visits, min_q, max_q).total
    }
    pub fn average_score(&self) -> f32 {
        let score = self.total_score / self.visits as f32;
        score
    }
}

fn c_puct(parent_visits: f32) -> f32 {
    let numerator = parent_visits + PUCT_C2 + 1.0;
    let ratio = (numerator / PUCT_C2).max(1.0);
    PUCT_C1 + ratio.ln()
}

#[derive(Clone)]
pub struct MctsSideResult {
    pub move_choice: MoveChoice,
    pub total_score: f32,
    pub visits: u32,
}

impl MctsSideResult {
    pub fn average_score(&self) -> f32 {
        if self.visits == 0 {
            return 0.0;
        }
        let score = self.total_score / self.visits as f32;
        score
    }
}

pub struct MctsResult {
    pub s1: Vec<MctsSideResult>,
    pub s2: Vec<MctsSideResult>,
    pub iteration_count: u32,
}

pub fn perform_mcts(
    state: &mut State,
    side_one_options: Vec<MoveChoice>,
    side_two_options: Vec<MoveChoice>,
    max_time: Duration,
) -> MctsResult {
    let turn_start = std::time::Instant::now();
    let mut root_node = Node::new();
    verbose_eprintln!("Side one options: {:?}", side_one_options);
    verbose_eprintln!("Side two options: {:?}", side_two_options);
    
    // Reset timing accumulators for this turn
    neural_evaluate::reset_python_call_stats();
    
    let init_eval_start = std::time::Instant::now();
    // Evaluate first so we have policy priors before initial populate
    let root_eval = evaluate_with_fallback_for_side(state, SideReference::SideOne);
    let root_eval_s2 = evaluate_with_fallback_for_side(state, SideReference::SideTwo);
    root_node.policy_priors = root_eval.policy.clone();
    root_node.s2_policy_priors = root_eval_s2.policy.clone();
    let init_eval_time = init_eval_start.elapsed().as_secs_f64() * 1000.0;
    
    if let Some(policy) = &root_eval.policy {
        verbose_eprintln!("Policy priors: {:?}", policy);
    }
    unsafe {
        root_node.populate(&*state, side_one_options, side_two_options);
    }
    if let Some(ref mut options) = root_node.s1_options {
        apply_dirichlet_noise_to_priors(options);
    }
    if let Some(ref mut options) = root_node.s2_options {
        apply_dirichlet_noise_to_priors(options);
    }
    root_node.root = true;
    let turn_index = LOG_TURN.fetch_add(1, Ordering::Relaxed);
    SWITCH_DEBUG_TURN.store(turn_index, Ordering::Relaxed);
    SWITCH_DEBUG_PRINTED.store(false, Ordering::Relaxed);
    PRIOR_MAP_PRINTED.store(false, Ordering::Relaxed);
    let logging_paths = logging_paths_for_turn(turn_index);
    log_state_value(state, &root_eval, &logging_paths);
    
    // Initialize q/u values file with header
    if let Ok(mut file) = File::create(&logging_paths.q_u_values_path) {
        let _ = writeln!(file, "=== Q/U VALUES BY BATCH FOR TURN {} ===", turn_index);
        let _ = writeln!(file, "This file tracks q (action value) and u (exploration bonus) for each move after each batch.\n");
    }
    if SANITY_CHECK_POLICY_ONLY {
        if let Some(s1_options) = root_node.s1_options.as_mut() {
            for node in s1_options.iter_mut() {
                node.visits = (node.prior * 1000.0) as u32;
                node.total_score = 1.0; // uniform score so visits sort by prior
            }
        }
        if let Some(s2_options) = root_node.s2_options.as_mut() {
            for node in s2_options.iter_mut() {
                node.visits = (node.prior * 1000.0) as u32;
                node.total_score = 1.0;
            }
        }
        root_node.times_visited = 1;
        eprintln!("SANITY CHECK MODE: Using policy priors only, no MCTS rollouts");
    } else {
        let mut pending: Vec<PendingEvaluation> = Vec::new();
        let root_state = state.clone();
        let start_time = std::time::Instant::now();
        let mut batch_count = 0;
        let mut total_collect_time = 0.0;
        let mut total_eval_time = 0.0;
        let mut total_backprop_time = 0.0;
        let mut total_states_evaluated = 0;
        
        while start_time.elapsed() < max_time {
            let batch_start = std::time::Instant::now();
            while pending.len() < BATCH_SIZE && start_time.elapsed() < max_time {
                let mut work_state = root_state.clone();
                let mut path = Vec::new();
                let (selected_node, s1_idx, s2_idx) =
                    unsafe { root_node.selection(&mut work_state, &mut path) };
                let expanded_node = unsafe { (*selected_node).expand(&mut work_state, s1_idx, s2_idx) };

                let terminal = work_state.battle_is_over();
                if terminal != 0.0 {
                    // Terminal state: battle_is_over returns 1.0 if side_one wins, -1.0 if side_two wins
                    // So reward_s1 = 1.0 when side_one wins, 0.0 when side_two wins
                    // And reward_s2 = 0.0 when side_one wins, 1.0 when side_two wins
                    let reward_s1 = if terminal == 1.0 { 1.0 } else { 0.0 };
                    let reward_s2 = if terminal == -1.0 { 1.0 } else { 0.0 };
                    unsafe { (*expanded_node).backpropagate(reward_s1, reward_s2, &mut work_state) };
                    continue;
                }

                // Log the selected moves with their q+u values before applying virtual loss
                unsafe {
                    log_queued_action(
                        &root_node,
                        &root_state,
                        &path,
                        s1_idx,
                        s2_idx,
                        &logging_paths,
                        batch_count + 1,
                    );
                }

                apply_virtual_loss(&path, expanded_node);
                pending.push(PendingEvaluation {
                    path,
                    leaf: expanded_node,
                    state: work_state,
                });

                if root_node.times_visited >= 10_000_000 {
                    break;
                }
            }
            let batch_collect_time = batch_start.elapsed().as_secs_f64() * 1000.0;
            total_collect_time += batch_collect_time;

            if !pending.is_empty() {
                let batch_size = pending.len();
                total_states_evaluated += batch_size;
                let (eval_time, backprop_time) = flush_pending(
                    &mut pending,
                    &root_eval,
                    &root_eval_s2,
                    &mut root_node as *mut Node,
                    &root_state,
                    &logging_paths,
                    batch_count,
                );
                total_eval_time += eval_time;
                total_backprop_time += backprop_time;
                batch_count += 1;
            }

            if root_node.times_visited >= 10_000_000 {
                break;
            }
        }

        if !pending.is_empty() {
            let batch_size = pending.len();
            total_states_evaluated += batch_size;
            let (eval_time, backprop_time) = flush_pending(
                &mut pending,
                &root_eval,
                &root_eval_s2,
                &mut root_node as *mut Node,
                &root_state,
                &logging_paths,
                batch_count,
            );
            total_eval_time += eval_time;
            total_backprop_time += backprop_time;
            batch_count += 1;
        }
        
        let mcts_loop_time = start_time.elapsed().as_secs_f64() * 1000.0;
        let total_inference_time = total_eval_time + total_backprop_time;
        eprintln!("[MCTS_TIMING] batches={} states_eval={} visits={} collect={:.1}ms eval={:.1}ms backprop={:.1}ms inference={:.1}ms loop={:.1}ms", 
            batch_count, total_states_evaluated, root_node.times_visited, 
            total_collect_time, total_eval_time, total_backprop_time, total_inference_time, mcts_loop_time);
    }
    
    eprintln!("Iterations {}: {}", turn_index, root_node.times_visited);
    
    // Log aggregated Python call timing
    neural_evaluate::log_python_call_stats();

    let raw_value_samples = drain_raw_value_samples();
    log_value_scale_samples(&logging_paths.value_scale_path, &raw_value_samples);
    
    // (Sanity-check seeding handled earlier; no rollouts performed in that mode.)

    let _visits = root_node.times_visited;
    let source_label = match root_eval.source {
        ValueSource::Neural => "neural",
        ValueSource::Heuristic => "heuristic",
    };
    verbose_eprintln!(
        "MCTS root eval: {:.3} ({}) | iterations: {} (policy only)",
        root_eval.value, source_label, root_node.times_visited
    );

    if let Some(options) = root_node.s1_options.as_ref() {
        let tree_stats = collect_root_stats(options);
        // tree_stats.log_summary();
        tree_stats.write_json(&logging_paths.stats_path);
        let mut state_for_logging = state.clone();
        dump_tree_json(&root_node, &mut state_for_logging, &logging_paths.tree_path);
        log_policy_priors_comparison(
            &state,
            options,
            SideReference::SideOne,
            &logging_paths.comparison_path,
            root_node.policy_priors.as_deref(),
            None, // root_node not needed for side1
        );

        // Side two comparison uses the opponent options present at root
        if let Some(options_s2) = root_node.s2_options.as_ref() {
            log_policy_priors_comparison(
                &state,
                options_s2,
                SideReference::SideTwo,
                &logging_paths.comparison_path_side2,
                root_node.s2_policy_priors.as_deref(),
                Some(&root_node as *const Node), // Pass root_node for joint statistics
            );
        }

        let mut ranked: Vec<_> = options.iter().enumerate().collect();
        ranked.sort_by(|a, b| b.1.visits.cmp(&a.1.visits));
        
        // Print selected action (highest visits)
        if let Some((selected_idx, selected_node)) = ranked.first() {
            let selected_move_str = selected_node.move_choice.to_string(&state.side_one);
            verbose_eprintln!(
                "╔════════════════════════════════════════════════════════════════╗"
            );
            verbose_eprintln!(
                "║ MCTS SELECTED ACTION (by visits)                              ║"
            );
            verbose_eprintln!(
                "╠════════════════════════════════════════════════════════════════╣"
            );
            verbose_eprintln!("║ Move: {:48} ║", selected_move_str);
            verbose_eprintln!(
                "║ Index: {:2}  Visits: {:6}  Avg: {:.3}  Prior: {:.3}          ║",
                selected_idx,
                selected_node.visits,
                if selected_node.visits > 0 {
                    selected_node.total_score / selected_node.visits as f32
                } else {
                    0.0
                },
                selected_node.prior
            );
            verbose_eprintln!(
                "╚════════════════════════════════════════════════════════════════╝"
            );
        }

        verbose_eprintln!("Top move candidates:");
        for (rank, (index, move_node)) in ranked.into_iter().take(3).enumerate() {
            let avg = if move_node.visits == 0 {
                0.0
            } else {
                move_node.total_score / move_node.visits as f32
            };
            let move_label = format!("{:?}", move_node.move_choice);
            verbose_eprintln!(
                "  {}. {} | idx {} | visits: {} | avg: {:.3} | score: {:.3} | prior: {:.3}",
                rank + 1,
                move_label,
                index,
                move_node.visits,
                avg,
                move_node.total_score,
                move_node.prior
            );
        }
    }

    let result = MctsResult {
        s1: root_node
            .s1_options
            .as_ref()
            .unwrap()
            .iter()
            .map(|v| MctsSideResult {
                move_choice: v.move_choice.clone(),
                total_score: v.total_score,
                visits: v.visits,
            })
            .collect(),
        s2: root_node
            .s2_options
            .as_ref()
            .unwrap()
            .iter()
            .map(|v| MctsSideResult {
                move_choice: v.move_choice.clone(),
                total_score: v.total_score,
                visits: v.visits,
            })
            .collect(),
        iteration_count: root_node.times_visited,
    };

    let total_turn_time = turn_start.elapsed().as_secs_f64() * 1000.0;
    eprintln!("[TURN_TIMING] turn={} init_eval={:.1}ms total={:.1}ms visits={}", 
        turn_index, init_eval_time, total_turn_time, root_node.times_visited);

    result
}
fn dump_tree_json(root: &Node, state: &mut State, path: &str) {
    if let Ok(mut file) = File::create(path) {
        let _ = writeln!(file, "total_iterations: {}", root.times_visited);
        let _ = writeln!(
            file,
            "{{depth:0, description:\"root\", node_visits:{}}}",
            root.times_visited
        );
        unsafe { dump_node_recursive(root, state, String::new(), 1, &mut file) };
    }
}

fn apply_virtual_loss(path: &[PathStep], leaf: *mut Node) {
    for step in path {
        unsafe {
            let node = &mut *step.node;
            node.times_visited += 1;
            if let Some(s1_opts) = node.s1_options.as_mut() {
                if let Some(choice) = s1_opts.get_mut(step.s1_choice) {
                    choice.visits = choice.visits.saturating_add(1);
                }
            }
            if let Some(s2_opts) = node.s2_options.as_mut() {
                if let Some(choice) = s2_opts.get_mut(step.s2_choice) {
                    choice.visits = choice.visits.saturating_add(1);
                }
            }
        }
    }
    unsafe {
        (*leaf).times_visited += 1;
    }
}

fn revert_virtual_loss(path: &[PathStep], leaf: *mut Node) {
    for step in path.iter().rev() {
        unsafe {
            let node = &mut *step.node;
            node.times_visited = node.times_visited.saturating_sub(1);
            if let Some(s1_opts) = node.s1_options.as_mut() {
                if let Some(choice) = s1_opts.get_mut(step.s1_choice) {
                    choice.visits = choice.visits.saturating_sub(1);
                }
            }
            if let Some(s2_opts) = node.s2_options.as_mut() {
                if let Some(choice) = s2_opts.get_mut(step.s2_choice) {
                    choice.visits = choice.visits.saturating_sub(1);
                }
            }
        }
    }
    unsafe {
        (*leaf).times_visited = (*leaf).times_visited.saturating_sub(1);
    }
}

fn flush_pending(
    pending: &mut Vec<PendingEvaluation>,
    root_eval: &EvalOutcome,
    root_eval_s2: &EvalOutcome,
    root_node: *mut Node,
    state: &State,
    logging_paths: &LoggingPaths,
    batch_num: usize,
) -> (f64, f64) {
    if pending.is_empty() {
        return (0.0, 0.0);
    }
    let mut entries = Vec::new();
    entries.append(pending);
    let state_refs: Vec<&State> = entries.iter().map(|entry| &entry.state).collect();

    let eval_start = std::time::Instant::now();
    let evals_s1 = evaluate_with_fallback_batch_for_side(&state_refs, SideReference::SideOne);
    let evals_s2 = evaluate_with_fallback_batch_for_side(&state_refs, SideReference::SideTwo);
    let eval_time = eval_start.elapsed().as_secs_f64() * 1000.0;

    let backprop_start = std::time::Instant::now();
    for (entry, (eval_s1, eval_s2)) in entries
        .into_iter()
        .zip(evals_s1.into_iter().zip(evals_s2.into_iter()))
    {
        let score_s1 = transform_eval(&eval_s1, root_eval);
        let score_s2 = transform_eval(&eval_s2, root_eval_s2);
        revert_virtual_loss(&entry.path, entry.leaf);
        let mut state_for_backprop = entry.state;
        unsafe {
            // Store the raw state value before backpropagating (use side1's value for backward compatibility)
            (*entry.leaf).raw_state_value = Some(eval_s1.value);
            (*entry.leaf).policy_priors = eval_s1.policy.clone();
            (*entry.leaf).s2_policy_priors = eval_s2.policy.clone();
            (*entry.leaf).refresh_priors(&state_for_backprop);
            (*entry.leaf).backpropagate(score_s1, score_s2, &mut state_for_backprop);
        }
    }
    let backprop_time = backprop_start.elapsed().as_secs_f64() * 1000.0;

    // Collect q/u values from root node after this batch
    unsafe {
        log_q_u_values_by_batch(root_node, state, logging_paths, batch_num);
    }

    (eval_time, backprop_time)
}

fn min_max_q(nodes: &[MoveNode]) -> (f32, f32) {
    let mut min_q = f32::MAX;
    let mut max_q = f32::MIN;
    for node in nodes.iter() {
        let effective_visits = node.visits as f32 + node.virtual_visits;
        if effective_visits > 0.0 {
            let q = (node.total_score + node.virtual_score) / effective_visits;
            if q < min_q {
                min_q = q;
            }
            if q > max_q {
                max_q = q;
            }
        }
    }
    if min_q == f32::MAX {
        min_q = 0.0;
    }
    if max_q == f32::MIN {
        max_q = 1.0;
    }
    (min_q, max_q)
}

fn format_optional(value: Option<f32>) -> String {
    match value {
        Some(v) => format!("{:.6}", v),
        None => "pending".to_string(),
    }
}

fn log_queued_action(
    root_node: *const Node,
    state: &State,
    path: &[PathStep],
    s1_idx: usize,
    s2_idx: usize,
    logging_paths: &LoggingPaths,
    batch_num: usize,
) {
    unsafe {
        if let Ok(mut file) = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&logging_paths.q_u_values_path)
        {
            let parent_visits = (*root_node).times_visited;

            // Extract root action from path (first step)
            let root_s1_idx = path.first().map(|step| step.s1_choice).unwrap_or(s1_idx);
            let root_s2_idx = path.first().map(|step| step.s2_choice).unwrap_or(s2_idx);

            if let (Some(s1_options), Some(s2_options)) = ((*root_node).s1_options.as_ref(), (*root_node).s2_options.as_ref()) {
                if let (Some(s1_node), Some(s2_node), Some(root_s1_node), Some(root_s2_node)) = (
                    s1_options.get(s1_idx),
                    s2_options.get(s2_idx),
                    s1_options.get(root_s1_idx),
                    s2_options.get(root_s2_idx),
                ) {
                    let (min_q_s1, max_q_s1) = min_max_q(s1_options);
                    let (min_q_s2, max_q_s2) = min_max_q(s2_options);

                    let root_s1_stats = root_s1_node.ucb_stats(parent_visits, min_q_s1, max_q_s1);
                    let root_s2_stats = root_s2_node.ucb_stats(parent_visits, min_q_s2, max_q_s2);
                    let s1_stats = s1_node.ucb_stats(parent_visits, min_q_s1, max_q_s1);
                    let s2_stats = s2_node.ucb_stats(parent_visits, min_q_s2, max_q_s2);

                    let s1_move_str = s1_node.move_choice.to_string(&state.side_one);
                    let s2_move_str = s2_node.move_choice.to_string(&state.side_two);

                    let _ = writeln!(file, "
--- Queued for Batch {} ---", batch_num);
                    let _ = writeln!(
                        file,
                        "Side1: {} (idx: {}) | q_raw: {}, q_scaled: {}, u: {:.6}, q+u: {:.6}",
                        s1_move_str,
                        s1_idx,
                        format_optional(s1_stats.raw_q),
                        format_optional(s1_stats.scaled_q),
                        s1_stats.u,
                        s1_stats.total
                    );
                    let _ = writeln!(
                        file,
                        "Side2: {} (idx: {}) | q_raw: {}, q_scaled: {}, u: {:.6}, q+u: {:.6}",
                        s2_move_str,
                        s2_idx,
                        format_optional(s2_stats.raw_q),
                        format_optional(s2_stats.scaled_q),
                        s2_stats.u,
                        s2_stats.total
                    );

                    let root_s1_move = root_s1_node.move_choice.to_string(&state.side_one);
                    let root_s2_move = root_s2_node.move_choice.to_string(&state.side_two);
                    let _ = writeln!(
                        file,
                        "Current Selection (root) | Side1: {} (idx: {}) | q_raw: {}, q_scaled: {}, u: {:.6}, q+u: {:.6} | Side2: {} (idx: {}) | q_raw: {}, q_scaled: {}, u: {:.6}, q+u: {:.6}",
                        root_s1_move,
                        root_s1_idx,
                        format_optional(root_s1_stats.raw_q),
                        format_optional(root_s1_stats.scaled_q),
                        root_s1_stats.u,
                        root_s1_stats.total,
                        root_s2_move,
                        root_s2_idx,
                        format_optional(root_s2_stats.raw_q),
                        format_optional(root_s2_stats.scaled_q),
                        root_s2_stats.u,
                        root_s2_stats.total
                    );
                }
            }
        }
    }
}

fn log_q_u_values_by_batch(
    root_node: *mut Node,
    state: &State,
    logging_paths: &LoggingPaths,
    batch_num: usize,
) {
    unsafe {
        if let Ok(mut file) = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&logging_paths.q_u_values_path)
        {
            let _ = writeln!(file, "\n=== BATCH {} ===", batch_num);
            let _ = writeln!(file, "Total visits: {}", (*root_node).times_visited);
            
            let parent_visits = (*root_node).times_visited;
            
            // Find and display root actions (highest q+u for each side)
            if let (Some(s1_options), Some(s2_options)) = ((*root_node).s1_options.as_ref(), (*root_node).s2_options.as_ref()) {
                let (min_q_s1, max_q_s1) = min_max_q(s1_options);
                let (min_q_s2, max_q_s2) = min_max_q(s2_options);
                
                // Find root action for side1 (highest q+u)
                let mut best_s1_idx = 0;
                let mut best_s1_q_plus_u = f32::MIN;
                for (idx, move_node) in s1_options.iter().enumerate() {
                    let stats = move_node.ucb_stats(parent_visits, min_q_s1, max_q_s1);
                    let q_plus_u = stats.total;
                    if q_plus_u > best_s1_q_plus_u {
                        best_s1_q_plus_u = q_plus_u;
                        best_s1_idx = idx;
                    }
                }
                
                // Find root action for side2 (highest q+u)
                let mut best_s2_idx = 0;
                let mut best_s2_q_plus_u = f32::MIN;
                for (idx, move_node) in s2_options.iter().enumerate() {
                    let stats = move_node.ucb_stats(parent_visits, min_q_s2, max_q_s2);
                    let q_plus_u = stats.total;
                    if q_plus_u > best_s2_q_plus_u {
                        best_s2_q_plus_u = q_plus_u;
                        best_s2_idx = idx;
                    }
                }
                
                // Display root actions
                if let (Some(best_s1_node), Some(best_s2_node)) = (s1_options.get(best_s1_idx), s2_options.get(best_s2_idx)) {
                    let s1_stats = best_s1_node.ucb_stats(parent_visits, min_q_s1, max_q_s1);
                    let s2_stats = best_s2_node.ucb_stats(parent_visits, min_q_s2, max_q_s2);
                    let s1_move_str = best_s1_node.move_choice.to_string(&state.side_one);
                    let s2_move_str = best_s2_node.move_choice.to_string(&state.side_two);
                    
                    let _ = writeln!(file, "\n--- Root Actions (highest q+u) ---");
                    let _ = writeln!(file, "Side1: {} (idx: {}) | q_raw: {:.6}, q_scaled: {:.6}, u: {:.6}, q+u: {:.6}, visits: {}", 
                        s1_move_str, 
                        best_s1_idx,
                        s1_stats.raw_q.unwrap_or(0.0),
                        s1_stats.scaled_q.unwrap_or(0.5),
                        s1_stats.u,
                        s1_stats.total,
                        best_s1_node.visits
                    );
                    let _ = writeln!(file, "Side2: {} (idx: {}) | q_raw: {:.6}, q_scaled: {:.6}, u: {:.6}, q+u: {:.6}, visits: {}", 
                        s2_move_str, 
                        best_s2_idx,
                        s2_stats.raw_q.unwrap_or(0.0),
                        s2_stats.scaled_q.unwrap_or(0.5),
                        s2_stats.u,
                        s2_stats.total,
                        best_s2_node.visits
                    );
                }
            }
            
            // Collect statistics across all moves
            let mut all_q_values: Vec<f32> = Vec::new();
            let mut all_u_values: Vec<f32> = Vec::new();
            let mut all_q_scaled_values: Vec<f32> = Vec::new();
            let mut u_max = f32::MIN;
            let mut visited_children = 0;
            let mut total_children = 0;
            
            // Collect S1 values
            if let Some(s1_options) = (*root_node).s1_options.as_ref() {
                let (min_q_s1, max_q_s1) = min_max_q(s1_options);
                total_children += s1_options.len();

                let _ = writeln!(file, "\n--- Side One Moves ---");
                let _ = writeln!(file, "{:<6} {:<30} {:>10} {:>12} {:>10} {:>8} {:>12} {:>10} {:>12} {:>10}",
                    "Idx", "Move", "q", "q_scaled", "u", "Visits", "TotalScore", "Prior", "ParentVisits", "Q+U");
                let _ = writeln!(file, "{}", "-".repeat(132));
                for (idx, move_node) in s1_options.iter().enumerate() {
                    if move_node.visits > 0 {
                        visited_children += 1;
                        let stats = move_node.ucb_stats(parent_visits, min_q_s1, max_q_s1);
                        let raw_q = stats.raw_q.unwrap_or(0.0);
                        let scaled_q = stats.scaled_q.unwrap_or(0.5);
                        let u = stats.u;
                        let move_str = move_node.move_choice.to_string(&state.side_one);
                        let _ = writeln!(file, "{:<6} {:<30} {:>10.6} {:>12.6} {:>10.6} {:>8} {:>12.6} {:>10.6} {:>12} {:>10.6}",
                            idx, move_str, raw_q, scaled_q, u, move_node.visits, move_node.total_score, move_node.prior, parent_visits, scaled_q + u);

                        // Collect for statistics
                        all_q_values.push(raw_q);
                        all_u_values.push(u);
                        all_q_scaled_values.push(scaled_q);
                        if u > u_max {
                            u_max = u;
                        }
                    }
                }
            }

            // Collect S2 values
            if let Some(s2_options) = (*root_node).s2_options.as_ref() {
                let (min_q_s2, max_q_s2) = min_max_q(s2_options);
                total_children += s2_options.len();

                let _ = writeln!(file, "\n--- Side Two Moves ---");
                let _ = writeln!(file, "{:<6} {:<30} {:>10} {:>12} {:>10} {:>8} {:>12} {:>10} {:>12} {:>10}",
                    "Idx", "Move", "q", "q_scaled", "u", "Visits", "TotalScore", "Prior", "ParentVisits", "Q+U");
                let _ = writeln!(file, "{}", "-".repeat(132));
                for (idx, move_node) in s2_options.iter().enumerate() {
                    if move_node.visits > 0 {
                        visited_children += 1;
                        let stats = move_node.ucb_stats(parent_visits, min_q_s2, max_q_s2);
                        let raw_q = stats.raw_q.unwrap_or(0.0);
                        let scaled_q = stats.scaled_q.unwrap_or(0.5);
                        let u = stats.u;
                        let move_str = move_node.move_choice.to_string(&state.side_two);
                        let _ = writeln!(file, "{:<6} {:<30} {:>10.6} {:>12.6} {:>10.6} {:>8} {:>12.6} {:>10.6} {:>12} {:>10.6}",
                            idx, move_str, raw_q, scaled_q, u, move_node.visits, move_node.total_score, move_node.prior, parent_visits, scaled_q + u);

                        // Collect for statistics
                        all_q_values.push(raw_q);
                        all_u_values.push(u);
                        all_q_scaled_values.push(scaled_q);
                        if u > u_max {
                            u_max = u;
                        }
                    }
                }
            }
            
            // Compute batch statistics
            let _ = writeln!(file, "\n--- Batch Statistics ---");
            
            // Compute q_span (max_q - min_q over visited children)
            let q_span = if !all_q_values.is_empty() {
                let min_q_all = all_q_values.iter().fold(f32::MAX, |a, &b| a.min(b));
                let max_q_all = all_q_values.iter().fold(f32::MIN, |a, &b| a.max(b));
                max_q_all - min_q_all
            } else {
                0.0
            };
            
            // Compute means and standard deviations
            let mean_q = if !all_q_values.is_empty() {
                all_q_values.iter().sum::<f32>() / all_q_values.len() as f32
            } else {
                0.0
            };
            let std_q = if all_q_values.len() > 1 {
                let variance = all_q_values.iter()
                    .map(|&x| (x - mean_q) * (x - mean_q))
                    .sum::<f32>() / all_q_values.len() as f32;
                variance.sqrt()
            } else {
                0.0
            };
            
            let mean_u = if !all_u_values.is_empty() {
                all_u_values.iter().sum::<f32>() / all_u_values.len() as f32
            } else {
                0.0
            };
            let std_u = if all_u_values.len() > 1 {
                let variance = all_u_values.iter()
                    .map(|&x| (x - mean_u) * (x - mean_u))
                    .sum::<f32>() / all_u_values.len() as f32;
                variance.sqrt()
            } else {
                0.0
            };
            
            let mean_q_scaled = if !all_q_scaled_values.is_empty() {
                all_q_scaled_values.iter().sum::<f32>() / all_q_scaled_values.len() as f32
            } else {
                0.0
            };
            let std_q_scaled = if all_q_scaled_values.len() > 1 {
                let variance = all_q_scaled_values.iter()
                    .map(|&x| (x - mean_q_scaled) * (x - mean_q_scaled))
                    .sum::<f32>() / all_q_scaled_values.len() as f32;
                variance.sqrt()
            } else {
                0.0
            };
            
            let visited_frac = if total_children > 0 {
                visited_children as f32 / total_children as f32
            } else {
                0.0
            };
            
            // Handle u_max when no visited children
            let u_max_display = if u_max == f32::MIN { 0.0 } else { u_max };
            
            let _ = writeln!(file, "Statistics across all visited moves:");
            let _ = writeln!(file, "  q:        mean = {:.6}, std = {:.6}", mean_q, std_q);
            let _ = writeln!(file, "  u:        mean = {:.6}, std = {:.6}", mean_u, std_u);
            let _ = writeln!(file, "  q_scaled: mean = {:.6}, std = {:.6}", mean_q_scaled, std_q_scaled);
            let _ = writeln!(file, "  q_span:   {:.6} (max_q - min_q over visited children)", q_span);
            let _ = writeln!(file, "  u_max:    {:.6}", u_max_display);
            let _ = writeln!(file, "  visited_frac: {:.6} ({} / {})", visited_frac, visited_children, total_children);
        }
    }
}

fn log_value_scale_samples(path: &str, samples: &[f32]) {
    if samples.is_empty() {
        return;
    }
    let existed = std::fs::metadata(path).is_ok();
    if let Ok(mut file) = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        if !existed {
            let _ = writeln!(file, "raw_value");
        }
        for sample in samples {
            let _ = writeln!(file, "{:.6}", sample);
        }
    }
}

unsafe fn dump_node_recursive(
    node: &Node,
    state: &mut State,
    prefix: String,
    depth: usize,
    file: &mut File,
) {
    if node.children.is_empty() {
        return;
    }

    let s1_opts = match node.s1_options.as_ref() {
        Some(opts) => opts,
        None => return,
    };
    let s2_opts = match node.s2_options.as_ref() {
        Some(opts) => opts,
        None => return,
    };

    let mut entries: Vec<(usize, usize, &Node)> = Vec::new();
    for ((s1_idx, s2_idx), child_nodes) in node.children.iter() {
        for child in child_nodes.iter() {
            entries.push((*s1_idx, *s2_idx, child));
        }
    }

    // Sort by visits (descending) so most visited nodes appear first at each level
    entries.sort_by(|a, b| b.2.times_visited.cmp(&a.2.times_visited));

    let total = entries.len();
    for (idx, (s1_idx, s2_idx, child)) in entries.into_iter().enumerate() {
        let is_last = idx + 1 == total;
        let s1_choice = match s1_opts.get(s1_idx) {
            Some(choice) => choice,
            None => continue,
        };
        let s2_choice = match s2_opts.get(s2_idx) {
            Some(choice) => choice,
            None => continue,
        };
        let s1_move_label = s1_choice.move_choice.to_string(&state.side_one);
        let s2_move_label = s2_choice.move_choice.to_string(&state.side_two);
        let s1_avg = if s1_choice.visits == 0 {
            0.0
        } else {
            s1_choice.total_score / s1_choice.visits as f32
        };
        let s2_avg = if s2_choice.visits == 0 {
            0.0
        } else {
            s2_choice.total_score / s2_choice.visits as f32
        };
        let state_avg = if child.times_visited == 0 {
            0.0
        } else {
            child.total_state_score / child.times_visited as f32
        };
        // raw_state_value is only set when the node is directly evaluated as a leaf.
        // For intermediate nodes that accumulated visits via backpropagation, use state_avg_score.
        let state_score_str = match child.raw_state_value {
            Some(val) => format!("{:.6}", val),
            None => {
                if child.times_visited > 0 {
                    format!("{:.6} (avg)", state_avg)
                } else {
                    "null".to_string()
                }
            }
        };
        let node_line = format!(
            "{{depth:{}, s1_move:\"{}\", s2_move:\"{}\", move_visits:{}, state_visits:{}, s1_avg_score:{:.6}, s2_avg_score:{:.6}, state_score:{}, state_avg_score:{:.6}}}",
            depth,
            s1_move_label,
            s2_move_label,
            s1_choice.visits,
            child.times_visited,
            s1_avg,
            s2_avg,
            state_score_str,
            state_avg
        );
        let branch = prefix.clone();
        let connector = if is_last { "`- " } else { "|- " };
        let _ = writeln!(file, "{}{}{}", branch, connector, node_line);

        let mut next_prefix = prefix.clone();
        next_prefix.push_str(if is_last { "   " } else { "|  " });
        // Apply this child's instructions to advance state for deeper logging
        state.apply_instructions(&child.instructions.instruction_list);
        dump_node_recursive(child, state, next_prefix, depth + 1, file);
        // Revert to the parent's state after returning
        state.reverse_instructions(&child.instructions.instruction_list);
    }
}

fn assign_priors_to_move_nodes(
    move_nodes: &mut [MoveNode],
    state: &State,
    side_ref: SideReference,
    policy: Option<&[f32]>,
) {
    if move_nodes.is_empty() {
        return;
    }
    // Map choices to raw policy values (pre-normalization)
    let priors_raw: Vec<f32> = if policy.is_some() {
        move_nodes
            .iter()
            .map(|node| match node.move_choice {
                MoveChoice::None => 0.0,
                _ => choice_policy_value(state, side_ref, &node.move_choice, policy),
            })
            .collect()
    } else {
        // No policy yet (e.g., initial populate for non-root nodes). Use zeros; will be refreshed later.
        vec![0.0; move_nodes.len()]
    };
    let mut priors = priors_raw.clone();
    if !SANITY_CHECK_POLICY_ONLY {
        normalize_priors(&mut priors);
    }

    // One-time per turn debug dump of how policy indices map to options
    // if let Some(policy_vec) = policy {
    //     if side_ref == SideReference::SideOne {
    //         let turn = SWITCH_DEBUG_TURN.load(Ordering::Relaxed);
    //         if !PRIOR_MAP_PRINTED.swap(true, Ordering::Relaxed) {
    //             eprintln!("[assign_priors][turn {}] side={:?} options={} (policy len={})",
    //                 turn, side_ref, move_nodes.len(), policy_vec.len());
    //             // Print switch policy slice 4..9 when available
    //             if policy_vec.len() >= 9 {
    //                 let sw = &policy_vec[4..9];
    //                 eprintln!("[assign_priors][turn {}] switch policy slice [4..9): [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
    //                     turn, sw[0], sw[1], sw[2], sw[3], sw[4]);
    //             }
    //             for (i, node) in move_nodes.iter().enumerate() {
    //                 match &node.move_choice {
    //                     MoveChoice::Switch(pidx) => {
    //                         let team_idx = pokemon_index_to_usize(*pidx);
    //                         let slot_opt = switch_slot_index(state, side_ref, *pidx);
    //                         let action_idx = slot_opt.map(|s| 4 + s);
    //                         let pval = action_idx
    //                             .and_then(|ai| policy_vec.get(ai).copied())
    //                             .unwrap_or(0.0);
    //                         let side_state = match side_ref {
    //                             SideReference::SideOne => &state.side_one,
    //                             SideReference::SideTwo => &state.side_two,
    //                         };
    //                         let name = format!("{:?}", side_state.pokemon.pkmn[team_idx].id).to_lowercase();
    //                         eprintln!(
    //                             "[assign_priors][turn {}] opt={} SWITCH team={} name={} obs_slot={:?} action_idx={:?} policy_val={:.4} prior_raw={:.4} prior_norm={:.4}",
    //                             turn, i, team_idx, name, slot_opt, action_idx, pval, priors_raw[i], priors[i]
    //                         );
    //                     }
    //                     MoveChoice::Move(midx) => {
    //                         let mi = move_index_to_usize(*midx);
    //                         let pval = policy_vec.get(mi).copied().unwrap_or(0.0);
    //                         eprintln!(
    //                             "[assign_priors][turn {}] opt={} MOVE m{} action_idx={} policy_val={:.4} prior_raw={:.4} prior_norm={:.4}",
    //                             turn, i, mi, mi, pval, priors_raw[i], priors[i]
    //                         );
    //                     }
    //                     _ => {
    //                         eprintln!(
    //                             "[assign_priors][turn {}] opt={} OTHER prior_raw={:.4} prior_norm={:.4}",
    //                             turn, i, priors_raw[i], priors[i]
    //                         );
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }
    for (node, prior) in move_nodes.iter_mut().zip(priors.into_iter()) {
        node.prior = prior;
    }

    apply_virtual_priors(move_nodes);
}

fn apply_virtual_priors(move_nodes: &mut [MoveNode]) {
    if VIRTUAL_PRIOR_STRENGTH <= 0.0 {
        for node in move_nodes.iter_mut() {
            node.virtual_visits = 0.0;
            node.virtual_score = 0.0;
        }
        return;
    }

    for node in move_nodes.iter_mut() {
        if node.prior > 0.0 {
            let virtual_visits = node.prior * VIRTUAL_PRIOR_STRENGTH;
            node.virtual_visits = virtual_visits;
            node.virtual_score = virtual_visits * VIRTUAL_PRIOR_BASELINE;
        } else {
            node.virtual_visits = 0.0;
            node.virtual_score = 0.0;
        }
    }
}

fn choice_policy_value(
    state: &State,
    side_ref: SideReference,
    choice: &MoveChoice,
    policy: Option<&[f32]>,
) -> f32 {
    let policy = policy.unwrap_or_else(|| {
        eprintln!(
            "[FATAL] choice_policy_value: missing policy slice for side={:?} choice={:?}",
            side_ref, choice
        );
        panic!("policy slice missing");
    });

    let index = action_index_for_choice(state, side_ref, choice).unwrap_or_else(|| {
        eprintln!(
            "[FATAL] choice_policy_value: failed to compute action index for side={:?} choice={:?}",
            side_ref, choice
        );
        panic!("action index missing");
    });

    if index >= policy.len() {
        eprintln!(
            "[FATAL] choice_policy_value: action index out of bounds: idx={} policy_len={} side={:?} choice={:?}",
            index,
            policy.len(),
            side_ref,
            choice
        );
        panic!("policy index OOB");
    }

    policy[index]
}

fn action_index_for_choice(
    state: &State,
    side_ref: SideReference,
    choice: &MoveChoice,
) -> Option<usize> {
    match choice {
        // Map moves by the same alphabetical ordering used in observation
        MoveChoice::Move(idx) => Some(active_move_slot_index(state, side_ref, *idx)),
        #[cfg(not(any(feature = "gen1", feature = "gen2", feature = "gen3")))]
        MoveChoice::MoveTera(idx) => Some(9 + active_move_slot_index(state, side_ref, *idx)),
        #[cfg(not(any(feature = "gen1", feature = "gen2", feature = "gen3")))]
        MoveChoice::MoveMega(idx) => Some(active_move_slot_index(state, side_ref, *idx)),
        MoveChoice::Switch(pokemon_index) =>
            switch_slot_index(state, side_ref, *pokemon_index).map(|slot| 4 + slot),
        MoveChoice::None => None,
    }
}

// fn switch_slot_index(state: &State, side_ref: SideReference, target: PokemonIndex) -> Option<usize> {
//     let side = match side_ref {
//         SideReference::SideOne => &state.side_one,
//         SideReference::SideTwo => &state.side_two,
//     };
//     let target_idx = pokemon_index_to_usize(target);
//     let active_idx = pokemon_index_to_usize(side.active_index);
//     if target_idx == active_idx {
//         return None;
//     }
//     let mut slot = 0;
//     for (idx, pokemon) in side.pokemon.pkmn.iter().enumerate() {
//         if idx == active_idx || pokemon.hp <= 0 {
//             continue;
//         }
//         if idx == target_idx {
//             return Some(slot);
//         }
//         slot += 1;
//     }
//     None
// }
fn switch_slot_index(state: &State, side_ref: SideReference, target: PokemonIndex) -> Option<usize> {
    let side = match side_ref {
        SideReference::SideOne => &state.side_one,
        SideReference::SideTwo => &state.side_two,
    };

    let target_idx = pokemon_index_to_usize(target);
    let active_idx = pokemon_index_to_usize(side.active_index);

    // Team preview: allow selecting any team slot (including current active_idx placeholder)
    if state.team_preview {
        let mut switches: Vec<(usize, &crate::state::Pokemon)> = side
            .pokemon
            .pkmn
            .iter()
            .enumerate()
            .collect();
        switches.sort_by(|(idx_a, p_a), (idx_b, p_b)| {
            let alive_a = p_a.hp > 0;
            let alive_b = p_b.hp > 0;
            match alive_b.cmp(&alive_a) {
                std::cmp::Ordering::Equal => {
                    let name_a = format!("{:?}", p_a.id).to_lowercase();
                    let name_b = format!("{:?}", p_b.id).to_lowercase();
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
        return switches.iter().position(|(idx, _)| *idx == target_idx);
    }

    let _should_log = !SWITCH_DEBUG_PRINTED.swap(true, Ordering::Relaxed);
    let _current_turn = SWITCH_DEBUG_TURN.load(Ordering::Relaxed);

    // if should_log {
    //     eprintln!(
    //         "[switch_slot_index][turn {}] side={:?} active_idx={} target_idx={}",
    //         current_turn, side_ref, active_idx, target_idx
    //     );
    // }

    if target_idx == active_idx {
        // if should_log {
        //     eprintln!(
        //         "[switch_slot_index][turn {}] target_idx == active_idx ({}). No switch possible.",
        //         current_turn, target_idx
        //     );
        // }
        return None;
    }

    // Collect candidates excluding the active slot.
    let mut switches: Vec<(usize, &crate::state::Pokemon)> = side
        .pokemon
        .pkmn
        .iter()
        .enumerate()
        .filter(|(idx, _)| *idx != active_idx)
        .collect();

    // if should_log {
    //     // Log BEFORE sort (index, id, hp, alive?)
    //     let before: Vec<String> = switches
    //         .iter()
    //         .map(|(idx, p)| format!("#{} id={:?} hp={} alive={}", idx, p.id, p.hp, p.hp > 0))
    //         .collect();
    //     eprintln!(
    //         "[switch_slot_index][turn {}] candidates BEFORE sort: [{}]",
    //         current_turn,
    //         before.join(", ")
    //     );
    // }

    // Sort: alive first, then name (case-insensitive), then original index.
    switches.sort_by(|(idx_a, p_a), (idx_b, p_b)| {
        let alive_a = p_a.hp > 0;
        let alive_b = p_b.hp > 0;
        match alive_b.cmp(&alive_a) {
            std::cmp::Ordering::Equal => {
                let name_a = format!("{:?}", p_a.id).to_lowercase();
                let name_b = format!("{:?}", p_b.id).to_lowercase();
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

    // if should_log {
    //     // Log AFTER sort
    //     let after: Vec<String> = switches
    //         .iter()
    //         .map(|(idx, p)| format!("#{} id={:?} hp={} alive={}", idx, p.id, p.hp, p.hp > 0))
    //         .collect();
    //     eprintln!(
    //         "[switch_slot_index][turn {}] candidates AFTER  sort: [{}]",
    //         current_turn,
    //         after.join(", ")
    //     );
    // }

    // Where does the target land in the sorted list?
    let pos = switches.iter().position(|(idx, _)| *idx == target_idx);
    // if should_log {
    //     eprintln!(
    //         "[switch_slot_index][turn {}] target_idx={} resolved_position={:?} (0-based in sorted candidates)",
    //         current_turn, target_idx, pos
    //     );
    // }

    pos
}


fn normalize_priors(priors: &mut [f32]) {
    if priors.is_empty() {
        return;
    }
    let sum: f32 = priors.iter().sum();
    if sum <= f32::EPSILON {
        let uniform = 1.0 / priors.len() as f32;
        for value in priors.iter_mut() {
            *value = uniform;
        }
    } else {
        for value in priors.iter_mut() {
            *value /= sum;
        }
    }
}

fn move_index_to_usize(index: PokemonMoveIndex) -> usize {
    match index {
        PokemonMoveIndex::M0 => 0,
        PokemonMoveIndex::M1 => 1,
        PokemonMoveIndex::M2 => 2,
        PokemonMoveIndex::M3 => 3,
    }
}

// Determine the alphabetical slot (0..3) of an active move, matching observation ordering
fn active_move_slot_index(state: &State, side_ref: SideReference, target: PokemonMoveIndex) -> usize {
    let side = match side_ref {
        SideReference::SideOne => &state.side_one,
        SideReference::SideTwo => &state.side_two,
    };
    let active = side.get_active_immutable();
    // Collect (original_index, name)
    let mut entries: Vec<(PokemonMoveIndex, String)> = vec![
        (PokemonMoveIndex::M0, format!("{:?}", active.moves.m0.id).to_lowercase()),
        (PokemonMoveIndex::M1, format!("{:?}", active.moves.m1.id).to_lowercase()),
        (PokemonMoveIndex::M2, format!("{:?}", active.moves.m2.id).to_lowercase()),
        (PokemonMoveIndex::M3, format!("{:?}", active.moves.m3.id).to_lowercase()),
    ];
    entries.sort_by(|a, b| {
        let cmp = a.1.cmp(&b.1);
        if cmp == std::cmp::Ordering::Equal {
            move_index_to_usize(a.0).cmp(&move_index_to_usize(b.0))
        } else {
            cmp
        }
    });
    entries
        .iter()
        .position(|(idx, _)| *idx == target)
        .unwrap_or(move_index_to_usize(target))
}

fn pokemon_index_to_usize(index: PokemonIndex) -> usize {
    match index {
        PokemonIndex::P0 => 0,
        PokemonIndex::P1 => 1,
        PokemonIndex::P2 => 2,
        PokemonIndex::P3 => 3,
        PokemonIndex::P4 => 4,
        PokemonIndex::P5 => 5,
    }
}
