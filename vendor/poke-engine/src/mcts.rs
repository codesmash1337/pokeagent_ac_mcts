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
const PUCT_C2: f32 = 5000.0; // INCREASE on more iterations
const DIRICHLET_NOISE_EPSILON: f32 = 0.25;
const DIRICHLET_NOISE_ALPHA: f32 = 0.3; // number of bumped options / total options = 4/13
const VIRTUAL_PRIOR_STRENGTH: f32 = 0.0; // Normally 8; keep at 0
const VIRTUAL_PRIOR_BASELINE: f32 = 0.0;
const SIDE_TWO_PRIORS_ONLY: bool = true;
// When true, Side 2 samples from Q-values (soft minimax) instead of priors
// Temperature controls determinism: lower = more deterministic (closer to pure minimax)
const SIDE_TWO_SOFT_MINIMAX: bool = false;
const SIDE_TWO_MINIMAX_TEMPERATURE: f32 = 0.1; // Lower = more deterministic (0.01 = almost pure minimax, 1.0 = more exploratory)
// When true, adaptively scale U based on global Q standard deviation (PopArt)
const ADAPTIVE_U_GAIN: bool = true;
const U_GAIN_BETA: f32 = 1.0; // Scaling factor: g_U = beta / global_std(Q)
const U_GAIN_EPSILON: f32 = 1e-4; // Prevent division by zero in std calculation
const U_GAIN_MIN: f32 = 0.1; // Minimum U gain
const U_GAIN_MAX: f32 = 10.0; // Maximum U gain
const U_GAIN_EMA_ALPHA: f32 = 0.75; // EMA smoothing: new_gain = alpha * computed + (1-alpha) * old_gain
const VERBOSE_LOGGING: bool = false;
const BATCH_SIZE: usize = 64;
const VIRTUAL_LOSS_AMOUNT: f32 = 1.0; // Amount to decrement scores for virtual loss
// When true, skip MCTS rollouts and seed visits directly from policy priors (for debugging)
const SANITY_CHECK_POLICY_ONLY: bool = false;
// When true, use hand-crafted heuristic evaluation instead of neural network values
const USE_HEURISTIC_VALUE: bool = false;
// When true, enable detailed debugging with small iteration count and batch size
const SMALL_DEBUG: bool = false;
const SMALL_DEBUG_MAX_ITERS: u32 = 100;
const SMALL_DEBUG_BATCH_SIZE: usize = 2;

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
    raw_value: f32,
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

// PopArt-style global statistics for Q-value normalization
struct GlobalQStats {
    mean: f32,
    variance: f32,
    count: u64,
    alpha: f32, // exponential moving average decay factor
}

impl GlobalQStats {
    fn new() -> Self {
        GlobalQStats {
            mean: 0.0,
            variance: 1.0,
            count: 0,
            alpha: 0.01, // slowly adapt to new statistics
        }
    }

    fn update(&mut self, value: f32) {
        self.count += 1;
        
        if self.count == 1 {
            self.mean = value;
            self.variance = 1.0;
            return;
        }
        
        // Exponential moving average for mean
        let delta = value - self.mean;
        self.mean += self.alpha * delta;
        
        // Exponential moving average for variance (use squared deviation)
        let new_variance = delta * delta;
        self.variance = (1.0 - self.alpha) * self.variance + self.alpha * new_variance;
    }

    fn update_batch(&mut self, values: &[f32]) {
        for &value in values {
            self.update(value);
        }
    }

    fn get_stats(&self) -> (f32, f32) {
        (self.mean, self.variance)
    }

    fn standardize(&self, value: f32) -> f32 {
        let (mean, var) = self.get_stats();
        let std_dev = (var + 1e-8).sqrt();
        (value - mean) / std_dev
    }
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
            raw_value: heuristic_value,
            source: ValueSource::Heuristic,
            policy: Some(policy),
        }
    } else if let Some(value) = neural_evaluate::neural_state_value(state) {
        EvalOutcome {
            value: value.value,
            raw_value: value.raw_value,
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
            raw_value: adjusted_value,
            source: ValueSource::Heuristic,
            policy: Some(policy),
        }
    } else if let Some(value) = neural_evaluate::neural_state_value_for_side(state, side) {
        EvalOutcome {
            value: value.value,
            raw_value: value.raw_value,
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
                    raw_value: heuristic_value,
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
                raw_value: val.raw_value,
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
                    raw_value: adjusted_value,
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
                raw_value: val.raw_value,
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
    debug_detail_path: String,
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
        debug_detail_path: format!("{}/debug_detail.txt", turn_dir),
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

fn log_debug_detail_for_leaf(
    leaf_state: &State,
    leaf_node: &Node,
    eval_s1: &EvalOutcome,
    eval_s2: &EvalOutcome,
    batch_num: usize,
    batch_idx: usize,
    path: &[PathStep],
    root_state: &State,
    paths: &LoggingPaths,
) {
    if !SMALL_DEBUG {
        return;
    }

    if let Ok(mut file) = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&paths.debug_detail_path)
    {
        let _ = writeln!(file, "\n{}", "=".repeat(100));
        let _ = writeln!(file, "BATCH: {}  INDEX: {}", batch_num, batch_idx);
        let _ = writeln!(file, "{}\n", "=".repeat(100));
        
        // Log the action path that led to this state
        let _ = writeln!(file, "=== ACTION PATH (from root to leaf) ===");
        if path.is_empty() {
            let _ = writeln!(file, "  (ROOT - no actions taken)");
        } else {
            unsafe {
                let mut current_state = root_state.clone();
                for (depth, step) in path.iter().enumerate() {
                    let node = &*step.node;
                    
                    // Get move choices from the node's options
                    let s1_move_str = if let Some(s1_opts) = node.s1_options.as_ref() {
                        if let Some(node) = s1_opts.get(step.s1_choice) {
                            node.move_choice.to_string(&current_state.side_one)
                        } else {
                            format!("s1_idx_{}", step.s1_choice)
                        }
                    } else {
                        format!("s1_idx_{}", step.s1_choice)
                    };
                    
                    let s2_move_str = if let Some(s2_opts) = node.s2_options.as_ref() {
                        if let Some(node) = s2_opts.get(step.s2_choice) {
                            node.move_choice.to_string(&current_state.side_two)
                        } else {
                            format!("s2_idx_{}", step.s2_choice)
                        }
                    } else {
                        format!("s2_idx_{}", step.s2_choice)
                    };
                    
                    let _ = writeln!(file, "  Depth {}: ({}, {})", depth, s1_move_str, s2_move_str);
                    
                    // Apply the instructions to advance the state for the next step's move names
                    // This is just for better logging; we don't use this state afterward
                    if let Some(s1_choice_node) = node.s1_options.as_ref().and_then(|opts| opts.get(step.s1_choice)) {
                        if let Some(s2_choice_node) = node.s2_options.as_ref().and_then(|opts| opts.get(step.s2_choice)) {
                            // We could apply instructions here if we wanted more accurate state tracking,
                            // but for now the move names from the current state are sufficient
                        }
                    }
                }
            }
        }
        let _ = writeln!(file, "=====================================\n");

        // Log the actual leaf state being evaluated
        let _ = writeln!(file, "=== LEAF STATE (being evaluated) ===");
        let readable_state = format_state_readable(leaf_state);
        let _ = write!(file, "{}", readable_state);
        let _ = writeln!(file, "=====================================\n");

        // Log Side One evaluation and policy for this leaf state
        let _ = writeln!(file, "=== SIDE ONE (PLAYER) EVALUATION ===");
        let _ = writeln!(file, "Neural Value (normalized to [-1,1]): {:.6}", eval_s1.value);
        let source_s1 = match eval_s1.source {
            ValueSource::Neural => "neural",
            ValueSource::Heuristic => "heuristic",
        };
        let _ = writeln!(file, "Source: {}", source_s1);
        
        if let Some(policy) = &eval_s1.policy {
            let _ = writeln!(file, "\nPolicy Priors (length {}):", policy.len());
            let _ = writeln!(file, "{:<6} {:<40} {:>12}", "Index", "Action", "Prior");
            let _ = writeln!(file, "{}", "-".repeat(60));
            
            // Get available moves for this leaf state
            let (s1_moves, _) = leaf_state.get_all_options();
            for (idx, move_choice) in s1_moves.iter().enumerate() {
                let move_str = move_choice.to_string(&leaf_state.side_one);
                let action_idx = action_index_for_choice(leaf_state, SideReference::SideOne, move_choice);
                let prior_val = action_idx.and_then(|ai| policy.get(ai).copied()).unwrap_or(0.0);
                let _ = writeln!(file, "{:<6} {:<40} {:>12.6}", 
                    action_idx.map(|v| v.to_string()).unwrap_or_else(|| "--".to_string()),
                    move_str, 
                    prior_val);
            }
        }
        let _ = writeln!(file, "=====================================\n");

        // Log Side Two evaluation and policy for this leaf state
        let _ = writeln!(file, "=== SIDE TWO (OPPONENT) EVALUATION ===");
        let _ = writeln!(file, "Neural Value (normalized to [-1,1]): {:.6}", eval_s2.value);
        let source_s2 = match eval_s2.source {
            ValueSource::Neural => "neural",
            ValueSource::Heuristic => "heuristic",
        };
        let _ = writeln!(file, "Source: {}", source_s2);
        
        if let Some(policy) = &eval_s2.policy {
            let _ = writeln!(file, "\nPolicy Priors (length {}):", policy.len());
            let _ = writeln!(file, "{:<6} {:<40} {:>12}", "Index", "Action", "Prior");
            let _ = writeln!(file, "{}", "-".repeat(60));
            
            // Get available moves for this leaf state
            let (_, s2_moves) = leaf_state.get_all_options();
            for (idx, move_choice) in s2_moves.iter().enumerate() {
                let move_str = move_choice.to_string(&leaf_state.side_two);
                let action_idx = action_index_for_choice(leaf_state, SideReference::SideTwo, move_choice);
                let prior_val = action_idx.and_then(|ai| policy.get(ai).copied()).unwrap_or(0.0);
                let _ = writeln!(file, "{:<6} {:<40} {:>12.6}", 
                    action_idx.map(|v| v.to_string()).unwrap_or_else(|| "--".to_string()),
                    move_str, 
                    prior_val);
            }
        }
        let _ = writeln!(file, "======================================\n");
    }
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

        // Print raw policy priors from the neural network
        let _ = writeln!(file, "=== RAW POLICY PRIORS (from neural network) ===");
        if let Some(policy) = policy_priors {
            let _ = writeln!(file, "Policy prior vector (length {}): ", policy.len());
            let _ = writeln!(file, "{:<8} {:>12}", "Index", "Prior");
            let _ = writeln!(file, "{}", "-".repeat(22));
            for (idx, &prior_val) in policy.iter().enumerate() {
                let action_type = if idx < 4 {
                    "Move"
                } else if idx < 9 {
                    "Switch"
                } else if idx < 13 {
                    "Tera"
                } else {
                    "Other"
                };
                let _ = writeln!(file, "{:<8} {:>12.6}  # {}", idx, prior_val, action_type);
            }
        } else {
            let _ = writeln!(file, "Policy priors not available");
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
    // Values are already advantages in [-1, 1]; no further transform needed
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
    
    // Adaptive U gain for each side (EMA smoothed)
    s1_u_gain: f32,
    s2_u_gain: f32,
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
            s1_u_gain: 1.0,
            s2_u_gain: 1.0,
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

    fn standardized_scores(
        &mut self,
        side_map: &[MoveNode],
        is_side_one: bool,
        global_q_stats: &mut GlobalQStats,
    ) -> Vec<(f32, f32, f32, f32)> {
        let use_joint_expectation = is_side_one && self.s2_options.is_some();

        let mut raw_adv: Vec<f32> = Vec::with_capacity(side_map.len());
        let mut bonuses: Vec<f32> = Vec::with_capacity(side_map.len());

        for (index, node) in side_map.iter().enumerate() {
            let stats = node.ucb_stats(self.times_visited);
            let mut adv = stats.raw_q.unwrap_or(0.0);
            if use_joint_expectation {
                if let Some(expected) = self.expected_value_against_opponent(index) {
                    adv = expected;
                }
            }

            raw_adv.push(adv);
            bonuses.push(stats.u);
            
            // Update global Q statistics (PopArt)
            global_q_stats.update(adv);
        }

        let mut results = Vec::with_capacity(side_map.len());

        // Per-sibling local z-score for Q (no global normalization)
        // Compute local mean and std ONLY from visited nodes
        let visited_q: Vec<f32> = raw_adv
            .iter()
            .zip(side_map.iter())
            .filter(|(_, node)| node.visits > 0 || node.virtual_visits > 0.0)
            .map(|(q, _)| *q)
            .collect();
        
        let q_m = if visited_q.is_empty() {
            0.0
        } else {
            visited_q.iter().copied().sum::<f32>() / visited_q.len() as f32
        };
        
        let mut q_s2 = 0.0;
        for &q in &visited_q {
            let d = q - q_m;
            q_s2 += d * d;
        }
        let q_s = if visited_q.is_empty() {
            1.0
        } else {
            (q_s2 / visited_q.len() as f32).sqrt().max(1e-6)
        };
        
        // Apply local z-score normalization to raw Q values
        let mut normalized_q: Vec<f32> = Vec::with_capacity(raw_adv.len());
        for (idx, &q) in raw_adv.iter().enumerate() {
            // Only normalize if the node has been visited
            if side_map[idx].visits > 0 || side_map[idx].virtual_visits > 0.0 {
                normalized_q.push((q - q_m) / q_s);
            } else {
                // Unvisited nodes get 0.0 normalized Q (neutral)
                normalized_q.push(0.0);
            }
        }

        // Compute adaptive U gain if enabled
        let u_gain = if ADAPTIVE_U_GAIN {
            // Get global Q statistics from PopArt
            let (_, global_var) = global_q_stats.get_stats();
            let global_std = (global_var + U_GAIN_EPSILON).sqrt();
            
            // Compute gain: g_U = clip(beta / global_std, g_min, g_max)
            // This normalizes Q to unit scale, so U can be directly compared
            let computed_gain = (U_GAIN_BETA / global_std)
                .max(U_GAIN_MIN)
                .min(U_GAIN_MAX);
            
            // Apply EMA smoothing
            let old_gain = if is_side_one { self.s1_u_gain } else { self.s2_u_gain };
            let smoothed_gain = U_GAIN_EMA_ALPHA * computed_gain + (1.0 - U_GAIN_EMA_ALPHA) * old_gain;
            
            // Update stored gain
            if is_side_one {
                self.s1_u_gain = smoothed_gain;
            } else {
                self.s2_u_gain = smoothed_gain;
            }
            
            smoothed_gain
        } else {
            1.0 // No scaling if adaptive U gain is disabled
        };

        // Combine Q (normalized) with U (scaled by adaptive gain)
        let w_q: f32 = 1.0;
        for idx in 0..side_map.len() {
            let adv = raw_adv[idx];
            let local_q = normalized_q[idx];  // Locally normalized Q
            let raw_u = bonuses[idx];         // Raw U
            let scaled_u = u_gain * raw_u;    // Apply adaptive gain
            let total = w_q * local_q + scaled_u;
            results.push((adv, local_q, scaled_u, total));
        }

        results
    }

    pub fn maximize_ucb_for_side(&mut self, side_map: &[MoveNode], is_side_one: bool, global_q_stats: &mut GlobalQStats) -> usize {
        let scores = self.standardized_scores(side_map, is_side_one, global_q_stats);
        let mut choice = 0;
        let mut best_score = f32::MIN;

        for (index, (_, _, _, total)) in scores.iter().enumerate() {
            if *total > best_score {
                best_score = *total;
                choice = index;
            }
        }

        choice
    }
    
    /// Sample move based on Q values with temperature (soft minimax)
    /// Lower temperature = more deterministic (closer to pure minimax)
    /// Higher temperature = more exploratory
    fn sample_q_with_temperature(&self, side_map: &[MoveNode], is_side_one: bool, temperature: f32) -> usize {
        // Collect Q values for all moves
        let mut q_values: Vec<f32> = Vec::with_capacity(side_map.len());
        for node in side_map.iter() {
            let stats = node.ucb_stats(self.times_visited);
            if let Some(raw_q) = stats.raw_q {
                // From opponent's perspective (negate if side 2)
                let q = if is_side_one { raw_q } else { -raw_q };
                q_values.push(q);
            } else {
                q_values.push(0.0);
            }
        }
        
        // Apply softmax with temperature
        let max_q = q_values.iter().copied().fold(f32::MIN, f32::max);
        let mut exp_values: Vec<f32> = q_values
            .iter()
            .map(|&q| ((q - max_q) / temperature).exp())
            .collect();
        
        let sum: f32 = exp_values.iter().sum();
        if sum <= 0.0 {
            // Fallback to uniform if something went wrong
            return 0;
        }
        
        for val in exp_values.iter_mut() {
            *val /= sum;
        }
        
        // Sample from the distribution
        let dist = WeightedIndex::new(&exp_values).unwrap();
        let mut rng = rng();
        dist.sample(&mut rng)
    }

    fn expected_value_against_opponent(&self, s1_index: usize) -> Option<f32> {
        let s2_options = self.s2_options.as_ref()?;
        if s2_options.is_empty() {
            return None;
        }

        let mut opponent_total = 0.0f32;
        for opt in s2_options.iter() {
            opponent_total += opt.prior.max(0.0);
        }
        if opponent_total <= f32::EPSILON {
            return None;
        }

        let mut weighted_sum = 0.0f32;
        let mut weight_accum = 0.0f32;

        for (s2_index, opponent_choice) in s2_options.iter().enumerate() {
            let weight = (opponent_choice.prior.max(0.0)) / opponent_total;
            if weight <= 0.0 {
                continue;
            }

            if let Some(children) = self.children.get(&(s1_index, s2_index)) {
                let mut joint_visits = 0u32;
                let mut joint_score = 0.0f32;
                for child in children.iter() {
                    joint_visits += child.times_visited;
                    joint_score += child.total_state_score;
                }

                if joint_visits > 0 {
                    let avg = joint_score / joint_visits as f32;
                    weighted_sum += weight * avg;
                    weight_accum += weight;
                }
            }
        }

        if weight_accum <= f32::EPSILON {
            None
        } else {
            Some(weighted_sum / weight_accum)
        }
    }

    fn sample_prior_index(&self, side_map: &[MoveNode]) -> Option<usize> {
        let mut weights: Vec<f64> = side_map
            .iter()
            .map(|node| f64::from(node.prior.max(0.0)))
            .collect();

        let total: f64 = weights.iter().sum();
        if total <= f64::EPSILON {
            return None;
        }
        for w in &mut weights {
            *w /= total;
        }

        let dist = WeightedIndex::new(weights).ok()?;
        let mut rng = rng();
        Some(dist.sample(&mut rng))
    }

    pub unsafe fn selection(
        &mut self,
        state: &mut State,
        path: &mut Vec<PathStep>,
        global_q_stats: &mut GlobalQStats,
    ) -> (*mut Node, usize, usize) {
        let return_node = self as *mut Node;
        if self.s1_options.is_none() {
            let (s1_options, s2_options) = state.get_all_options();
            self.populate(&*state, s1_options, s2_options);
        }

        // Extract references to avoid borrow checker issues with mutable self
        let s1_options_ptr = self.s1_options.as_ref().unwrap() as *const Vec<MoveNode>;
        let s2_options_ptr = self.s2_options.as_ref().unwrap() as *const Vec<MoveNode>;
        
        let s1_mc_index = unsafe {
            self.maximize_ucb_for_side(&*s1_options_ptr, true, global_q_stats)
        };
        let s2_mc_index = if SIDE_TWO_SOFT_MINIMAX {
            // Soft Minimax: Sample from Q-values with temperature (high Q moves more likely)
            unsafe { self.sample_q_with_temperature(&*s2_options_ptr, false, SIDE_TWO_MINIMAX_TEMPERATURE) }
        } else if SIDE_TWO_PRIORS_ONLY {
            unsafe {
                self.sample_prior_index(&*s2_options_ptr)
                    .unwrap_or_else(|| self.maximize_ucb_for_side(&*s2_options_ptr, false, global_q_stats))
            }
        } else {
            unsafe { self.maximize_ucb_for_side(&*s2_options_ptr, false, global_q_stats) }
        };
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
                (*chosen_child).selection(state, path, global_q_stats)
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
        // Advantage for side one; compute from both perspectives
        let advantage = 0.5 * (score_s1 - score_s2);

        self.times_visited += 1;
        self.total_state_score += advantage;
        if self.root {
            return;
        }

        let parent = &mut *self.parent;
        let parent_s1_move = &mut parent.s1_options.as_mut().unwrap()[self.s1_choice as usize];
        parent_s1_move.total_score += advantage;
        parent_s1_move.visits += 1;

        let parent_s2_move = &mut parent.s2_options.as_mut().unwrap()[self.s2_choice as usize];
        parent_s2_move.total_score -= advantage;
        parent_s2_move.visits += 1;

        state.reverse_instructions(&self.instructions.instruction_list);
        parent.backpropagate(score_s1, score_s2, state);
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
            // Terminal state advantage from side-one perspective
            let terminal_score = battle_is_over;
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
    fn ucb_stats(&self, parent_visits: u32) -> UcbStats {
        let parent = (parent_visits.max(1)) as f32;
        let c = c_puct(parent);
        let effective_visits = self.visits as f32 + self.virtual_visits;
        let u = c * self.prior * parent.sqrt() / (1.0 + effective_visits);
        if effective_visits <= 0.0 {
            return UcbStats {
                raw_q: None,
                scaled_q: None,
                u,
                total: u,
            };
        }

        let mut adv_q = (self.total_score + self.virtual_score) / effective_visits;
        if !adv_q.is_finite() {
            adv_q = 0.0;
        }

        UcbStats {
            raw_q: Some(adv_q),
            scaled_q: Some(adv_q),
            u,
            total: adv_q + u,
        }
    }

    pub fn ucb_value(&self, parent_visits: u32, _min_q: f32, _max_q: f32) -> f32 {
        self.ucb_stats(parent_visits).total
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
    let populate_start = std::time::Instant::now();
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
    let populate_time = populate_start.elapsed().as_secs_f64() * 1000.0;
    
    let turn_index = LOG_TURN.fetch_add(1, Ordering::Relaxed);
    SWITCH_DEBUG_TURN.store(turn_index, Ordering::Relaxed);
    SWITCH_DEBUG_PRINTED.store(false, Ordering::Relaxed);
    PRIOR_MAP_PRINTED.store(false, Ordering::Relaxed);
    let logging_paths = logging_paths_for_turn(turn_index);
    
    let logging_setup_start = std::time::Instant::now();
    log_state_value(state, &root_eval, &logging_paths);
    
    // Initialize q/u values file with header
    if let Ok(mut file) = File::create(&logging_paths.q_u_values_path) {
        let _ = writeln!(file, "=== Q/U VALUES BY BATCH FOR TURN {} ===", turn_index);
        let _ = writeln!(file, "This file tracks q (action value) and u (exploration bonus) for each move after each batch.\n");
    }
    let logging_setup_time = logging_setup_start.elapsed().as_secs_f64() * 1000.0;
    
    // Initialize timing variables (will be updated in MCTS loop)
    let mut total_collect_time = 0.0;
    let mut total_eval_time = 0.0;
    let mut total_backprop_time = 0.0;
    let mut mcts_loop_time = 0.0;
    let mut batch_count = 0;
    let mut total_states_evaluated = 0;
    let mut total_state_clone_time = 0.0;
    let mut total_tree_selection_time = 0.0;
    let mut total_node_expand_time = 0.0;
    let mut total_terminal_check_time = 0.0;
    let mut total_terminal_backprop_time = 0.0;
    let mut total_queue_log_time = 0.0;
    let mut total_virtual_loss_time = 0.0;
    let mut fast_terminal_count = 0;
    
    // Initialize global Q statistics for PopArt-style normalization
    let mut global_q_stats = GlobalQStats::new();
    
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
        let effective_batch_size = if SMALL_DEBUG { SMALL_DEBUG_BATCH_SIZE } else { BATCH_SIZE };
        
        while start_time.elapsed() < max_time {
            // Check iteration limit for SMALL_DEBUG mode
            if SMALL_DEBUG && root_node.times_visited >= SMALL_DEBUG_MAX_ITERS {
                eprintln!("[SMALL_DEBUG] Reached max iterations: {}", SMALL_DEBUG_MAX_ITERS);
                break;
            }
            
            let batch_start = std::time::Instant::now();
            while pending.len() < effective_batch_size && start_time.elapsed() < max_time {
                let clone_start = std::time::Instant::now();
                let mut work_state = root_state.clone();
                total_state_clone_time += clone_start.elapsed().as_secs_f64() * 1000.0;

                let mut path = Vec::new();
                let selection_start = std::time::Instant::now();
                let (selected_node, s1_idx, s2_idx) =
                    unsafe { root_node.selection(&mut work_state, &mut path, &mut global_q_stats) };
                total_tree_selection_time += selection_start.elapsed().as_secs_f64() * 1000.0;

                let expand_start = std::time::Instant::now();
                let expanded_node = unsafe { (*selected_node).expand(&mut work_state, s1_idx, s2_idx) };
                total_node_expand_time += expand_start.elapsed().as_secs_f64() * 1000.0;

                let terminal_start = std::time::Instant::now();
                let terminal = work_state.battle_is_over();
                total_terminal_check_time += terminal_start.elapsed().as_secs_f64() * 1000.0;
                if terminal != 0.0 {
                    let reward_s1 = terminal;
                    let reward_s2 = -reward_s1;
                    let fast_backprop_start = std::time::Instant::now();
                    unsafe { (*expanded_node).backpropagate(reward_s1, reward_s2, &mut work_state) };
                    let terminal_backprop_time = fast_backprop_start.elapsed().as_secs_f64() * 1000.0;
                    total_terminal_backprop_time += terminal_backprop_time;
                    total_backprop_time += terminal_backprop_time;
                    fast_terminal_count += 1;
                    continue;
                }

                // Log the selected moves with their q+u values before applying virtual loss
                let log_start = std::time::Instant::now();
                log_queued_action(
                    &root_node,
                    &work_state,
                    &path,
                    s1_idx,
                    s2_idx,
                    &logging_paths,
                    batch_count + 1,
                    &mut global_q_stats,
                );
                total_queue_log_time += log_start.elapsed().as_secs_f64() * 1000.0;

                let virtual_loss_start = std::time::Instant::now();
                apply_virtual_loss(&path, expanded_node);
                total_virtual_loss_time += virtual_loss_start.elapsed().as_secs_f64() * 1000.0;
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
                    &mut global_q_stats,
                );
                total_eval_time += eval_time;
                total_backprop_time += backprop_time;
                batch_count += 1;
            }

            if root_node.times_visited >= 10_000_000 {
                break;
            }
            
            // Also check SMALL_DEBUG limit here
            if SMALL_DEBUG && root_node.times_visited >= SMALL_DEBUG_MAX_ITERS {
                eprintln!("[SMALL_DEBUG] Reached max iterations: {}", SMALL_DEBUG_MAX_ITERS);
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
                &mut global_q_stats,
            );
            total_eval_time += eval_time;
            total_backprop_time += backprop_time;
            batch_count += 1;
        }
        
        mcts_loop_time = start_time.elapsed().as_secs_f64() * 1000.0;
        let total_inference_time = total_eval_time + total_backprop_time;
        eprintln!("[MCTS_TIMING] batches={} states_eval={} visits={} collect={:.1}ms eval={:.1}ms backprop={:.1}ms inference={:.1}ms loop={:.1}ms fast_terminal={} clone={:.1}ms select={:.1}ms expand={:.1}ms log={:.1}ms vloss={:.1}ms", 
            batch_count, total_states_evaluated, root_node.times_visited, 
            total_collect_time, total_eval_time, total_backprop_time, total_inference_time, mcts_loop_time,
            fast_terminal_count, total_state_clone_time, total_tree_selection_time, total_node_expand_time,
            total_queue_log_time, total_virtual_loss_time);
    }
    
    eprintln!("Iterations {}: {}", turn_index, root_node.times_visited);
    
    // Log aggregated Python call timing
    neural_evaluate::log_python_call_stats();

    let logging_phase_start = std::time::Instant::now();
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

    let stats_time = {
        let start = std::time::Instant::now();
    if let Some(options) = root_node.s1_options.as_ref() {
        let tree_stats = collect_root_stats(options);
            // tree_stats.log_summary();
        tree_stats.write_json(&logging_paths.stats_path);
        }
        start.elapsed().as_secs_f64() * 1000.0
    };
    
    let tree_dump_time = {
        let start = std::time::Instant::now();
        if let Some(_options) = root_node.s1_options.as_ref() {
        let mut state_for_logging = state.clone();
        dump_tree_json(&root_node, &mut state_for_logging, &logging_paths.tree_path);
        }
        start.elapsed().as_secs_f64() * 1000.0
    };
    
    let policy_log_time = {
        let start = std::time::Instant::now();
        if let Some(options) = root_node.s1_options.as_ref() {
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
        }
        start.elapsed().as_secs_f64() * 1000.0
    };

    let logging_phase_time = logging_phase_start.elapsed().as_secs_f64() * 1000.0;

    if let Some(options) = root_node.s1_options.as_ref() {
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
    
    // Comprehensive latency breakdown per turn
    eprintln!("\n╔══════════════════════════════════════════════════════════════════════════════╗");
    eprintln!("║ TURN {} LATENCY BREAKDOWN                                                     ║", turn_index);
    eprintln!("╠══════════════════════════════════════════════════════════════════════════════╣");
    eprintln!("║ Phase                    │ Time (ms)  │ % of Total                           ║");
    eprintln!("╠══════════════════════════════════════════════════════════════════════════════╣");
    eprintln!("║ Initial Evaluation        │ {:>10.2} │ {:>5.1}%                               ║", 
        init_eval_time, (init_eval_time / total_turn_time * 100.0));
    eprintln!("║ Node Population          │ {:>10.2} │ {:>5.1}%                               ║", 
        populate_time, (populate_time / total_turn_time * 100.0));
    eprintln!("║ Logging Setup            │ {:>10.2} │ {:>5.1}%                               ║", 
        logging_setup_time, (logging_setup_time / total_turn_time * 100.0));
    
    if !SANITY_CHECK_POLICY_ONLY {
        let selection_clone_share = if total_collect_time > 0.0 {
            (total_state_clone_time / total_collect_time * 100.0).clamp(0.0, 999.9)
        } else {
            0.0
        };
        let selection_traverse_share = if total_collect_time > 0.0 {
            (total_tree_selection_time / total_collect_time * 100.0).clamp(0.0, 999.9)
        } else {
            0.0
        };
        let selection_expand_share = if total_collect_time > 0.0 {
            (total_node_expand_time / total_collect_time * 100.0).clamp(0.0, 999.9)
        } else {
            0.0
        };
        let selection_log_share = if total_collect_time > 0.0 {
            (total_queue_log_time / total_collect_time * 100.0).clamp(0.0, 999.9)
        } else {
            0.0
        };
        let selection_vloss_share = if total_collect_time > 0.0 {
            (total_virtual_loss_time / total_collect_time * 100.0).clamp(0.0, 999.9)
        } else {
            0.0
        };
        let selection_terminal_share = if total_collect_time > 0.0 {
            (total_terminal_check_time / total_collect_time * 100.0).clamp(0.0, 999.9)
        } else {
            0.0
        };

        eprintln!("║ Selection/Collection     │ {:>10.2} │ {:>5.1}%                               ║", 
            total_collect_time, (total_collect_time / total_turn_time * 100.0));
        eprintln!("║   - State cloning        │ {:>10.2} │ {:>5.1}% (Sel {:>5.1}%)              ║", 
            total_state_clone_time,
            (total_state_clone_time / total_turn_time * 100.0),
            selection_clone_share);
        eprintln!("║   - Tree traversal       │ {:>10.2} │ {:>5.1}% (Sel {:>5.1}%)              ║", 
            total_tree_selection_time,
            (total_tree_selection_time / total_turn_time * 100.0),
            selection_traverse_share);
        eprintln!("║   - Node expansion       │ {:>10.2} │ {:>5.1}% (Sel {:>5.1}%)              ║", 
            total_node_expand_time,
            (total_node_expand_time / total_turn_time * 100.0),
            selection_expand_share);
        eprintln!("║   - Terminal checks      │ {:>10.2} │ {:>5.1}% (Sel {:>5.1}%)              ║", 
            total_terminal_check_time,
            (total_terminal_check_time / total_turn_time * 100.0),
            selection_terminal_share);
        eprintln!("║   - Queue logging        │ {:>10.2} │ {:>5.1}% (Sel {:>5.1}%)              ║", 
            total_queue_log_time,
            (total_queue_log_time / total_turn_time * 100.0),
            selection_log_share);
        eprintln!("║   - Virtual loss         │ {:>10.2} │ {:>5.1}% (Sel {:>5.1}%)              ║", 
            total_virtual_loss_time,
            (total_virtual_loss_time / total_turn_time * 100.0),
            selection_vloss_share);
        if fast_terminal_count > 0 {
            eprintln!("║   - Fast terminal backprop│ {:>9.2} │ {:>5.1}% ({} nodes)               ║",
                total_terminal_backprop_time,
                (total_terminal_backprop_time / total_turn_time * 100.0),
                fast_terminal_count);
        }
        eprintln!("║ State Evaluation        │ {:>10.2} │ {:>5.1}%                               ║", 
            total_eval_time, (total_eval_time / total_turn_time * 100.0));
        eprintln!("║ Backpropagation         │ {:>10.2} │ {:>5.1}%                               ║", 
            total_backprop_time, (total_backprop_time / total_turn_time * 100.0));
        eprintln!("║ MCTS Loop (total)       │ {:>10.2} │ {:>5.1}%                               ║", 
            mcts_loop_time, (mcts_loop_time / total_turn_time * 100.0));
    } else {
        eprintln!("║ MCTS Loop               │ {:>10.2} │ {:>5.1}% (SANITY CHECK MODE)           ║", 
            0.0, 0.0);
    }
    
    eprintln!("║ Final Logging            │ {:>10.2} │ {:>5.1}%                               ║", 
        logging_phase_time, (logging_phase_time / total_turn_time * 100.0));
    eprintln!("║   - Stats Collection     │ {:>10.2} │                                         ║", stats_time);
    eprintln!("║   - Tree Dump           │ {:>10.2} │                                         ║", tree_dump_time);
    eprintln!("║   - Policy Comparison   │ {:>10.2} │                                         ║", policy_log_time);
    eprintln!("╠══════════════════════════════════════════════════════════════════════════════╣");
    eprintln!("║ TOTAL TURN TIME         │ {:>10.2} │ 100.0%                                ║", total_turn_time);
    eprintln!("╠══════════════════════════════════════════════════════════════════════════════╣");
    if !SANITY_CHECK_POLICY_ONLY {
        eprintln!("║ Batches: {:>3}  States Evaluated: {:>6}  Visits: {:>10}              ║", 
            batch_count, total_states_evaluated, root_node.times_visited);
    } else {
        eprintln!("║ Visits: {:>10} (SANITY CHECK MODE - policy priors only)               ║", 
            root_node.times_visited);
    }
    eprintln!("╚══════════════════════════════════════════════════════════════════════════════╝\n");

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
    // Apply virtual loss ONLY to the edges (choices), not to node aggregates
    // This avoids double-counting and perturbing unrelated edges
    for step in path {
        unsafe {
            let node = &mut *step.node;
            
            // Apply virtual loss to side1 choice edge
            if let Some(s1_opts) = node.s1_options.as_mut() {
                if let Some(choice) = s1_opts.get_mut(step.s1_choice) {
                    choice.virtual_visits += 1.0;
                    choice.virtual_score -= VIRTUAL_LOSS_AMOUNT; // Penalty from s1's perspective
                }
            }
            
            // Apply virtual loss to side2 choice edge
            if let Some(s2_opts) = node.s2_options.as_mut() {
                if let Some(choice) = s2_opts.get_mut(step.s2_choice) {
                    choice.virtual_visits += 1.0;
                    choice.virtual_score += VIRTUAL_LOSS_AMOUNT; // Penalty from s2's perspective (positive is bad for s1)
                }
            }
        }
    }
    // No need to touch leaf's total_state_score
}

fn revert_virtual_loss(path: &[PathStep], leaf: *mut Node) {
    // Revert virtual loss by undoing changes to edge virtual_visits and virtual_score
    for step in path.iter().rev() {
        unsafe {
            let node = &mut *step.node;
            
            // Revert side1 choice virtual loss
            if let Some(s1_opts) = node.s1_options.as_mut() {
                if let Some(choice) = s1_opts.get_mut(step.s1_choice) {
                    choice.virtual_visits -= 1.0;
                    choice.virtual_score += VIRTUAL_LOSS_AMOUNT;
                }
            }
            
            // Revert side2 choice virtual loss
            if let Some(s2_opts) = node.s2_options.as_mut() {
                if let Some(choice) = s2_opts.get_mut(step.s2_choice) {
                    choice.virtual_visits -= 1.0;
                    choice.virtual_score -= VIRTUAL_LOSS_AMOUNT;
                }
            }
        }
    }
    // No need to touch leaf's total_state_score
}

fn flush_pending(
    pending: &mut Vec<PendingEvaluation>,
    root_eval: &EvalOutcome,
    root_eval_s2: &EvalOutcome,
    root_node: *mut Node,
    state: &State,
    logging_paths: &LoggingPaths,
    batch_num: usize,
    global_q_stats: &mut GlobalQStats,
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
    for (batch_idx, (entry, (eval_s1, eval_s2))) in entries
        .into_iter()
        .zip(evals_s1.into_iter().zip(evals_s2.into_iter()))
        .enumerate()
    {
        // Log the actual state being evaluated in SMALL_DEBUG mode
        if SMALL_DEBUG {
            unsafe {
                let leaf_node = &*entry.leaf;
                log_debug_detail_for_leaf(
                    &entry.state,
                    leaf_node,
                    &eval_s1,
                    &eval_s2,
                    batch_num,
                    batch_idx,
                    &entry.path,
                    state,  // root state for move name lookup
                    logging_paths,
                );
            }
        }
        
        let score_s1 = transform_eval(&eval_s1, root_eval);
        let score_s2 = transform_eval(&eval_s2, root_eval_s2);
        
        // Log the neural evaluation values for this state
        if let Ok(mut file) = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&logging_paths.q_u_values_path)
        {
            let _ = writeln!(
                file,
                "    → Leaf State Values: Side1 raw={:.2}, norm={:.6} | Side2 raw={:.2}, norm={:.6}",
                eval_s1.raw_value,
                eval_s1.value,
                eval_s2.raw_value,
                eval_s2.value
            );
        }
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
        log_q_u_values_by_batch(root_node, state, logging_paths, batch_num, global_q_stats);
    }

    (eval_time, backprop_time)
}



fn log_queued_action(
    root_node: *const Node,
    state: &State,
    path: &[PathStep],
    s1_idx: usize,
    s2_idx: usize,
    logging_paths: &LoggingPaths,
    batch_num: usize,
    global_q_stats: &mut GlobalQStats,
) {
    unsafe {
        if let Ok(mut file) = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&logging_paths.q_u_values_path)
        {
            let parent_visits = (*root_node).times_visited;
            let root_ref = &mut *(root_node as *mut Node);

            let root_s1_idx = path.first().map(|step| step.s1_choice).unwrap_or(s1_idx);
            let root_s2_idx = path.first().map(|step| step.s2_choice).unwrap_or(s2_idx);

            if let (Some(s1_options), Some(s2_options)) = ((*root_node).s1_options.as_ref(), (*root_node).s2_options.as_ref()) {
                let s1_scores = root_ref.standardized_scores(s1_options, true, global_q_stats);
                let s2_scores = root_ref.standardized_scores(s2_options, false, global_q_stats);

                if let (Some(s1_node), Some(s2_node)) = (s1_options.get(s1_idx), s2_options.get(s2_idx)) {
                    let (adv_s1, std_s1, u_s1, total_s1) = s1_scores[s1_idx];
                    let (adv_s2, std_s2, u_s2, total_s2) = s2_scores[s2_idx];

                    let s1_move_str = s1_node.move_choice.to_string(&state.side_one);
                    let s2_move_str = s2_node.move_choice.to_string(&state.side_two);

                    let _ = writeln!(file, "
--- Queued for Batch {} ---", batch_num);
                    let _ = writeln!(
                        file,
                        "Side1: {} (idx: {}) | q_adv: {:.6}, q_std: {:.6}, u: {:.6}, q+u: {:.6}",
                        s1_move_str,
                        s1_idx,
                        adv_s1,
                        std_s1,
                        u_s1,
                        total_s1
                    );
                    let _ = writeln!(
                        file,
                        "Side2: {} (idx: {}) | q_adv: {:.6}, q_std: {:.6}, u: {:.6}, q+u: {:.6}",
                        s2_move_str,
                        s2_idx,
                        adv_s2,
                        std_s2,
                        u_s2,
                        total_s2
                    );

                    if let (Some(root_s1_node), Some(root_s2_node)) = (
                        s1_options.get(root_s1_idx),
                        s2_options.get(root_s2_idx),
                    ) {
                        let (root_adv_s1, root_std_s1, root_u_s1, root_total_s1) = s1_scores[root_s1_idx];
                        let (root_adv_s2, root_std_s2, root_u_s2, root_total_s2) = s2_scores[root_s2_idx];

                        let root_s1_move = root_s1_node.move_choice.to_string(&state.side_one);
                        let root_s2_move = root_s2_node.move_choice.to_string(&state.side_two);
                        let _ = writeln!(
                            file,
                            "Current Selection (root) | Side1: {} (idx: {}) | q_adv: {:.6}, q_std: {:.6}, u: {:.6}, q+u: {:.6} | Side2: {} (idx: {}) | q_adv: {:.6}, q_std: {:.6}, u: {:.6}, q+u: {:.6}",
                            root_s1_move,
                            root_s1_idx,
                            root_adv_s1,
                            root_std_s1,
                            root_u_s1,
                            root_total_s1,
                            root_s2_move,
                            root_s2_idx,
                            root_adv_s2,
                            root_std_s2,
                            root_u_s2,
                            root_total_s2
                        );
                    }
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
    global_q_stats: &mut GlobalQStats,
) {
    unsafe {
        if let Ok(mut file) = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&logging_paths.q_u_values_path)
        {
            let _ = writeln!(file, "
=== BATCH {} ===", batch_num);
            let _ = writeln!(file, "Total visits: {}", (*root_node).times_visited);

            let parent_visits = (*root_node).times_visited;
            let root_ref = &mut *root_node;

            if let (Some(s1_options), Some(s2_options)) = ((*root_node).s1_options.as_ref(), (*root_node).s2_options.as_ref()) {
                // Log all available root actions first
                let _ = writeln!(file, "
--- Available Root Actions ---");
                let _ = writeln!(file, "Side 1 ({} options):", s1_options.len());
                for (idx, node) in s1_options.iter().enumerate() {
                    let move_str = node.move_choice.to_string(&state.side_one);
                    let _ = writeln!(file, "  [{}] {}", idx, move_str);
                }
                let _ = writeln!(file, "
Side 2 ({} options):", s2_options.len());
                for (idx, node) in s2_options.iter().enumerate() {
                    let move_str = node.move_choice.to_string(&state.side_two);
                    let _ = writeln!(file, "  [{}] {}", idx, move_str);
                }

                let s1_scores = root_ref.standardized_scores(s1_options, true, global_q_stats);
                let s2_scores = root_ref.standardized_scores(s2_options, false, global_q_stats);

                if let Some((best_s1_idx, best_s1)) = s1_scores
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1 .3.partial_cmp(&b.1 .3).unwrap())
                {
                    if let Some(best_s1_node) = s1_options.get(best_s1_idx) {
                        let (adv, std, u, total) = *best_s1;
                        let move_str = best_s1_node.move_choice.to_string(&state.side_one);
                        let _ = writeln!(file, "
--- Root Actions (highest q+u) ---");
                        let _ = writeln!(
                            file,
                            "Side1: {} (idx: {}) | q_adv: {:.6}, q_std: {:.6}, u: {:.6}, q+u: {:.6}, visits: {}",
                            move_str,
                            best_s1_idx,
                            adv,
                            std,
                            u,
                            total,
                            best_s1_node.visits
                        );
                    }
                }

                if let Some((best_s2_idx, best_s2)) = s2_scores
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1 .3.partial_cmp(&b.1 .3).unwrap())
                {
                    if let Some(best_s2_node) = s2_options.get(best_s2_idx) {
                        let (adv, std, u, total) = *best_s2;
                        let move_str = best_s2_node.move_choice.to_string(&state.side_two);
                        let _ = writeln!(
                            file,
                            "Side2: {} (idx: {}) | q_adv: {:.6}, q_std: {:.6}, u: {:.6}, q+u: {:.6}, visits: {}",
                            move_str,
                            best_s2_idx,
                            adv,
                            std,
                            u,
                            total,
                            best_s2_node.visits
                        );
                    }
                }

                let mut adv_values: Vec<f32> = Vec::new();
                let mut std_values: Vec<f32> = Vec::new();
                let mut u_values: Vec<f32> = Vec::new();
                let mut qpu_values: Vec<f32> = Vec::new();
                let mut s1_adv_values: Vec<f32> = Vec::new();
                let mut s2_adv_values: Vec<f32> = Vec::new();
                let mut u_max = f32::MIN;
                let mut visited_children = 0;
                let mut total_children = 0;

                let _ = writeln!(file, "
--- Side One Moves ---");
                let _ = writeln!(file, "{:<6} {:<30} {:>10} {:>10} {:>10} {:>10} {:>8} {:>12} {:>12} {:>10}",
                    "Idx", "Move", "raw_q", "q_adv", "q_std", "u", "Visits", "TotalScore", "Prior", "q+u");
                let _ = writeln!(file, "{}", "-".repeat(134));
                total_children += s1_options.len();
                for (idx, (node, score)) in s1_options.iter().zip(s1_scores.iter()).enumerate() {
                    if node.visits > 0 {
                        visited_children += 1;
                    }
                    let (adv, std, u, total) = *score;
                    let raw_q = adv * 1100.0; // Scale back to raw critic range
                    let move_str = node.move_choice.to_string(&state.side_one);
                    let _ = writeln!(file, "{:<6} {:<30} {:>10.2} {:>10.6} {:>10.6} {:>10.6} {:>8} {:>12.6} {:>12.6} {:>10.6}",
                        idx,
                        move_str,
                        raw_q,
                        adv,
                        std,
                        u,
                        node.visits,
                        node.total_score,
                        node.prior,
                        total);
                    adv_values.push(adv);
                    s1_adv_values.push(adv);
                    std_values.push(std);
                    u_values.push(u);
                    qpu_values.push(total);
                    if u > u_max {
                        u_max = u;
                    }
                }
                
                let s1_mean_adv = if s1_adv_values.is_empty() { 0.0 } else { s1_adv_values.iter().sum::<f32>() / s1_adv_values.len() as f32 };
                let _ = writeln!(file, "Average q_adv for Side One: {:.6}", s1_mean_adv);

                let _ = writeln!(file, "
--- Side Two Moves ---");
                let _ = writeln!(file, "{:<6} {:<30} {:>10} {:>10} {:>10} {:>10} {:>8} {:>12} {:>12} {:>10}",
                    "Idx", "Move", "raw_q", "q_adv", "q_std", "u", "Visits", "TotalScore", "Prior", "q+u");
                let _ = writeln!(file, "{}", "-".repeat(134));
                total_children += s2_options.len();
                for (idx, (node, score)) in s2_options.iter().zip(s2_scores.iter()).enumerate() {
                    if node.visits > 0 {
                        visited_children += 1;
                    }
                    let (adv, std, u, total) = *score;
                    let raw_q = adv * 1100.0; // Scale back to raw critic range
                    let move_str = node.move_choice.to_string(&state.side_two);
                    let _ = writeln!(file, "{:<6} {:<30} {:>10.2} {:>10.6} {:>10.6} {:>10.6} {:>8} {:>12.6} {:>12.6} {:>10.6}",
                        idx,
                        move_str,
                        raw_q,
                        adv,
                        std,
                        u,
                        node.visits,
                        node.total_score,
                        node.prior,
                        total);
                    adv_values.push(adv);
                    s2_adv_values.push(adv);
                    std_values.push(std);
                    u_values.push(u);
                    qpu_values.push(total);
                    if u > u_max {
                        u_max = u;
                    }
                }
                
                let s2_mean_adv = if s2_adv_values.is_empty() { 0.0 } else { s2_adv_values.iter().sum::<f32>() / s2_adv_values.len() as f32 };
                let _ = writeln!(file, "Average q_adv for Side Two: {:.6}", s2_mean_adv);

                let _ = writeln!(file, "
--- Batch Statistics ---");
                let mean = |v: &Vec<f32>| if v.is_empty() { 0.0 } else { v.iter().sum::<f32>() / v.len() as f32 };
                let std = |v: &Vec<f32>, m: f32| if v.len() > 1 {
                    let var = v.iter().map(|&x| (x - m).powi(2)).sum::<f32>() / v.len() as f32;
                    var.sqrt()
                } else { 0.0 };

                let mean_adv = mean(&adv_values);
                let std_adv = std(&adv_values, mean_adv);
                let mean_std = mean(&std_values);
                let std_std = std(&std_values, mean_std);
                let mean_u = mean(&u_values);
                let std_u = std(&u_values, mean_u);
                let mean_qpu = mean(&qpu_values);
                let std_qpu = std(&qpu_values, mean_qpu);
                let q_span = if adv_values.is_empty() {
                    0.0
                } else {
                    let min_q = adv_values.iter().fold(f32::MAX, |a, &b| a.min(b));
                    let max_q = adv_values.iter().fold(f32::MIN, |a, &b| a.max(b));
                    max_q - min_q
                };
                let visited_frac = if total_children > 0 {
                    visited_children as f32 / total_children as f32
                } else {
                    0.0
                };
                let u_max_display = if u_max == f32::MIN { 0.0 } else { u_max };

                let _ = writeln!(file, "q_adv:    mean = {:.6}, std = {:.6}", mean_adv, std_adv);
                let _ = writeln!(file, "q_std:    mean = {:.6}, std = {:.6}", mean_std, std_std);
                let _ = writeln!(file, "u:        mean = {:.6}, std = {:.6}", mean_u, std_u);
                let _ = writeln!(file, "q+u:      mean = {:.6}, std = {:.6}", mean_qpu, std_qpu);
                let _ = writeln!(file, "q_span:   {:.6}", q_span);
                let _ = writeln!(file, "u_max:    {:.6}", u_max_display);
                let _ = writeln!(file, "visited_frac: {:.6} ({} / {})", visited_frac, visited_children, total_children);
            }
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
