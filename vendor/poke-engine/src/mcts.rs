use crate::engine::generate_instructions::generate_instructions_from_move_pair;
use crate::engine::state::MoveChoice;
use crate::instruction::StateInstructions;
use crate::neural_evaluate;
use crate::state::{PokemonIndex, PokemonMoveIndex, SideReference, State};
use rand::distr::weighted::WeightedIndex;
use rand::prelude::*;
use rand::rng;
use std::collections::HashMap;
use std::fs::{create_dir_all, File, OpenOptions};
use std::io::Write;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const PUCT_EXPLORATION: f32 = 2.0;
const VERBOSE_LOGGING: bool = false;
const BATCH_SIZE: usize = 16;

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

fn evaluate_with_fallback(state: &State) -> EvalOutcome {
    if let Some(value) = neural_evaluate::neural_state_value(state) {
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
    if let Some(value) = neural_evaluate::neural_state_value_for_side(state, side) {
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

fn evaluate_with_fallback_batch_for_side(states: &[&State], side: SideReference) -> Vec<EvalOutcome> {
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
}

static LOG_TURN: AtomicU32 = AtomicU32::new(0);

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
) {
    if let Ok(mut file) = File::create(path) {
        let _ = writeln!(file, "=== POLICY PRIORS vs MCTS RESULTS ===\n");

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
    }
}

fn transform_eval(eval: &EvalOutcome, root: &EvalOutcome) -> f32 {
    match (eval.source, root.source) {
        (ValueSource::Heuristic, ValueSource::Heuristic) => sigmoid(eval.value - root.value),
        _ => eval.value,
    }
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
            })
            .collect();
        let mut s2_options_vec: Vec<MoveNode> = s2_options
            .into_iter()
            .map(|x| MoveNode {
                move_choice: x,
                total_score: 0.0,
                visits: 0,
                prior: 0.0,
            })
            .collect();

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
        let mut choice = 0;
        let mut best_ucb1 = f32::MIN;
        for (index, node) in side_map.iter().enumerate() {
            let this_ucb1 = node.ucb1(self.times_visited);
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

    pub unsafe fn backpropagate(&mut self, score: f32, state: &mut State) {
        self.times_visited += 1;
        self.total_state_score += score;
        if self.root {
            return;
        }

        let parent_s1_movenode =
            &mut (*self.parent).s1_options.as_mut().unwrap()[self.s1_choice as usize];
        parent_s1_movenode.total_score += score;
        parent_s1_movenode.visits += 1;

        let parent_s2_movenode =
            &mut (*self.parent).s2_options.as_mut().unwrap()[self.s2_choice as usize];
        parent_s2_movenode.total_score += 1.0 - score;
        parent_s2_movenode.visits += 1;

        state.reverse_instructions(&self.instructions.instruction_list);
        (*self.parent).backpropagate(score, state);
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
}

impl MoveNode {
    pub fn ucb1(&self, parent_visits: u32) -> f32 {
        let parent = (parent_visits.max(1)) as f32;
        if self.visits == 0 {
            // return f32::INFINITY;
            return PUCT_EXPLORATION * self.prior * parent.sqrt();

        }
        let q = self.total_score / self.visits as f32;
        let u = PUCT_EXPLORATION * self.prior * parent.sqrt() / (1.0 + self.visits as f32);
        q + u
    }
    pub fn average_score(&self) -> f32 {
        let score = self.total_score / self.visits as f32;
        score
    }
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
    let mut root_node = Node::new();
    verbose_eprintln!("Side one options: {:?}", side_one_options);
    verbose_eprintln!("Side two options: {:?}", side_two_options);
    unsafe {
        root_node.populate(&*state, side_one_options, side_two_options);
    }
    root_node.root = true;

    let root_eval = evaluate_with_fallback_for_side(state, SideReference::SideOne);
    root_node.policy_priors = root_eval.policy.clone();
    // Also seed opponent priors at root using opponent perspective
    if let Some(opp) = neural_evaluate::neural_state_value_for_side(state, SideReference::SideTwo) {
        root_node.s2_policy_priors = Some(opp.policy);
    }
    if let Some(policy) = &root_eval.policy {
        verbose_eprintln!("Policy priors: {:?}", policy);
    }
    root_node.refresh_priors(&*state);
    let turn_index = LOG_TURN.fetch_add(1, Ordering::Relaxed);
    let logging_paths = logging_paths_for_turn(turn_index);
    log_state_value(state, &root_eval, &logging_paths);
    let mut pending: Vec<PendingEvaluation> = Vec::new();
    let root_state = state.clone();
    let start_time = std::time::Instant::now();
    let mut batch_count = 0;
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
                let reward = if terminal == -1.0 { 0.0 } else { terminal };
                unsafe { (*expanded_node).backpropagate(reward, &mut work_state) };
                continue;
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

        if !pending.is_empty() {
            let flush_start = std::time::Instant::now();
            let batch_size = pending.len();
            flush_pending(&mut pending, &root_eval);
            let flush_time = flush_start.elapsed().as_secs_f64() * 1000.0;
            batch_count += 1;
            eprintln!("[RUST_BATCH_TIMING] batch_num={} size={} collect={:.2}ms flush={:.2}ms total_visits={}",
                batch_count, batch_size, batch_collect_time, flush_time, root_node.times_visited);
        }

        if root_node.times_visited >= 10_000_000 {
            break;
        }
    }

    if !pending.is_empty() {
        let flush_start = std::time::Instant::now();
        let batch_size = pending.len();
        flush_pending(&mut pending, &root_eval);
        let flush_time = flush_start.elapsed().as_secs_f64() * 1000.0;
        batch_count += 1;
        eprintln!("[RUST_BATCH_TIMING] batch_num={} size={} collect=N/A flush={:.2}ms total_visits={} (final)",
            batch_count, batch_size, flush_time, root_node.times_visited);
    }
       
    // // SANITY CHECK: Skip MCTS rollouts, use only policy priors
    // // Set visits and scores directly from priors for side one
    // if let Some(s1_options) = root_node.s1_options.as_mut() {
    //     for node in s1_options.iter_mut() {
    //         // Give each option 1 visit with score equal to its prior
    //         node.visits = (node.prior * 1000.0) as u32;
    //         // node.total_score = node.prior;
    //         node.total_score = 1.0;
    //     }
    // }
    
    // // For side two, use uniform/default
    // if let Some(s2_options) = root_node.s2_options.as_mut() {
    //     for node in s2_options.iter_mut() {
    //         node.visits = (node.prior * 1000.0) as u32;
    //         node.total_score = 1.0;
    //     }
    // }
    
    // root_node.times_visited = 1;
    
    // eprintln!("SANITY CHECK MODE: Using policy priors only, no MCTS rollouts");

    let visits = root_node.times_visited;
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
        tree_stats.log_summary();
        tree_stats.write_json(&logging_paths.stats_path);
        let mut state_for_logging = state.clone();
        dump_tree_json(&root_node, &mut state_for_logging, &logging_paths.tree_path);
        log_policy_priors_comparison(&state, options, SideReference::SideOne, &logging_paths.comparison_path);

        // Side two comparison uses the opponent options present at root
        if let Some(options_s2) = root_node.s2_options.as_ref() {
            log_policy_priors_comparison(
                &state,
                options_s2,
                SideReference::SideTwo,
                &logging_paths.comparison_path_side2,
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

fn flush_pending(pending: &mut Vec<PendingEvaluation>, root_eval: &EvalOutcome) {
    if pending.is_empty() {
        return;
    }
    let prep_start = std::time::Instant::now();
    let mut entries = Vec::new();
    entries.append(pending);
    let state_refs: Vec<&State> = entries.iter().map(|entry| &entry.state).collect();
    let prep_time = prep_start.elapsed().as_secs_f64() * 1000.0;

    let eval_start = std::time::Instant::now();
    let evals_s1 = evaluate_with_fallback_batch_for_side(&state_refs, SideReference::SideOne);
    let evals_s2 = evaluate_with_fallback_batch_for_side(&state_refs, SideReference::SideTwo);
    let eval_time = eval_start.elapsed().as_secs_f64() * 1000.0;

    let backprop_start = std::time::Instant::now();
    for (mut entry, (eval_s1, eval_s2)) in entries
        .into_iter()
        .zip(evals_s1.into_iter().zip(evals_s2.into_iter()))
    {
        let score = transform_eval(&eval_s1, root_eval);
        revert_virtual_loss(&entry.path, entry.leaf);
        let mut state_for_backprop = entry.state;
        unsafe {
            // Store the raw state value before backpropagating
            (*entry.leaf).raw_state_value = Some(eval_s1.value);
            (*entry.leaf).policy_priors = eval_s1.policy.clone();
            (*entry.leaf).s2_policy_priors = eval_s2.policy.clone();
            (*entry.leaf).refresh_priors(&state_for_backprop);
            (*entry.leaf).backpropagate(score, &mut state_for_backprop);
        }
    }
    let backprop_time = backprop_start.elapsed().as_secs_f64() * 1000.0;
    eprintln!("[FLUSH_TIMING] prep={:.2}ms eval={:.2}ms backprop={:.2}ms",
        prep_time, eval_time, backprop_time);
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
        let state_score_str = match child.raw_state_value {
            Some(val) => format!("{:.6}", val),
            None => "null".to_string(),
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
    let mut priors: Vec<f32> = move_nodes
        .iter()
        .map(|node| choice_policy_value(state, side_ref, &node.move_choice, policy))
        .collect();
    normalize_priors(&mut priors);
    for (node, prior) in move_nodes.iter_mut().zip(priors.into_iter()) {
        node.prior = prior;
    }
}

fn choice_policy_value(
    state: &State,
    side_ref: SideReference,
    choice: &MoveChoice,
    policy: Option<&[f32]>,
) -> f32 {
    if let (Some(policy), Some(index)) = (policy, action_index_for_choice(state, side_ref, choice)) {
        policy.get(index).copied().unwrap_or(0.0)
    } else {
        0.0
    }
}

fn action_index_for_choice(
    state: &State,
    side_ref: SideReference,
    choice: &MoveChoice,
) -> Option<usize> {
    match choice {
        MoveChoice::Move(idx) => Some(move_index_to_usize(*idx)),
        #[cfg(not(any(feature = "gen1", feature = "gen2", feature = "gen3")))]
        MoveChoice::MoveTera(idx) => Some(9 + move_index_to_usize(*idx)),
        #[cfg(not(any(feature = "gen1", feature = "gen2", feature = "gen3")))]
        MoveChoice::MoveMega(idx) => Some(move_index_to_usize(*idx)),
        MoveChoice::Switch(pokemon_index) =>
            switch_slot_index(state, side_ref, *pokemon_index).map(|slot| 4 + slot),
        MoveChoice::None => None,
    }
}

fn switch_slot_index(state: &State, side_ref: SideReference, target: PokemonIndex) -> Option<usize> {
    let side = match side_ref {
        SideReference::SideOne => &state.side_one,
        SideReference::SideTwo => &state.side_two,
    };
    let target_idx = pokemon_index_to_usize(target);
    let active_idx = pokemon_index_to_usize(side.active_index);
    if target_idx == active_idx {
        return None;
    }
    let mut slot = 0;
    for (idx, pokemon) in side.pokemon.pkmn.iter().enumerate() {
        if idx == active_idx || pokemon.hp <= 0 {
            continue;
        }
        if idx == target_idx {
            return Some(slot);
        }
        slot += 1;
    }
    None
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
