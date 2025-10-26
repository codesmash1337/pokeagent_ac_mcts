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

    // represents the instructions & s1/s2 moves that led to this node from the parent
    pub instructions: StateInstructions,
    pub s1_choice: u8,
    pub s2_choice: u8,

    // represents the total score and number of visits for this node
    // de-coupled for s1 and s2
    pub s1_options: Option<Vec<MoveNode>>,
    pub s2_options: Option<Vec<MoveNode>>,
    policy_priors: Option<Vec<f32>>,
}

impl Node {
    fn new() -> Node {
        Node {
            root: false,
            parent: std::ptr::null_mut(),
            instructions: StateInstructions::default(),
            times_visited: 0,
            children: HashMap::new(),
            s1_choice: 0,
            s2_choice: 0,
            s1_options: None,
            s2_options: None,
            policy_priors: None,
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
            None,
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
            assign_priors_to_move_nodes(options, state, SideReference::SideTwo, None);
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

    pub unsafe fn selection(&mut self, state: &mut State) -> (*mut Node, usize, usize) {
        let return_node = self as *mut Node;
        if self.s1_options.is_none() {
            let (s1_options, s2_options) = state.get_all_options();
            self.populate(&*state, s1_options, s2_options);
        }

        let s1_mc_index = self.maximize_ucb_for_side(&self.s1_options.as_ref().unwrap());
        let s2_mc_index = self.maximize_ucb_for_side(&self.s2_options.as_ref().unwrap());
        let child_vector = self.children.get_mut(&(s1_mc_index, s2_mc_index));
        match child_vector {
            Some(child_vector) => {
                let child_vec_ptr = child_vector as *mut Vec<Node>;
                let chosen_child = self.sample_node(child_vec_ptr);
                state.apply_instructions(&(*chosen_child).instructions.instruction_list);
                (*chosen_child).selection(state)
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
            let eval = evaluate_with_fallback(&*state);
            self.policy_priors = eval.policy.clone();
            self.refresh_priors(&*state);
            transform_eval(&eval, root_eval)
        } else {
            if battle_is_over == -1.0 {
                0.0
            } else {
                battle_is_over
            }
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
            return f32::INFINITY;
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

fn do_mcts(root_node: &mut Node, state: &mut State, root_eval: &EvalOutcome) {
    let (mut new_node, s1_move, s2_move) = unsafe { root_node.selection(state) };
    new_node = unsafe { (*new_node).expand(state, s1_move, s2_move) };
    let rollout_result = unsafe { (*new_node).rollout(state, root_eval) };
    unsafe { (*new_node).backpropagate(rollout_result, state) }
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

    let root_eval = evaluate_with_fallback(state);
    root_node.policy_priors = root_eval.policy.clone();
    if let Some(policy) = &root_eval.policy {
        verbose_eprintln!("Policy priors: {:?}", policy);
    }
    root_node.refresh_priors(&*state);
    let turn_index = LOG_TURN.fetch_add(1, Ordering::Relaxed);
    let logging_paths = logging_paths_for_turn(turn_index);
    log_state_value(state, &root_eval, &logging_paths);
    let start_time = std::time::Instant::now();
    while start_time.elapsed() < max_time {
        for _ in 0..10 {
            do_mcts(&mut root_node, state, &root_eval);
        }

        /*
        Cut off after 10 million iterations

        Under normal circumstances the bot will only run for 2.5-3.5 million iterations
        however towards the end of a battle the bot may perform tens of millions of iterations

        Beyond about 30 million iterations some floating point nonsense happens where
        MoveNode.total_score stops updating because f32 does not have enough precision

        I can push the problem farther out by using f64 but if the bot is running for 10 million iterations
        then it almost certainly sees a forced win
        */
        if root_node.times_visited == 10_000_000 {
            break;
        }
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
        dump_tree_json(&root_node, &state, &logging_paths.tree_path);

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
fn dump_tree_json(root: &Node, state: &State, path: &str) {
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

unsafe fn dump_node_recursive(
    node: &Node,
    state: &State,
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
        let node_line = format!(
            "{{depth:{}, s1_move:\"{}\", s2_move:\"{}\", visits:{}, node_visits:{}, s1_avg:{:.6}, s2_avg:{:.6}}}",
            depth,
            s1_move_label,
            s2_move_label,
            s1_choice.visits,
            child.times_visited,
            s1_avg,
            s2_avg
        );
        let branch = prefix.clone();
        let connector = if is_last { "`- " } else { "|- " };
        let _ = writeln!(file, "{}{}{}", branch, connector, node_line);

        let mut next_prefix = prefix.clone();
        next_prefix.push_str(if is_last { "   " } else { "|  " });
        dump_node_recursive(child, state, next_prefix, depth + 1, file);
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
