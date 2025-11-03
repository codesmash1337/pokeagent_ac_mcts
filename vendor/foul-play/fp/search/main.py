import atexit
import logging

# import random  # Currently unused (sampling commented out)
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path

from constants import BattleType
from fp.battle import Battle
from config import FoulPlayConfig
from .standard_battles import prepare_battles
from .random_battles import prepare_random_battles

from poke_engine import (
    State as PokeEngineState,
    monte_carlo_tree_search,
    MctsResult,
)

from fp.search.poke_engine_helpers import battle_to_poke_engine_state

logger = logging.getLogger(__name__)

# Add project root to path for compare_tokens module
PROJECT_ROOT = (
    Path(__file__).resolve().parents[4]
)  # vendor/foul-play/fp/search/main.py -> project root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Add metamon to path if not already there
METAMON_PATH = PROJECT_ROOT / "vendor" / "metamon"
TEST_UTILS_PATH = METAMON_PATH / "test"
if METAMON_PATH.exists() and str(METAMON_PATH) not in sys.path:
    sys.path.insert(0, str(METAMON_PATH))

if TEST_UTILS_PATH.exists() and str(TEST_UTILS_PATH) not in sys.path:
    sys.path.insert(0, str(TEST_UTILS_PATH))

# Import metamon for comparison
try:
    import amago  # noqa: F401 - needed for model initialization
    from metamon.interface import UniversalState  # noqa: F401 - used in type hints
    from metamon.poke_engine_adapter import poke_engine_state_to_universal_state
    from metamon.stateful_inference import (
        prepare_observation,
        init_inference_inputs,
        get_policy_and_value,
        DEFAULT_TARGET_ENTROPY_RATIO,
        DEFAULT_ADAPT_STRENGTH,
    )
    from metamon.rl.pretrained import get_pretrained_model
    import torch
    import numpy as np

    METAMON_AVAILABLE = True
except ImportError:
    METAMON_AVAILABLE = False

_PROCESS_POOL: ProcessPoolExecutor | None = None
_POLICY = None
_DEVICE = None
_OBSERVATION_SPACE = None
_MCTS_LOG_DIR = None
_TOKEN_DECODER = None  # Cache for reverse vocabulary mapping (token_id -> word)


def _try_decode_token(token_id):
    """Try to decode a token ID to its string representation.

    Returns a string like '(ironmoth)' or '' if decoding fails.
    """
    global _TOKEN_DECODER

    if token_id is None:
        return ""

    if not METAMON_AVAILABLE:
        return ""

    # Lazy load the token decoder (reverse vocabulary)
    if _TOKEN_DECODER is None:
        try:
            import json

            vocab_path = (
                PROJECT_ROOT
                / "vendor"
                / "metamon"
                / "metamon"
                / "tokenizer"
                / "DefaultObservationSpace-v1.json"
            )

            with open(vocab_path) as f:
                vocab = json.load(f)

            # Build reverse mapping: token_id -> word
            _TOKEN_DECODER = {v: k for k, v in vocab.items()}
        except Exception:
            _TOKEN_DECODER = {}  # Empty dict to avoid retrying
            return ""

    if not _TOKEN_DECODER:
        return ""

    try:
        # Decode the token using reverse vocab
        word = _TOKEN_DECODER.get(int(token_id))
        if word:
            return f"({word})"
        return ""
    except Exception:
        return ""


def _get_token_position_description(index):
    """Get a human-readable description of what this token position represents.

    Based on TeamPreviewObservationSpace structure (106 tokens total).
    """
    # Format and force switch (0-1)
    if index == 0:
        return "Battle format tag"
    if index == 1:
        return "Force switch flag"

    # Player active pokemon (2-9): <player>, name, item, ability, types(1-2), effect, status, tera_type
    if 2 <= index <= 9:
        labels = [
            "<player> tag",
            "Active pokemon name",
            "Active item",
            "Active ability",
            "Active type 1/2",
            "Active type 2/special",
            "Active effect",
            "Active status",
            "Active tera type",
        ]
        return f"Player active: {labels[index - 2]}"

    # Player moves (10-25): 4 moves × (<move>, name, type, category)
    if 10 <= index <= 25:
        move_offset = (index - 10) // 4
        move_field = (index - 10) % 4
        field_names = ["<move> tag", "move name", "move type", "move category"]
        return f"Player move {move_offset + 1}: {field_names[move_field]}"

    # Player switches (26-75): 5 switches × (<switch>, name, item, ability, <moveset>, 4 moves, tera_type)
    if 26 <= index <= 75:
        switch_offset = (index - 26) // 10
        switch_field = (index - 26) % 10
        if switch_field == 0:
            return f"Player switch {switch_offset + 1}: <switch> tag"
        elif switch_field == 1:
            return f"Player switch {switch_offset + 1}: pokemon name"
        elif switch_field == 2:
            return f"Player switch {switch_offset + 1}: item"
        elif switch_field == 3:
            return f"Player switch {switch_offset + 1}: ability"
        elif switch_field == 4:
            return f"Player switch {switch_offset + 1}: <moveset> tag"
        elif 5 <= switch_field <= 8:
            return f"Player switch {switch_offset + 1}: move {switch_field - 4}"
        else:  # switch_field == 9
            return f"Player switch {switch_offset + 1}: tera type"

    # Opponent active (76-83): <opponent>, name, item, ability, types(1-2), effect, status, tera_type
    if 76 <= index <= 83:
        labels = [
            "<opponent> tag",
            "Opponent name",
            "Opponent item",
            "Opponent ability",
            "Opponent type 1/2",
            "Opponent type 2/special",
            "Opponent effect",
            "Opponent status",
            "Opponent tera type",
        ]
        return f"Opponent active: {labels[index - 76]}"

    # Conditions (84-87): <conditions>, weather, player side conditions, opponent side conditions
    if 84 <= index <= 87:
        labels = [
            "<conditions> tag",
            "Weather",
            "Player side conditions",
            "Opponent side conditions",
        ]
        return f"Conditions: {labels[index - 84]}"

    # Previous moves (88-91): <player_prev>, player move, <opp_prev>, opp move
    if 88 <= index <= 91:
        labels = [
            "<player_prev> tag",
            "Player previous move",
            "<opp_prev> tag",
            "Opponent previous move",
        ]
        return f"Previous moves: {labels[index - 88]}"

    # Revealed opponents (92-97): 6 revealed opponent names
    if 92 <= index <= 97:
        return f"Revealed opponent {index - 91}"

    # Team preview (98-103): 6 team preview names
    if 98 <= index <= 103:
        return f"Team preview {index - 97}"

    # Padding or extended tokens
    if index >= 104:
        return "Extended/padding token"

    return "Unknown position"


def _get_process_pool() -> ProcessPoolExecutor:
    global _PROCESS_POOL
    if _PROCESS_POOL is None:
        _PROCESS_POOL = ProcessPoolExecutor(max_workers=FoulPlayConfig.parallelism)

        def _shutdown() -> None:
            if _PROCESS_POOL is not None:
                _PROCESS_POOL.shutdown(wait=True)

        atexit.register(_shutdown)
    return _PROCESS_POOL


def _get_policy_and_observation_space():
    """Lazy load the policy and observation space."""
    global _POLICY, _DEVICE, _OBSERVATION_SPACE

    if not METAMON_AVAILABLE:
        return None, None, None

    if _POLICY is None:
        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_name = "Abra"
        checkpoint = None  # Use latest

        try:
            model = get_pretrained_model(model_name)
            agent = model.initialize_agent(checkpoint=checkpoint, log=False)
            _POLICY = agent.policy
            _POLICY.eval()
            _POLICY.to(_DEVICE)
            # Use the observation space from the model
            _OBSERVATION_SPACE = model.observation_space
        except Exception:
            # logger.error(f"[POLICY] Failed to load model: {e}", exc_info=True)
            return None, None, None

    return _POLICY, _DEVICE, _OBSERVATION_SPACE


def _get_mcts_log_dir(force_refresh=False):
    """Get or create the MCTS logging directory for this session.

    Tries to reuse a recent directory (within last 60 seconds) to stay
    synchronized with the Rust MCTS logging.

    Args:
        force_refresh: If True, re-scan for the most recent directory (use after first MCTS call)
    """
    global _MCTS_LOG_DIR
    if _MCTS_LOG_DIR is None or force_refresh:
        current_time = int(time.time())
        mcts_logs_root = Path("mcts_logs")

        # Check if there's a recent directory we can reuse (within last 60 seconds)
        if mcts_logs_root.exists():
            recent_dirs = []
            for dir_path in mcts_logs_root.iterdir():
                if dir_path.is_dir() and dir_path.name.isdigit():
                    dir_timestamp = int(dir_path.name)
                    if current_time - dir_timestamp < 60:  # Within last 60 seconds
                        recent_dirs.append((dir_timestamp, dir_path))

            if recent_dirs:
                # Use the most recent directory
                recent_dirs.sort(reverse=True)
                new_dir = recent_dirs[0][1]
                if _MCTS_LOG_DIR != new_dir:
                    _MCTS_LOG_DIR = new_dir
                    # logger.info(
                    #     f"[LOGGING] {'Switched to' if force_refresh else 'Using'} MCTS log directory: {_MCTS_LOG_DIR}"
                    # )
                return _MCTS_LOG_DIR

        # Create new directory if no recent one exists (only if not forcing refresh)
        if not force_refresh:
            _MCTS_LOG_DIR = mcts_logs_root / str(current_time)
            _MCTS_LOG_DIR.mkdir(parents=True, exist_ok=True)
            # logger.info(f"[LOGGING] Created new MCTS log directory: {_MCTS_LOG_DIR}")
    return _MCTS_LOG_DIR


def print_universal_state_and_prior(
    battle: Battle, poke_engine_state: PokeEngineState, turn_num: int = None
):
    """Print the universal state and prior distribution for the current turn.

    Args:
        battle: The battle state
        poke_engine_state: The poke-engine state
        turn_num: Optional turn number. If None, will detect from MCTS log directory.
    """
    if not METAMON_AVAILABLE:
        # logger.info("[PRIOR] Metamon not available, skipping prior distribution")
        return

    policy, device, obs_space = _get_policy_and_observation_space()
    if policy is None:
        # logger.info("[PRIOR] Policy not loaded, skipping prior distribution")
        return

    # Set up file logging
    log_dir = _get_mcts_log_dir()

    # If turn_num not provided, find the most recent turn directory created by MCTS
    if turn_num is None:
        turn_dirs = sorted(
            [d for d in log_dir.iterdir() if d.is_dir() and d.name.startswith("turn_")]
        )
        if turn_dirs:
            turn_dir = turn_dirs[-1]  # Use most recent
            turn_num = int(turn_dir.name.split("_")[1])
        else:
            # logger.error("[PRIOR] No turn directories found")
            return
    else:
        turn_dir = log_dir / f"turn_{turn_num:03d}"
        turn_dir.mkdir(parents=True, exist_ok=True)

    log_file_path = turn_dir / "universal_state_prior.txt"

    try:
        with open(log_file_path, "w") as log_file:

            def log_both(message):
                """Log to both console and file."""
                # logger.info(message)
                log_file.write(message + "\n")

            # Convert poke-engine state to universal state
            battle_format = battle.pokemon_format or "gen9ou"
            log_both(f"\n{'=' * 80}")
            game_turn = getattr(battle, "turn", "?")
            log_both(
                f"MCTS TURN {turn_num} (Game Turn {game_turn}) - UNIVERSAL STATE AND PRIOR DISTRIBUTION"
            )
            log_both(f"{'=' * 80}")

            universal_state = poke_engine_state_to_universal_state(
                poke_engine_state, battle_format=battle_format, perspective="side_one"
            )

            # Print universal state
            log_both("\n--- UNIVERSAL STATE ---")
            log_both(f"Format: {universal_state.format}")
            log_both(
                f"Battle Status: Won={universal_state.battle_won}, Lost={universal_state.battle_lost}, Forced Switch={universal_state.forced_switch}"
            )

            log_both(f"\nPlayer Active: {universal_state.player_active_pokemon.name}")
            log_both(f"  HP: {universal_state.player_active_pokemon.hp_pct:.2%}")
            log_both(f"  Types: {universal_state.player_active_pokemon.types}")
            log_both(f"  Item: {universal_state.player_active_pokemon.item}")
            log_both(f"  Ability: {universal_state.player_active_pokemon.ability}")
            log_both(f"  Status: {universal_state.player_active_pokemon.status}")
            log_both(f"  Effect: {universal_state.player_active_pokemon.effect}")
            log_both(
                f"  Tera Type: {universal_state.player_active_pokemon.tera_type} (Can Tera: {universal_state.can_tera})"
            )
            log_both("  Moves (alphabetically sorted):")
            from metamon.interface import consistent_move_order

            sorted_moves = consistent_move_order(
                universal_state.player_active_pokemon.moves
            )
            for i, move in enumerate(sorted_moves):
                log_both(
                    f"    {i}. {move.name} ({move.move_type}/{move.category}) - Power: {move.base_power}, PP: {move.current_pp}/{move.max_pp}"
                )

            log_both(
                f"\nOpponent Active: {universal_state.opponent_active_pokemon.name}"
            )
            log_both(f"  HP: {universal_state.opponent_active_pokemon.hp_pct:.2%}")
            log_both(f"  Types: {universal_state.opponent_active_pokemon.types}")
            log_both(f"  Status: {universal_state.opponent_active_pokemon.status}")
            log_both(
                f"  Tera Type: {universal_state.opponent_active_pokemon.tera_type}"
            )
            log_both(f"  Opponents Remaining: {universal_state.opponents_remaining}")

            log_both(
                f"\nAvailable Switches ({len(universal_state.available_switches)}) (alphabetically sorted):"
            )
            from metamon.interface import consistent_pokemon_order

            sorted_switches = consistent_pokemon_order(
                universal_state.available_switches
            )
            for i, switch in enumerate(sorted_switches):
                log_both(
                    f"  {i + 4}. {switch.name} - HP: {switch.hp_pct:.2%}, Status: {switch.status}"
                )

            log_both("\nField Conditions:")
            log_both(f"  Weather: {universal_state.weather}")
            log_both(f"  Field: {universal_state.battle_field}")
            log_both(f"  Player Conditions: {universal_state.player_conditions}")
            log_both(f"  Opponent Conditions: {universal_state.opponent_conditions}")

            log_both(
                f"\nOpponent Team Preview: {', '.join(universal_state.opponent_teampreview)}"
            )

            # Get observation and prior distribution
            obs_space.reset()  # Reset for new battle context
            obs = obs_space.state_to_obs(universal_state)

            # Determine legal actions
            legal_actions = []
            if not universal_state.forced_switch:
                # Add moves (0-3)
                for i in range(len(universal_state.player_active_pokemon.moves)):
                    legal_actions.append(i)
                # Add tera moves (9-12) if can tera
                if universal_state.can_tera:
                    for i in range(len(universal_state.player_active_pokemon.moves)):
                        legal_actions.append(9 + i)
            # Add switches (4-8)
            for i in range(len(universal_state.available_switches)):
                legal_actions.append(4 + i)

            log_both(f"\nLegal Actions: {legal_actions}")

            # Get prior distribution
            obs_torch = prepare_observation(
                obs, legal_actions, 13, device
            )  # 13 actions total
            rl2s, time_idxs, hidden_state = init_inference_inputs(1, device, policy)

            action_probs, q_values, state_value, _, _ = get_policy_and_value(
                policy,
                obs_torch,
                rl2s,
                time_idxs,
                hidden_state,
                gamma_idx=-1,
                target_entropy_ratio=DEFAULT_TARGET_ENTROPY_RATIO,
                adapt_strength=DEFAULT_ADAPT_STRENGTH,
            )

            # Print prior distribution
            log_both("\n--- PRIOR DISTRIBUTION (from Abra policy) ---")
            log_both(f"State Value: {state_value.item():.4f}")

            probs_np = action_probs.cpu().numpy()
            q_vals_np = q_values.cpu().numpy()

            # Print raw probabilities for all 13 actions
            log_both("\nRaw Probabilities (all 13 actions):")
            log_both(
                f"  Moves (0-3):       {probs_np[0]:.6f}, {probs_np[1]:.6f}, {probs_np[2]:.6f}, {probs_np[3]:.6f}"
            )
            log_both(
                f"  Switches (4-8):    {probs_np[4]:.6f}, {probs_np[5]:.6f}, {probs_np[6]:.6f}, {probs_np[7]:.6f}, {probs_np[8]:.6f}"
            )
            log_both(
                f"  Tera+Moves (9-12): {probs_np[9]:.6f}, {probs_np[10]:.6f}, {probs_np[11]:.6f}, {probs_np[12]:.6f}"
            )

            # Create sorted move list (alphabetically sorted)
            from metamon.interface import consistent_move_order

            sorted_moves = consistent_move_order(
                universal_state.player_active_pokemon.moves
            )

            # Create sorted switch list (alphabetically sorted, excluding fainted)
            from metamon.interface import consistent_pokemon_order

            sorted_switches = consistent_pokemon_order(
                universal_state.available_switches
            )

            # Map action indices to descriptions using sorted lists
            action_names = []
            for i in range(13):
                if i < 4:
                    if i < len(sorted_moves):
                        action_names.append(f"Move: {sorted_moves[i].name}")
                    else:
                        action_names.append(f"Move {i} (invalid)")
                elif i < 9:
                    switch_idx = i - 4
                    if switch_idx < len(sorted_switches):
                        action_names.append(
                            f"Switch: {sorted_switches[switch_idx].name}"
                        )
                    else:
                        action_names.append(f"Switch {switch_idx} (invalid)")
                else:
                    move_idx = i - 9
                    if move_idx < len(sorted_moves):
                        action_names.append(f"Tera+Move: {sorted_moves[move_idx].name}")
                    else:
                        action_names.append(f"Tera+Move {move_idx} (invalid)")

            log_both(
                "\nAction Probabilities (sorted by probability, legal actions only):"
            )

            # Sort by probability for better readability
            sorted_indices = np.argsort(probs_np)[::-1]
            for idx in sorted_indices:
                if (
                    idx in legal_actions and probs_np[idx] > 0.001
                ):  # Only show legal actions with >0.1% prob
                    log_both(
                        f"  [{idx:2d}] {action_names[idx]:30s} - Prob: {probs_np[idx]:6.2%}, Q-value: {q_vals_np[idx]:7.4f}"
                    )

            # Print tokenized observation
            log_both("\n--- TOKENIZED OBSERVATION ---")

            if "text_tokens" in obs:
                text_tokens = obs["text_tokens"]
                log_both(f"Text Tokens ({len(text_tokens)} tokens):")
                # Print tokens in chunks of 10 for readability
                for i in range(0, len(text_tokens), 10):
                    chunk = text_tokens[i : i + 10]
                    log_both(f"  [{i:3d}-{i + len(chunk) - 1:3d}]: {chunk.tolist()}")
            elif "text" in obs:
                text = obs["text"]
                if hasattr(text, "item"):
                    text = text.item()
                log_both(f"Text (raw): {text}")

            if "numbers" in obs:
                numbers = obs["numbers"]
                log_both(f"\nNumerical Features ({len(numbers)} features):")
                # Print numbers in chunks of 8 for readability
                for i in range(0, len(numbers), 8):
                    chunk = numbers[i : i + 8]
                    formatted_chunk = ", ".join([f"{x:7.4f}" for x in chunk])
                    log_both(f"  [{i:2d}-{i + len(chunk) - 1:2d}]: {formatted_chunk}")

            log_both(f"\n{'=' * 80}\n")

        # logger.info(f"[LOGGING] Saved universal state and prior to: {log_file_path}")
        pass

    except Exception:
        # logger.error(
        #     f"[PRIOR] Error printing universal state and prior: {e}", exc_info=True
        # )
        pass


def select_move_from_mcts_results(mcts_results: list[(MctsResult, float, int)]) -> str:
    final_policy = {}
    final_total_scores = {}  # Track aggregated total scores for greedy action
    final_total_visits = {}  # Track aggregated total visits for greedy action

    for mcts_result, sample_chance, index in mcts_results:
        this_policy = max(mcts_result.side_one, key=lambda x: x.visits)
        logger.info(
            "Policy {}: {} visited {}% avg_score={} sample_chance_multiplier={}".format(
                index,
                this_policy.move_choice,
                round(100 * this_policy.visits / mcts_result.total_visits, 2),
                round(this_policy.total_score / this_policy.visits, 3)
                if this_policy.visits > 0
                else 0.0,
                round(sample_chance, 3),
            )
        )
        for s1_option in mcts_result.side_one:
            move = s1_option.move_choice
            # Weight visits by sample_chance (normalized by total_visits)
            weighted_visits = sample_chance * (
                s1_option.visits / mcts_result.total_visits
            )
            # Weight average score (total_score/visits) by sample_chance and visits
            # This preserves the average value when aggregating: avg = total_score / visits
            # We multiply by visits to get total_score contribution, then normalize by total_visits
            weighted_total_score = (
                sample_chance * (s1_option.total_score / mcts_result.total_visits)
                if s1_option.visits > 0
                else 0.0
            )

            final_policy[move] = final_policy.get(move, 0) + weighted_visits
            final_total_scores[move] = (
                final_total_scores.get(move, 0.0) + weighted_total_score
            )
            final_total_visits[move] = (
                final_total_visits.get(move, 0.0) + weighted_visits
            )

    # Always compute max visits option (for comparison when greedy_action is enabled)
    final_policy_sorted = sorted(final_policy.items(), key=lambda x: x[1], reverse=True)
    max_visits_choice = final_policy_sorted[0][0] if final_policy_sorted else None

    if FoulPlayConfig.greedy_action:
        # Greedy action: select move with highest average value among moves with at least max_visits//2 visits
        max_visits = max(final_total_visits.values()) if final_total_visits else 0.0
        min_visits_threshold = max_visits / 2.0

        # Filter to moves with at least max_visits//2 visits and calculate average value
        qualified_moves = {}
        for move in final_policy.keys():
            if (
                final_total_visits[move] >= min_visits_threshold
                and final_total_visits[move] > 0
            ):
                avg_value = final_total_scores[move] / final_total_visits[move]
                qualified_moves[move] = avg_value

        if qualified_moves:
            # Select move with highest average value
            choice_move = max(qualified_moves.items(), key=lambda x: x[1])[0]

            # Compare with max visits selection
            if choice_move != max_visits_choice:
                max_visits_avg = (
                    final_total_scores[max_visits_choice]
                    / final_total_visits[max_visits_choice]
                    if max_visits_choice
                    and final_total_visits.get(max_visits_choice, 0) > 0
                    else 0.0
                )
                logger.info(
                    f"⚠️  GREEDY ACTION DIFFERENCE: Selected {choice_move} (avg_value={qualified_moves[choice_move]:.4f}, visits={final_total_visits[choice_move]:.4f}) "
                    f"instead of {max_visits_choice} (avg_value={max_visits_avg:.4f}, visits={final_total_visits[max_visits_choice]:.4f})"
                )

            logger.info(
                f"Greedy Action: Selected {choice_move} (avg_value={qualified_moves[choice_move]:.4f}) from {len(qualified_moves)} moves with >= {min_visits_threshold:.4f} visits"
            )
            logger.info("Qualified moves (visits >= max_visits//2):")
            sorted_qualified = sorted(
                qualified_moves.items(), key=lambda x: x[1], reverse=True
            )
            for move, avg_value in sorted_qualified[:5]:  # Show top 5
                logger.info(
                    f"\t{move:30s} - avg_value: {avg_value:.4f}, visits: {final_total_visits[move]:.4f}"
                )
            return choice_move
        else:
            # Fallback: no moves qualified, use default selection
            logger.info(
                f"Greedy Action: No moves qualified (min_visits_threshold={min_visits_threshold:.4f}), falling back to max visits"
            )

    # Default: select by visit count
    logger.info("Final Policy (sorted by aggregated visit percentage):")
    for i, policy in enumerate(final_policy_sorted[:5]):  # Show top 5
        logger.info(f"\t{round(policy[1] * 100, 3)}%: {policy[0]}")

    # Return the move with the highest visit count (no sampling)
    return max_visits_choice


def get_result_from_mcts(state_str: str, search_time_ms: int, index: int) -> MctsResult:
    # logger.debug("Calling with {} state: {}".format(index, state_str))
    poke_engine_state = PokeEngineState.from_string(state_str)

    res = monte_carlo_tree_search(poke_engine_state, search_time_ms)
    logger.info("Iterations {}: {}".format(index, res.total_visits))
    return res


def get_result_from_mcts_direct(
    poke_engine_state: PokeEngineState, search_time_ms: int, index: int
) -> MctsResult:
    """Direct version that takes a State object instead of a string."""
    # logger.debug("Calling with {} (direct state object)".format(index))
    res = monte_carlo_tree_search(poke_engine_state, search_time_ms)
    logger.info("Iterations {}: {}".format(index, res.total_visits))
    return res


def search_time_num_battles_randombattles(battle):
    revealed_pkmn = len(battle.opponent.reserve)
    if battle.opponent.active is not None:
        revealed_pkmn += 1

    opponent_active_num_moves = len(battle.opponent.active.moves)
    in_time_pressure = battle.time_remaining is not None and battle.time_remaining <= 60

    # it is still quite early in the battle and the pkmn in front of us
    # hasn't revealed any moves: search a lot of battles shallowly
    if (
        revealed_pkmn <= 3
        and battle.opponent.active.hp > 0
        and opponent_active_num_moves == 0
    ):
        num_battles_multiplier = 2 if in_time_pressure else 4
        return FoulPlayConfig.parallelism * num_battles_multiplier, int(
            FoulPlayConfig.search_time_ms // 2
        )

    else:
        num_battles_multiplier = 1 if in_time_pressure else 2
        return FoulPlayConfig.parallelism * num_battles_multiplier, int(
            FoulPlayConfig.search_time_ms
        )


def search_time_num_battles_standard_battle(battle):
    opponent_active_num_moves = len(battle.opponent.active.moves)
    in_time_pressure = battle.time_remaining is not None and battle.time_remaining <= 60

    if (
        battle.team_preview
        or (battle.opponent.active.hp > 0 and opponent_active_num_moves == 0)
        or opponent_active_num_moves < 3
    ):
        # num_battles_multiplier = 1 if in_time_pressure else 2
        num_battles_multiplier = 1  # SINGLE
        return FoulPlayConfig.parallelism * num_battles_multiplier, int(
            FoulPlayConfig.search_time_ms
        )
    else:
        return FoulPlayConfig.parallelism, FoulPlayConfig.search_time_ms


def compare_turn_observations(turn_num: int, log_dir: Path) -> dict:
    """Compare observations from MCTS (Rust) vs Prior (Python) for a single turn.

    Returns a dictionary with comparison results.
    """
    try:
        turn_dir = log_dir / f"turn_{turn_num:03d}"
        mcts_file = turn_dir / "policy_vs_mcts.txt"
        prior_file = turn_dir / "universal_state_prior.txt"

        if not mcts_file.exists() or not prior_file.exists():
            return {
                "turn": turn_num,
                "status": "missing_files",
                "mcts_exists": mcts_file.exists(),
                "prior_exists": prior_file.exists(),
            }

        # Read both files
        with open(mcts_file, "r") as f:
            mcts_content = f.read()

        with open(prior_file, "r") as f:
            prior_content = f.read()

        # Use compare_tokens to parse and compare
        from compare_tokens import parse_old_format, parse_new_format

        # Both MCTS and Prior files now use new format (multi-line)
        mcts_tokens, mcts_nums = parse_old_format(mcts_content)
        prior_tokens, prior_nums = parse_new_format(prior_content)

        # Find differences
        token_diffs = []
        for i in range(max(len(mcts_tokens), len(prior_tokens))):
            val_mcts = mcts_tokens[i] if i < len(mcts_tokens) else None
            val_prior = prior_tokens[i] if i < len(prior_tokens) else None
            if val_mcts != val_prior:
                token_diffs.append((i, val_mcts, val_prior))

        num_diffs = []
        for i in range(max(len(mcts_nums), len(prior_nums))):
            val_mcts = mcts_nums[i] if i < len(mcts_nums) else None
            val_prior = prior_nums[i] if i < len(prior_nums) else None

            if val_mcts is None or val_prior is None:
                if val_mcts != val_prior:
                    num_diffs.append((i, val_mcts, val_prior, None))
            elif abs(val_mcts - val_prior) > 1e-6:
                diff = val_prior - val_mcts
                num_diffs.append((i, val_mcts, val_prior, diff))

        return {
            "turn": turn_num,
            "status": "compared",
            "token_total": max(len(mcts_tokens), len(prior_tokens)),
            "token_diffs": len(token_diffs),
            "token_diff_details": token_diffs,
            "num_total": max(len(mcts_nums), len(prior_nums)),
            "num_diffs": len(num_diffs),
            "num_diff_details": num_diffs,
            "identical": len(token_diffs) == 0 and len(num_diffs) == 0,
        }

    except Exception as e:
        # logger.error(f"Error comparing turn {turn_num}: {e}", exc_info=True)
        return {
            "turn": turn_num,
            "status": "error",
            "error": str(e),
        }


def _parse_policy_vs_mcts_file(file_path: Path) -> dict:
    """Parse policy_vs_mcts.txt to extract top prior and top MCTS actions."""
    try:
        with open(file_path, "r") as f:
            content = f.read()

        # Find the "Initial Policy" section
        top_prior_action = None
        top_prior_prob = None
        in_prior_section = False

        # Find the "MCTS Results" section
        top_mcts_action = None
        top_mcts_visits = None
        in_mcts_section = False

        for line in content.split("\n"):
            # Skip empty lines and separator lines first
            if (
                not line.strip()
                or line.startswith("Idx")
                or line.startswith("----")
                or line.startswith("====")
            ):
                continue

            # Check for section headers (must have "--- " with space after)
            if "--- Initial Policy" in line:
                in_prior_section = True
                in_mcts_section = False
                continue
            elif "--- MCTS Results" in line:
                in_prior_section = False
                in_mcts_section = True
                continue
            elif line.startswith("--- "):
                in_prior_section = False
                in_mcts_section = False
                continue

            # Parse the first data line in each section
            if in_prior_section and top_prior_action is None:
                parts = line.split()
                if len(parts) >= 3:
                    # Format: Idx Move Prior
                    top_prior_action = parts[1]
                    top_prior_prob = float(parts[2])

            if in_mcts_section and top_mcts_action is None:
                parts = line.split()
                if len(parts) >= 4:
                    # Format: Idx Move Prior Visits AvgVal
                    top_mcts_action = parts[1]
                    top_mcts_visits = int(parts[3])

        disagrees = None
        if top_prior_action is not None and top_mcts_action is not None:
            disagrees = top_prior_action != top_mcts_action

        return {
            "top_prior_action": top_prior_action,
            "top_prior_prob": top_prior_prob,
            "top_mcts_action": top_mcts_action,
            "top_mcts_visits": top_mcts_visits,
            "disagrees": disagrees,
        }
    except Exception:
        return {
            "top_prior_action": None,
            "top_prior_prob": None,
            "top_mcts_action": None,
            "top_mcts_visits": None,
            "disagrees": None,
        }


def generate_battle_comparison_report():
    """Generate a comprehensive comparison report for all turns in the current battle."""
    try:
        # Find the most recent mcts_logs directory instead of creating a new one
        mcts_logs_root = Path("mcts_logs")
        if not mcts_logs_root.exists():
            return

        # Get all timestamped directories, sorted by timestamp (most recent last)
        log_dirs = sorted(
            [d for d in mcts_logs_root.iterdir() if d.is_dir() and d.name.isdigit()],
            key=lambda d: int(d.name),
        )

        if not log_dirs:
            return

        # Use the most recent directory
        log_dir = log_dirs[-1]

        # Find all turn directories
        turn_dirs = sorted(
            [d for d in log_dir.iterdir() if d.is_dir() and d.name.startswith("turn_")]
        )

        if not turn_dirs:
            # logger.info("[COMPARISON] No turn directories found")
            return

        # logger.info(
        #     f"[COMPARISON] Comparing observations for {len(turn_dirs)} turns..."
        # )

        # Compare each turn
        results = []
        for turn_dir in turn_dirs:
            turn_num = int(turn_dir.name.split("_")[1])
            result = compare_turn_observations(turn_num, log_dir)
            results.append(result)

        # Parse policy vs MCTS for each turn
        policy_mcts_data = {}
        disagreement_turns = []
        for turn_dir in turn_dirs:
            turn_num = int(turn_dir.name.split("_")[1])
            mcts_file = turn_dir / "policy_vs_mcts.txt"
            if mcts_file.exists():
                data = _parse_policy_vs_mcts_file(mcts_file)
                policy_mcts_data[turn_num] = data
                if data.get("disagrees"):
                    disagreement_turns.append(turn_num)

        # Generate report
        report_path = log_dir / "observation_comparison_report.txt"

        with open(report_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("OBSERVATION COMPARISON REPORT\n")
            f.write("MCTS (Rust) vs Prior (Python) Token Comparison\n")
            f.write("=" * 80 + "\n\n")

            # Summary
            total_turns = len(results)
            compared_turns = sum(1 for r in results if r["status"] == "compared")
            identical_turns = sum(1 for r in results if r.get("identical", False))

            f.write("SUMMARY:\n")
            f.write(f"  Total Turns: {total_turns}\n")
            f.write(f"  Successfully Compared: {compared_turns}\n")
            f.write(f"  Identical Observations: {identical_turns}\n")
            f.write(f"  Different Observations: {compared_turns - identical_turns}\n")
            f.write("\n")

            # Add MCTS vs Policy disagreements section
            f.write("=" * 80 + "\n")
            f.write("MCTS vs POLICY PRIOR DISAGREEMENTS\n")
            f.write("=" * 80 + "\n")
            f.write(f"Total Turns with Disagreement: {len(disagreement_turns)}\n")
            f.write(
                f"Disagreement Rate: {len(disagreement_turns)}/{len(policy_mcts_data)} "
            )
            f.write(
                f"({100 * len(disagreement_turns) / max(len(policy_mcts_data), 1):.1f}%)\n"
            )
            f.write("=" * 80 + "\n\n")

            if disagreement_turns:
                f.write(
                    "Turns where MCTS selected a different action than the policy's top prior:\n\n"
                )
                for turn_num in sorted(disagreement_turns):
                    data = policy_mcts_data[turn_num]
                    f.write(f"Turn {turn_num:03d}:\n")
                    f.write(
                        f"  Policy Prior: {data['top_prior_action']:25} (prob={data['top_prior_prob']:.4f})\n"
                    )
                    f.write(
                        f"  MCTS Choice:  {data['top_mcts_action']:25} (visits={data['top_mcts_visits']})\n"
                    )
                    f.write("\n")
            else:
                f.write(
                    "No disagreements found - MCTS always chose the policy's top prior action.\n"
                )

            f.write("\n")

            # Per-turn details
            f.write("=" * 80 + "\n")
            f.write("PER-TURN COMPARISON\n")
            f.write("=" * 80 + "\n\n")

            for result in results:
                turn = result["turn"]
                status = result["status"]

                f.write(f"--- Turn {turn:03d} ---\n")

                if status == "missing_files":
                    f.write("  Status: Missing files\n")
                    f.write(
                        f"    MCTS file exists: {result.get('mcts_exists', False)}\n"
                    )
                    f.write(
                        f"    Prior file exists: {result.get('prior_exists', False)}\n"
                    )
                    f.write("\n")
                    continue

                if status == "error":
                    f.write("  Status: Error\n")
                    f.write(f"    Error: {result.get('error', 'Unknown')}\n")
                    f.write("\n")
                    continue

                # Detailed comparison
                if result["identical"]:
                    f.write("  ✅ IDENTICAL - All tokens and features match!\n")
                else:
                    f.write("  ❌ DIFFERENCES FOUND\n")
                    f.write(
                        f"    Text Tokens: {result['token_diffs']}/{result['token_total']} different\n"
                    )
                    f.write(
                        f"    Numerical Features: {result['num_diffs']}/{result['num_total']} different\n"
                    )

                    # Show ALL token differences (with decoding and position info)
                    if result["token_diffs"] > 0 and result["token_diff_details"]:
                        f.write(
                            f"\n    Token Differences ({result['token_diffs']} total):\n"
                        )
                        for idx, val_mcts, val_prior in result["token_diff_details"]:
                            # Get what this position represents
                            position_desc = _get_token_position_description(idx)

                            # Try to decode tokens to show what they represent
                            mcts_decoded = _try_decode_token(val_mcts)
                            prior_decoded = _try_decode_token(val_prior)

                            f.write(f"      [{idx:3d}] {position_desc}\n")
                            f.write(
                                f"             MCTS: {val_mcts or 'missing':>6} {mcts_decoded}  |  Prior: {val_prior or 'missing':>6} {prior_decoded}\n"
                            )

                    # Show ALL numerical differences
                    if result["num_diffs"] > 0 and result["num_diff_details"]:
                        f.write(
                            f"\n    Numerical Differences ({result['num_diffs']} total):\n"
                        )
                        for idx, val_mcts, val_prior, diff in result[
                            "num_diff_details"
                        ]:
                            if val_mcts is None:
                                f.write(
                                    f"      [{idx:2d}] MCTS: missing          Prior: {val_prior:8.4f}\n"
                                )
                            elif val_prior is None:
                                f.write(
                                    f"      [{idx:2d}] MCTS: {val_mcts:8.4f}  Prior: missing\n"
                                )
                            else:
                                f.write(
                                    f"      [{idx:2d}] MCTS: {val_mcts:8.4f}  Prior: {val_prior:8.4f}  Diff: {diff:+8.4f}\n"
                                )

                f.write("\n")

            # Generate detailed comparison files for each turn with differences
            f.write("=" * 80 + "\n")
            f.write("DETAILED COMPARISON FILES\n")
            f.write("=" * 80 + "\n\n")

            for result in results:
                if result.get("status") == "compared" and not result.get(
                    "identical", False
                ):
                    turn = result["turn"]
                    turn_dir = log_dir / f"turn_{turn:03d}"

                    # Generate detailed comparison using compare_tokens
                    detailed_file = turn_dir / "token_comparison_detailed.txt"

                    try:
                        mcts_file = turn_dir / "policy_vs_mcts.txt"
                        prior_file = turn_dir / "universal_state_prior.txt"

                        with open(mcts_file, "r") as mf:
                            mcts_content = mf.read()

                        with open(prior_file, "r") as pf:
                            prior_content = pf.read()

                        # Redirect stdout to capture compare_observations output
                        import io
                        import contextlib

                        output_buffer = io.StringIO()
                        with contextlib.redirect_stdout(output_buffer):
                            from compare_tokens import compare_observations

                            compare_observations(
                                mcts_content,
                                prior_content,
                                format1="new",
                                format2="new",
                            )

                        # Write captured output to file
                        with open(detailed_file, "w") as df:
                            df.write(output_buffer.getvalue())

                        f.write(
                            f"  Turn {turn:03d}: {detailed_file.relative_to(log_dir)}\n"
                        )

                    except Exception as e:
                        f.write(
                            f"  Turn {turn:03d}: Error generating detailed comparison - {e}\n"
                        )
                        # logger.error(
                        #     f"Error generating detailed comparison for turn {turn}: {e}"
                        # )

        # logger.info(f"[COMPARISON] Report saved to: {report_path}")
        # logger.info(
        #     f"[COMPARISON] Compared {compared_turns} turns, {identical_turns} identical, {compared_turns - identical_turns} different"
        # )

        return report_path

    except Exception:
        # logger.error(
        #     f"[COMPARISON] Error generating battle comparison report: {e}",
        #     exc_info=True,
        # )
        return None


def find_best_move(battle: Battle) -> str:
    battle = deepcopy(battle)
    if battle.team_preview:
        battle.user.active = battle.user.reserve.pop(0)
        battle.opponent.active = battle.opponent.reserve.pop(0)

    if battle.battle_type == BattleType.RANDOM_BATTLE:
        num_battles, search_time_per_battle = search_time_num_battles_randombattles(
            battle
        )
        battles = prepare_random_battles(battle, num_battles)
    elif battle.battle_type == BattleType.BATTLE_FACTORY:
        num_battles, search_time_per_battle = search_time_num_battles_standard_battle(
            battle
        )
        battles = prepare_random_battles(battle, num_battles)
    elif battle.battle_type == BattleType.STANDARD_BATTLE:
        num_battles, search_time_per_battle = search_time_num_battles_standard_battle(
            battle
        )
        battles = prepare_battles(battle, num_battles)
    else:
        raise ValueError("Unsupported battle type: {}".format(battle.battle_type))

    logger.info("Searching for a move using MCTS...")
    logger.info(
        "Sampling {} battles at {}ms each".format(num_battles, search_time_per_battle)
    )

    if FoulPlayConfig.parallelism <= 1:
        mcts_results = []
        for index, (b, chance) in enumerate(battles):
            pe_state = battle_to_poke_engine_state(b)
            result = get_result_from_mcts_direct(
                pe_state,
                search_time_per_battle,
                index,
            )
            mcts_results.append((result, chance, index))

            # After EACH MCTS call, refresh log directory and log to the turn directory MCTS just created

            # Only run this when we are debugging token difference
            # TODO: Update to poke-env instead of using our previous python implementation
            # _get_mcts_log_dir(force_refresh=True)
            # print_universal_state_and_prior(b, pe_state)  # Auto-detect turn number
    else:
        executor = _get_process_pool()
        futures = []
        for index, (b, chance) in enumerate(battles):
            fut = executor.submit(
                get_result_from_mcts,
                battle_to_poke_engine_state(b).to_string(),
                search_time_per_battle,
                index,
            )
            futures.append((fut, chance, index))

        # Wait for all results
        mcts_results = [
            (fut.result(), chance, index) for (fut, chance, index) in futures
        ]

        # TODO: In parallel mode, we need to log for each battle after all complete
        # For now, just log once using auto-detected turn
        if futures and battles:
            _get_mcts_log_dir(force_refresh=True)
            first_battle, _ = battles[0]
            first_pe_state = battle_to_poke_engine_state(first_battle)
            print_universal_state_and_prior(
                first_battle, first_pe_state
            )  # Auto-detect turn
    choice = select_move_from_mcts_results(mcts_results)
    logger.info("Choice: {}".format(choice))
    return choice
