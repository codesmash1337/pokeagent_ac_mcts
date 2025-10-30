import atexit
import logging
import random
import sys
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
    prepare_inference_payload_batch,
)

from fp.search.poke_engine_helpers import battle_to_poke_engine_state

logger = logging.getLogger(__name__)

# Add metamon to path if not already there
METAMON_PATH = Path(__file__).resolve().parents[3] / "metamon"
if METAMON_PATH.exists() and str(METAMON_PATH) not in sys.path:
    sys.path.insert(0, str(METAMON_PATH))
    logger.info(f"[DEBUG] Added metamon to path: {METAMON_PATH}")
else:
    logger.info(f"[DEBUG] Already in path or doesn't exist: {METAMON_PATH}")

# Import metamon for comparison
try:
    from metamon.interface import UniversalState
    from metamon.tokenizer import Tokenizer as AbraTokenizer
    from metamon.poke_engine_adapter import poke_engine_state_to_universal_state

    METAMON_AVAILABLE = True
    logger.info("[DEBUG] Successfully imported metamon modules")
except ImportError as e:
    METAMON_AVAILABLE = False
    logger.warning(f"Metamon not available for tokenization comparison: {e}")

_PROCESS_POOL: ProcessPoolExecutor | None = None


def _get_process_pool() -> ProcessPoolExecutor:
    global _PROCESS_POOL
    if _PROCESS_POOL is None:
        _PROCESS_POOL = ProcessPoolExecutor(max_workers=FoulPlayConfig.parallelism)

        def _shutdown() -> None:
            if _PROCESS_POOL is not None:
                _PROCESS_POOL.shutdown(wait=True)

        atexit.register(_shutdown)
    return _PROCESS_POOL


def select_move_from_mcts_results(mcts_results: list[(MctsResult, float, int)]) -> str:
    final_policy = {}
    for mcts_result, sample_chance, index in mcts_results:
        this_policy = max(mcts_result.side_one, key=lambda x: x.visits)
        logger.info(
            "Policy {}: {} visited {}% avg_score={} sample_chance_multiplier={}".format(
                index,
                this_policy.move_choice,
                round(100 * this_policy.visits / mcts_result.total_visits, 2),
                round(this_policy.total_score / this_policy.visits, 3),
                round(sample_chance, 3),
            )
        )
        for s1_option in mcts_result.side_one:
            final_policy[s1_option.move_choice] = final_policy.get(
                s1_option.move_choice, 0
            ) + (sample_chance * (s1_option.visits / mcts_result.total_visits))

    final_policy = sorted(final_policy.items(), key=lambda x: x[1], reverse=True)

    # Consider all moves that are close to the best move
    highest_percentage = final_policy[0][1]
    final_policy = [i for i in final_policy if i[1] >= highest_percentage * 0.75]
    logger.info("Considered Choices:")
    for i, policy in enumerate(final_policy):
        logger.info(f"\t{round(policy[1] * 100, 3)}%: {policy[0]}")

    choice = random.choices(final_policy, weights=[p[1] for p in final_policy])[0]
    return choice[0]


def get_result_from_mcts(state: str, search_time_ms: int, index: int) -> MctsResult:
    logger.debug("Calling with {} state: {}".format(index, state))
    poke_engine_state = PokeEngineState.from_string(state)

    res = monte_carlo_tree_search(poke_engine_state, search_time_ms)
    logger.info("Iterations {}: {}".format(index, res.total_visits))
    return res


def print_tokenization_comparison(battle: Battle, poke_engine_state: PokeEngineState):
    """Print side-by-side comparison of Metamon/Abra vs Poke-Engine tokenization"""
    logger.info("[DEBUG] print_tokenization_comparison called")
    logger.info(f"[DEBUG] METAMON_AVAILABLE: {METAMON_AVAILABLE}")
    logger.info(f"[DEBUG] battle type: {type(battle)}")
    logger.info(f"[DEBUG] battle.battle_type: {battle.battle_type}")
    logger.info(f"[DEBUG] battle.team_preview: {battle.team_preview}")
    logger.info(f"[DEBUG] battle.pokemon_format: {battle.pokemon_format}")

    if not METAMON_AVAILABLE:
        logger.warning("[DEBUG] Metamon not available, skipping comparison")
        return

    try:
        logger.info("\n" + "=" * 80)
        logger.info("TOKENIZATION COMPARISON")
        logger.info("=" * 80)

        # Convert poke-engine state to UniversalState
        battle_format = battle.pokemon_format or "gen9ou"
        logger.info(f"[DEBUG] Using battle_format: {battle_format}")
        logger.info("[DEBUG] Converting poke_engine_state to UniversalState")
        universal_state = poke_engine_state_to_universal_state(
            poke_engine_state, battle_format
        )
        logger.info(f"[DEBUG] Created universal_state: {type(universal_state)}")
        logger.info("\n--- METAMON STATE (UniversalState) ---")
        logger.info(
            f"Player active: {universal_state.player_active_pokemon.name if universal_state.player_active_pokemon else 'None'}"
        )
        logger.info(
            f"Opponent active: {universal_state.opponent_active_pokemon.name if universal_state.opponent_active_pokemon else 'None'}"
        )
        logger.info(f"Player teampreview: {universal_state.player_teampreview}")
        logger.info(f"Opponent teampreview: {universal_state.opponent_teampreview}")

        # Get Abra tokenization
        abra_tokenizer = AbraTokenizer()
        abra_text_tokens, abra_numerical = abra_tokenizer([universal_state])
        abra_text_tokens = abra_text_tokens[0]  # Get first batch element
        abra_numerical = abra_numerical[0]

        logger.info("\n--- ABRA TOKENIZATION (Ground Truth) ---")
        logger.info(f"Text tokens (length {len(abra_text_tokens)}): {abra_text_tokens}")
        logger.info(
            f"Numerical features (length {len(abra_numerical)}): {[round(x, 3) for x in abra_numerical]}"
        )

        # Get poke-engine tokenization
        payload = prepare_inference_payload_batch([poke_engine_state])
        pe_text_tokens = payload["text_tokens"][0]  # Get first batch element
        pe_numerical = payload["numbers"][0]

        logger.info("\n--- POKE-ENGINE TOKENIZATION (Ours) ---")
        logger.info(f"Text tokens (length {len(pe_text_tokens)}): {pe_text_tokens}")
        logger.info(
            f"Numerical features (length {len(pe_numerical)}): {[round(x, 3) for x in pe_numerical]}"
        )

        # Compare
        logger.info("\n--- COMPARISON ---")
        text_match = abra_text_tokens == pe_text_tokens
        numerical_match = all(
            abs(a - b) < 0.001 for a, b in zip(abra_numerical, pe_numerical)
        )

        logger.info(f"Text tokens match: {text_match}")
        if not text_match:
            for i, (a, b) in enumerate(zip(abra_text_tokens, pe_text_tokens)):
                if a != b:
                    logger.info(f"  Diff at index {i}: Abra={a}, PE={b}")

        logger.info(f"Numerical features match (within 0.001): {numerical_match}")
        if not numerical_match:
            for i, (a, b) in enumerate(zip(abra_numerical, pe_numerical)):
                if abs(a - b) >= 0.001:
                    logger.info(
                        f"  Diff at index {i}: Abra={a:.6f}, PE={b:.6f}, delta={abs(a - b):.6f}"
                    )

        logger.info("=" * 80 + "\n")

    except Exception as e:
        logger.error(f"Error in tokenization comparison: {e}", exc_info=True)


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

    # Print tokenization comparison for the first battle scenario
    logger.info(
        f"[DEBUG] About to check tokenization comparison: battles={len(battles) if battles else 0}, METAMON_AVAILABLE={METAMON_AVAILABLE}"
    )
    if battles:
        logger.info(f"[DEBUG] First battle in list: {type(battles[0])}")
        first_battle, chance = battles[0]
        logger.info(
            f"[DEBUG] Unpacked first_battle type: {type(first_battle)}, chance: {chance}"
        )
        first_pe_state = battle_to_poke_engine_state(first_battle)
        logger.info(f"[DEBUG] Created poke_engine_state: {type(first_pe_state)}")
        print_tokenization_comparison(first_battle, first_pe_state)
    else:
        logger.warning("[DEBUG] No battles to compare!")

    if FoulPlayConfig.parallelism <= 1:
        mcts_results = []
        for index, (b, chance) in enumerate(battles):
            result = get_result_from_mcts(
                battle_to_poke_engine_state(b).to_string(),
                search_time_per_battle,
                index,
            )
            mcts_results.append((result, chance, index))
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
        mcts_results = [
            (fut.result(), chance, index) for (fut, chance, index) in futures
        ]
    choice = select_move_from_mcts_results(mcts_results)
    logger.info("Choice: {}".format(choice))
    return choice
