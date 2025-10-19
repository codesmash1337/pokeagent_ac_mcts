import logging
import random
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from collections import namedtuple

from constants import BattleType
from fp.battle import Battle
from config import FoulPlayConfig
from .standard_battles import prepare_battles
from .random_battles import prepare_random_battles

from poke_engine import State as PokeEngineState, mcts as monte_carlo_tree_search

# Picklable result structures for PyMctsResult (Rust objects can't be pickled)
PicklableMctsResult = namedtuple('PicklableMctsResult', ['side_one', 'side_two', 'total_visits'])
PicklableMoveNode = namedtuple('PicklableMoveNode', ['move_choice', 'visits', 'total_score'])
try:
    from poke_engine import mcts_with_puct
    PUCT_AVAILABLE = True
except ImportError:
    PUCT_AVAILABLE = False
    logger.warning("mcts_with_puct not available - using vanilla MCTS")

from fp.search.poke_engine_helpers import battle_to_poke_engine_state

logger = logging.getLogger(__name__)

# Import neural guidance (optional, may fail if neural-mcts not available)
try:
    from fp.search.neural_guided import select_move_with_neural_guidance
    NEURAL_GUIDANCE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Neural guidance not available: {e}")
    NEURAL_GUIDANCE_AVAILABLE = False


def select_move_from_mcts_results(mcts_results: list[(PicklableMctsResult, float, int)]) -> str:
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


def get_result_from_mcts(state: str, search_time_ms: int, index: int) -> PicklableMctsResult:
    logger.debug("Calling with {} state: {}".format(index, state))
    poke_engine_state = PokeEngineState.from_string(state)

    res = monte_carlo_tree_search(poke_engine_state, search_time_ms)

    # Convert PyMctsResult to picklable format (Rust objects can't be pickled)
    # Note: PyMctsResult has attributes s1, s2, iteration_count (defined in poke-engine-py/src/lib.rs)
    side_one_nodes = [
        PicklableMoveNode(
            move_choice=node.move_choice,
            visits=node.visits,
            total_score=node.total_score
        )
        for node in res.s1
    ]

    side_two_nodes = [
        PicklableMoveNode(
            move_choice=node.move_choice,
            visits=node.visits,
            total_score=node.total_score
        )
        for node in res.s2
    ] if hasattr(res, 's2') and res.s2 else []

    total = res.iteration_count
    logger.info("Iterations {}: {}".format(index, total))

    return PicklableMctsResult(
        side_one=side_one_nodes,
        side_two=side_two_nodes,
        total_visits=total
    )


def get_neural_priors_for_battle(battle: Battle) -> tuple:
    """
    Get neural policy priors for a battle state.
    Returns (s1_priors, s2_priors) where each is a list of floats or None.
    """
    try:
        from neural_mcts import LocalPolicyProvider, StateTranslator

        # Initialize components (could be cached for efficiency)
        translator = StateTranslator()
        provider = LocalPolicyProvider(model_name="Minikazam", device="cpu")

        # Translate and get policy
        obs = translator.translate(battle)
        policy = provider.get_policy(obs)

        if policy is not None:
            # For now, use a simplified mapping
            # In production, would need proper action space alignment
            s1_priors = policy[:10].tolist()  # Use first 10 actions
            total = sum(s1_priors)
            if total > 0:
                s1_priors = [p / total for p in s1_priors]
                return (s1_priors, None)  # Uniform for opponent

        return (None, None)
    except Exception as e:
        logger.warning(f"Failed to get neural priors: {e}")
        return (None, None)


def get_neural_priors_batch(battles: list) -> list:
    """
    Get neural policy priors for a batch of battle states efficiently.

    Args:
        battles: List of (Battle, chance) tuples

    Returns:
        List of (s1_priors, s2_priors) tuples
    """
    try:
        from neural_mcts import LocalPolicyProvider, StateTranslator

        # Initialize components once (singleton pattern for efficiency)
        translator = StateTranslator()
        provider = LocalPolicyProvider(model_name="Minikazam", device="cpu")

        priors_list = []
        for battle, chance in battles:
            try:
                # Translate and get policy
                obs = translator.translate(battle)
                policy = provider.get_policy(obs)

                if policy is not None:
                    # Map Metamon's 13-dim action space to poke-engine move priors
                    # Metamon actions: 0-3 moves, 4-8 switches, 9-12 tera moves
                    # For poke-engine, we use first 10 (4 moves + 5 switches + 1 buffer)
                    s1_priors = policy[:10].tolist()
                    total = sum(s1_priors)
                    if total > 0:
                        s1_priors = [p / total for p in s1_priors]
                        priors_list.append((s1_priors, None))  # Uniform for opponent
                    else:
                        priors_list.append((None, None))
                else:
                    priors_list.append((None, None))

            except Exception as e:
                logger.warning(f"Failed to get priors for battle: {e}")
                priors_list.append((None, None))

        logger.info(f"Successfully computed neural priors for {sum(1 for p, _ in priors_list if p is not None)}/{len(battles)} battles")
        return priors_list

    except Exception as e:
        logger.warning(f"Failed to initialize neural components: {e}")
        # Return None priors for all battles
        return [(None, None) for _ in battles]


def get_result_from_puct_mcts(
    state: str,
    search_time_ms: int,
    index: int,
    c_puct: float = 1.5,
    s1_priors: list = None,
    s2_priors: list = None
) -> PicklableMctsResult:
    """
    Run MCTS with PUCT formula using neural priors.

    Args:
        state: Poke-engine state string
        search_time_ms: Search time budget
        index: Battle index for logging
        c_puct: PUCT exploration constant
        s1_priors: Neural priors for side 1 (our agent)
        s2_priors: Neural priors for side 2 (opponent)
    """
    logger.debug("Calling PUCT MCTS with {} state: {}".format(index, state))
    poke_engine_state = PokeEngineState.from_string(state)

    if s1_priors is not None:
        logger.debug(f"Using neural priors for battle {index}: {s1_priors[:4]}...")

    # Run PUCT MCTS with provided priors (or None for uniform)
    res = mcts_with_puct(
        poke_engine_state,
        search_time_ms,
        c_puct=c_puct,
        s1_neural_priors=s1_priors,
        s2_neural_priors=s2_priors
    )

    # Convert PyMctsResult to picklable format (Rust objects can't be pickled)
    # Note: PyMctsResult has attributes s1, s2, iteration_count (defined in poke-engine-py/src/lib.rs)
    side_one_nodes = [
        PicklableMoveNode(
            move_choice=node.move_choice,
            visits=node.visits,
            total_score=node.total_score
        )
        for node in res.s1
    ]

    side_two_nodes = [
        PicklableMoveNode(
            move_choice=node.move_choice,
            visits=node.visits,
            total_score=node.total_score
        )
        for node in res.s2
    ] if hasattr(res, 's2') and res.s2 else []

    # Calculate total visits
    try:
        total = res.iteration_count
    except AttributeError:
        total = sum(node.visits for node in side_one_nodes)

    logger.info("PUCT Iterations {}: {}".format(index, total))

    # Return picklable result
    return PicklableMctsResult(
        side_one=side_one_nodes,
        side_two=side_two_nodes,
        total_visits=total
    )


def search_time_num_battles_randombattles(battle):
    revealed_pkmn = len(battle.opponent.reserve)
    if battle.opponent.active is not None:
        revealed_pkmn += 1

    # Safely get opponent active moves count
    opponent_active_num_moves = 0
    if battle.opponent.active is not None:
        opponent_active_num_moves = len(battle.opponent.active.moves)

    in_time_pressure = battle.time_remaining is not None and battle.time_remaining <= 60

    # it is still quite early in the battle and the pkmn in front of us
    # hasn't revealed any moves: search a lot of battles shallowly
    if (
        revealed_pkmn <= 3
        and battle.opponent.active is not None
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
    # Safely get opponent active moves count
    opponent_active_num_moves = 0
    if battle.opponent.active is not None:
        opponent_active_num_moves = len(battle.opponent.active.moves)

    in_time_pressure = battle.time_remaining is not None and battle.time_remaining <= 60

    if (
        battle.team_preview
        or (battle.opponent.active is not None and battle.opponent.active.hp > 0 and opponent_active_num_moves == 0)
        or opponent_active_num_moves < 3
    ):
        num_battles_multiplier = 1 if in_time_pressure else 2
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

    # Choose between PUCT MCTS and vanilla/neural-guided MCTS
    use_puct = FoulPlayConfig.use_neural_mcts and PUCT_AVAILABLE

    if use_puct:
        logger.info("Searching for a move using PUCT MCTS with neural priors...")
    else:
        logger.info("Searching for a move using MCTS...")

    logger.info(
        "Sampling {} battles at {}ms each".format(num_battles, search_time_per_battle)
    )

    # Query neural priors in main process before submitting to workers
    # This avoids pickling Battle objects and allows neural inference
    neural_priors_list = []
    if use_puct:
        logger.info("Computing neural priors for all battles...")
        neural_priors_list = get_neural_priors_batch(battles)
    else:
        # For vanilla MCTS, use None priors
        neural_priors_list = [(None, None) for _ in battles]

    with ProcessPoolExecutor(max_workers=FoulPlayConfig.parallelism) as executor:
        futures = []
        for index, (b, chance) in enumerate(battles):
            s1_priors, s2_priors = neural_priors_list[index]

            if use_puct:
                # Use PUCT MCTS with neural priors computed in main process
                fut = executor.submit(
                    get_result_from_puct_mcts,
                    battle_to_poke_engine_state(b).to_string(),
                    search_time_per_battle,
                    index,
                    FoulPlayConfig.neural_c_puct,
                    s1_priors,
                    s2_priors
                )
            else:
                # Use vanilla MCTS
                fut = executor.submit(
                    get_result_from_mcts,
                    battle_to_poke_engine_state(b).to_string(),
                    search_time_per_battle,
                    index,
                )
            futures.append((fut, chance, index, b))  # Store battle object

    mcts_results = [(fut.result(), chance, index) for (fut, chance, index, b) in futures]

    # For PUCT, use vanilla selection (priors already integrated)
    # For non-PUCT, optionally use post-search neural guidance
    if use_puct:
        logger.info("Using PUCT-guided move selection (priors integrated during search)...")
        choice = select_move_from_mcts_results(mcts_results)
    elif FoulPlayConfig.use_neural_mcts and NEURAL_GUIDANCE_AVAILABLE:
        logger.info("Using neural-guided move selection (post-search reranking)...")
        battle_objects = [b for (fut, chance, index, b) in futures]
        choice = select_move_with_neural_guidance(
            mcts_results,
            battle_objects,
            c_puct=FoulPlayConfig.neural_c_puct
        )

        # Fallback to vanilla MCTS if neural guidance fails
        if choice is None:
            logger.warning("Neural guidance failed, falling back to vanilla MCTS")
            choice = select_move_from_mcts_results(mcts_results)
    else:
        choice = select_move_from_mcts_results(mcts_results)

    logger.info("Choice: {}".format(choice))
    return choice
