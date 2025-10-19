"""
PUCT-Based Neural-Guided MCTS
Integrates neural policy priors directly into MCTS tree search using PUCT formula
"""

import logging
import numpy as np
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


class PUCTSearch:
    """
    PUCT-based neural-guided MCTS that queries neural priors at the root node
    and uses them to guide tree search via the PUCT formula in poke-engine.

    This is the "true" PUCT integration as outlined in the PRD FR-3, using
    the PUCT formula: Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
    """

    def __init__(
        self,
        policy_provider,
        state_translator,
        c_puct: float = 1.0,
        use_neural_prior: bool = True,
    ):
        """
        Initialize PUCT search.

        Args:
            policy_provider: Provider for querying neural policies
            state_translator: Translator for Battle -> observations
            c_puct: PUCT exploration constant (typical values: 0.5 - 5.0)
            use_neural_prior: Whether to use neural priors (vs uniform)
        """
        self.policy_provider = policy_provider
        self.state_translator = state_translator
        self.c_puct = c_puct
        self.use_neural_prior = use_neural_prior

        logger.info(f"Initialized PUCTSearch: c_puct={c_puct}, use_neural={use_neural_prior}")

    def run_puct_mcts(
        self,
        battles: List,
        search_time_ms: int,
    ) -> List[Tuple]:
        """
        Run PUCT MCTS with neural priors for a list of sampled battle states.

        Args:
            battles: List of Battle objects (sampled states)
            search_time_ms: Search time budget in milliseconds

        Returns:
            List of (mcts_result, sample_chance, index) tuples
        """
        try:
            from poke_engine import State as PokeEngineState, mcts_with_puct
        except ImportError:
            logger.error("Failed to import poke_engine.mcts_with_puct - is poke-engine compiled with PUCT support?")
            raise

        results = []

        for idx, battle in enumerate(battles):
            try:
                # Convert Battle to poke-engine State
                state = self._battle_to_poke_engine_state(battle)

                # Get neural priors for the root node
                s1_priors = None
                s2_priors = None

                if self.use_neural_prior:
                    try:
                        # Translate battle to observation
                        obs = self.state_translator.translate(battle)

                        # Get policy from neural network
                        policy = self.policy_provider.get_policy(obs)

                        if policy is not None:
                            # Convert 13-dim action space to move priors
                            # This is a simplified mapping - in production would need
                            # proper alignment between Metamon actions and poke-engine moves
                            s1_priors = self._map_policy_to_move_priors(policy, state, side=1)
                            # For now, use uniform priors for opponent (side 2)
                            # In self-play or opponent modeling, could query opponent policy
                            s2_priors = None

                    except Exception as e:
                        logger.warning(f"Failed to get neural priors for battle {idx}: {e}")
                        # Continue with uniform priors

                # Run PUCT MCTS
                mcts_result = mcts_with_puct(
                    state,
                    search_time_ms,
                    c_puct=self.c_puct,
                    s1_neural_priors=s1_priors,
                    s2_neural_priors=s2_priors,
                )

                # Calculate sample chance (uniform for now)
                sample_chance = 1.0 / len(battles)

                results.append((mcts_result, sample_chance, idx))

            except Exception as e:
                logger.error(f"Failed to run PUCT MCTS for battle {idx}: {e}")
                # Skip this battle
                continue

        return results

    def _battle_to_poke_engine_state(self, battle) -> "PokeEngineState":
        """
        Convert poke-env Battle object to poke-engine State.

        This is a placeholder - the actual conversion depends on foul-play's
        implementation. In practice, foul-play already does this conversion.

        Args:
            battle: poke-env Battle object

        Returns:
            poke-engine State object
        """
        # This is a simplified placeholder
        # In reality, we would use foul-play's conversion utilities
        # For now, raise an error to indicate this needs proper implementation
        raise NotImplementedError(
            "Battle to poke-engine State conversion needs to be implemented. "
            "This should use foul-play's existing conversion utilities."
        )

    def _map_policy_to_move_priors(
        self,
        policy: np.ndarray,
        state: "PokeEngineState",
        side: int
    ) -> Optional[List[float]]:
        """
        Map 13-dimensional Metamon policy to poke-engine move priors.

        Metamon action space:
        - 0-3: Moves (alphabetical order)
        - 4-8: Switches (alphabetical order)
        - 9-12: Tera moves (alphabetical order)

        poke-engine move space:
        - Varies based on available moves and switches
        - Not necessarily in the same order

        This mapping is approximate and assumes some alignment.
        In production, would need proper action space translation.

        Args:
            policy: 13-dim Metamon policy distribution
            state: poke-engine State object
            side: Which side (1 or 2)

        Returns:
            List of priors for poke-engine moves, or None if mapping fails
        """
        try:
            # Get available moves from state
            # This is a placeholder - actual implementation depends on poke-engine API
            # For now, return first 4 actions as move priors
            # and normalize

            # Simple heuristic mapping:
            # - First 4 moves get policy[0:4]
            # - Switches get policy[4:9]
            # Would need proper alignment in production

            # Placeholder: return first 10 elements normalized
            priors = policy[:10].tolist()
            total = sum(priors)
            if total > 0:
                priors = [p / total for p in priors]
                return priors
            else:
                return None

        except Exception as e:
            logger.error(f"Failed to map policy to move priors: {e}")
            return None

    def select_move_from_puct_results(
        self,
        mcts_results: List[Tuple]
    ) -> str:
        """
        Select final move from PUCT MCTS results.

        Unlike post-search reranking, PUCT already incorporates neural priors
        during search, so we can directly use visit counts.

        Args:
            mcts_results: List of (mcts_result, sample_chance, index) tuples

        Returns:
            Selected move string
        """
        final_policy = {}

        for mcts_result, sample_chance, index in mcts_results:
            for s1_option in mcts_result.side_one:
                move_choice = s1_option.move_choice
                # PUCT already used neural priors, so visit count is sufficient
                visit_prob = s1_option.visits / max(mcts_result.total_visits, 1)
                final_policy[move_choice] = (
                    final_policy.get(move_choice, 0) +
                    sample_chance * visit_prob
                )

        if not final_policy:
            logger.warning("No valid moves found in PUCT results")
            return None

        # Select move with highest aggregated probability
        sorted_moves = sorted(final_policy.items(), key=lambda x: x[1], reverse=True)

        logger.info("Top 3 moves from PUCT search:")
        for i, (move, prob) in enumerate(sorted_moves[:3]):
            logger.info(f"  {i+1}. {move}: {prob*100:.1f}%")

        # Return top move
        return sorted_moves[0][0]

    def get_stats(self) -> dict:
        """Get statistics from policy provider."""
        return {
            "c_puct": self.c_puct,
            "use_neural_prior": self.use_neural_prior,
            **self.policy_provider.get_stats(),
        }
