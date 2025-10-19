"""
Neural-Guided Search
Modified version of foul-play's search that uses neural policy priors
"""

import logging
import random
from typing import List, Tuple
from copy import deepcopy

from .policy_client import PolicyClient

logger = logging.getLogger(__name__)


class NeuralGuidedSearch:
    """
    Extends foul-play's MCTS search to use neural policy priors.

    Integrates with Policy Client to query Metamon model and use
    policy probabilities to guide MCTS exploration.
    """

    def __init__(
        self,
        policy_client: PolicyClient,
        c_puct: float = 1.0,
        use_neural_prior: bool = True,
        fallback_to_uniform: bool = True
    ):
        """
        Initialize neural-guided search.

        Args:
            policy_client: Client for querying neural policies
            c_puct: Exploration constant for PUCT formula
            use_neural_prior: Whether to use neural priors (vs uniform)
            fallback_to_uniform: Fallback to uniform if neural query fails
        """
        self.policy_client = policy_client
        self.c_puct = c_puct
        self.use_neural_prior = use_neural_prior
        self.fallback_to_uniform = fallback_to_uniform

        logger.info(f"Initialized NeuralGuidedSearch: c_puct={c_puct}, "
                   f"use_neural={use_neural_prior}")

    def select_move_with_neural_prior(
        self,
        mcts_results: List[Tuple],
        states: List[str]
    ) -> str:
        """
        Select move from MCTS results, re-weighted by neural policy priors.

        This is a modified version of foul-play's select_move_from_mcts_results
        that incorporates neural policy guidance.

        Args:
            mcts_results: List of (MctsResult, sample_chance, index) tuples
            states: List of state strings corresponding to sampled battles

        Returns:
            Selected move string
        """
        if not self.use_neural_prior:
            # Fall back to original MCTS selection
            return self._select_move_vanilla(mcts_results)

        # Get neural policies for all sampled states
        neural_policies = self.policy_client.get_policy_batch(states)

        if neural_policies is None and self.fallback_to_uniform:
            logger.warning("Neural policy query failed, falling back to vanilla MCTS")
            return self._select_move_vanilla(mcts_results)

        # Combine MCTS visit counts with neural priors
        final_policy = {}

        for (mcts_result, sample_chance, index), neural_policy in zip(mcts_results, neural_policies):
            if neural_policy is None:
                # Skip this sample if neural policy unavailable
                continue

            # Convert neural policy to move choices
            # Note: This requires mapping between Metamon actions and MCTS moves
            # For now, use simplified logic

            for s1_option in mcts_result.side_one:
                move_choice = s1_option.move_choice

                # PUCT-inspired combination:
                # P(move) = (MCTS_visits / total) * neural_prior * sample_chance
                mcts_prob = s1_option.visits / max(mcts_result.total_visits, 1)
                neural_prob = self._map_move_to_neural_action(move_choice, neural_policy)

                combined_prob = (
                    (mcts_prob ** 0.5) *  # Square root to reduce MCTS dominance
                    (neural_prob ** 0.5) *
                    sample_chance
                )

                final_policy[move_choice] = final_policy.get(move_choice, 0) + combined_prob

                logger.debug(
                    f"Move {move_choice}: mcts={mcts_prob:.3f}, "
                    f"neural={neural_prob:.3f}, combined={combined_prob:.3f}"
                )

        if not final_policy:
            logger.warning("No valid moves found, falling back to vanilla MCTS")
            return self._select_move_vanilla(mcts_results)

        # Normalize probabilities
        total_prob = sum(final_policy.values())
        final_policy = {move: prob / total_prob for move, prob in final_policy.items()}

        # Sort by probability
        sorted_moves = sorted(final_policy.items(), key=lambda x: x[1], reverse=True)

        logger.info("Top 3 moves after neural guidance:")
        for i, (move, prob) in enumerate(sorted_moves[:3]):
            logger.info(f"  {i+1}. {move}: {prob*100:.1f}%")

        # Sample from top moves (with some randomization)
        highest_prob = sorted_moves[0][1]
        top_moves = [(m, p) for m, p in sorted_moves if p >= highest_prob * 0.75]

        choice = random.choices(top_moves, weights=[p for _, p in top_moves])[0]
        return choice[0]

    def _select_move_vanilla(self, mcts_results: List[Tuple]) -> str:
        """
        Original MCTS move selection (no neural guidance).

        Args:
            mcts_results: List of MCTS results

        Returns:
            Selected move string
        """
        final_policy = {}

        for mcts_result, sample_chance, index in mcts_results:
            for s1_option in mcts_result.side_one:
                mcts_prob = s1_option.visits / max(mcts_result.total_visits, 1)
                final_policy[s1_option.move_choice] = (
                    final_policy.get(s1_option.move_choice, 0) +
                    sample_chance * mcts_prob
                )

        sorted_moves = sorted(final_policy.items(), key=lambda x: x[1], reverse=True)

        # Consider top moves
        highest_prob = sorted_moves[0][1]
        top_moves = [(m, p) for m, p in sorted_moves if p >= highest_prob * 0.75]

        choice = random.choices(top_moves, weights=[p for _, p in top_moves])[0]
        return choice[0]

    def _map_move_to_neural_action(self, move_choice: str, neural_policy: "np.ndarray") -> float:
        """
        Map a poke-engine move choice to a neural action probability.

        This is a simplified mapping. In production, would need proper
        action space alignment between poke-engine and Metamon.

        Args:
            move_choice: Move choice string from poke-engine
            neural_policy: 13-dim policy distribution from Metamon

        Returns:
            Probability for this move
        """
        # Simplified logic: parse move_choice and map to action index
        # Format examples: "move 0", "switch 2", "move 1 terastallize"

        if "switch" in move_choice.lower():
            # Extract switch index (0-4 maps to actions 4-8)
            try:
                switch_idx = int(move_choice.split()[1])
                action_idx = 4 + switch_idx
                return neural_policy[action_idx]
            except:
                return 1.0 / 13  # Uniform fallback

        elif "terastallize" in move_choice.lower():
            # Tera moves (actions 9-12)
            try:
                move_idx = int(move_choice.split()[1])
                action_idx = 9 + move_idx
                return neural_policy[action_idx]
            except:
                return 1.0 / 13

        else:
            # Regular move (actions 0-3)
            try:
                move_idx = int(move_choice.split()[1])
                action_idx = move_idx
                return neural_policy[action_idx]
            except:
                return 1.0 / 13

    def get_stats(self) -> dict:
        """Get statistics from policy client."""
        return self.policy_client.get_stats()
