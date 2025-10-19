"""
State Translation Layer
Converts poke-engine State (Rust) to Metamon observation format (Python)
"""

import logging
from typing import Dict, List, Optional
import numpy as np

# These would be imported from the actual packages
# For now, using placeholder imports
try:
    from poke_engine import State as PokeEngineState
except ImportError:
    # Placeholder for development
    PokeEngineState = object

logger = logging.getLogger(__name__)


class StateTranslator:
    """
    Translates poke-engine State objects to Metamon observation format.

    Handles:
    - Enum to string conversion for species, moves, abilities
    - Stat normalization
    - Type effectiveness calculations
    - Field condition encoding
    - Partial observability (unrevealed opponent Pokemon)
    """

    def __init__(self, observation_space_type="TeamPreviewObservationSpace"):
        """
        Initialize the state translator.

        Args:
            observation_space_type: Which Metamon observation space to use
        """
        self.observation_space_type = observation_space_type

        # Load species/move mappings
        self._load_mappings()

        # Cache for frequently accessed conversions
        self._cache = {}

        logger.info(f"Initialized StateTranslator with {observation_space_type}")

    def _load_mappings(self):
        """Load mappings from poke-engine enums to Metamon strings."""
        # In production, these would load from actual data files
        # For now, create minimal mappings for common Pokemon

        # Example species mapping (poke-engine index -> species name)
        self.species_map = {
            0: "bulbasaur",
            1: "ivysaur",
            2: "venusaur",
            # ... (would be populated from poke-engine data)
        }

        # Example move mapping
        self.move_map = {
            0: "tackle",
            1: "thundershock",
            2: "watergun",
            # ... (would be populated from poke-engine data)
        }

        # Type effectiveness matrix (simplified)
        self.type_chart = self._build_type_chart()

        logger.debug("Loaded species, move, and type mappings")

    def _build_type_chart(self) -> Dict:
        """Build type effectiveness chart."""
        # Simplified type chart
        # In production, would use complete 18x18 matrix
        return {
            "normal": {"ghost": 0.0, "rock": 0.5, "steel": 0.5},
            "fire": {"fire": 0.5, "water": 0.5, "grass": 2.0, "ice": 2.0, "bug": 2.0, "steel": 2.0},
            "water": {"fire": 2.0, "water": 0.5, "grass": 0.5, "ground": 2.0, "rock": 2.0},
            # ... (would include all 18 types)
        }

    def translate(self, state: str, legal_actions: Optional[List[int]] = None) -> Dict[str, np.ndarray]:
        """
        Translate a single poke-engine state string to Metamon observation.

        Args:
            state: Serialized poke-engine state string
            legal_actions: List of legal action indices (for masking)

        Returns:
            Dictionary containing observation tensors compatible with Metamon
        """
        # Parse the state string (simplified - actual parsing more complex)
        state_data = self._parse_state_string(state)

        # Build observation dictionary
        obs = {}

        # Player's team info
        obs["my_team"] = self._encode_team(state_data.get("side_one", {}), visible=True)

        # Opponent's team info (may be partially hidden)
        obs["opp_team"] = self._encode_team(state_data.get("side_two", {}), visible=False)

        # Field conditions
        obs["field"] = self._encode_field_conditions(state_data)

        # Battle metadata
        obs["metadata"] = self._encode_metadata(state_data)

        # Legal action mask
        if legal_actions is not None:
            obs["legal_actions"] = self._encode_legal_actions(legal_actions)
        else:
            # Default: all actions legal
            obs["legal_actions"] = np.ones(13, dtype=np.float32)

        return obs

    def batch_translate(self, states: List[str], legal_actions_list: Optional[List[List[int]]] = None) -> Dict[str, np.ndarray]:
        """
        Translate multiple states in batch.

        Args:
            states: List of serialized state strings
            legal_actions_list: List of legal action lists for each state

        Returns:
            Dictionary with batched observation tensors
        """
        if legal_actions_list is None:
            legal_actions_list = [None] * len(states)

        batch_obs = [self.translate(s, la) for s, la in zip(states, legal_actions_list)]

        # Stack into batch dimension
        batched = {}
        for key in batch_obs[0].keys():
            batched[key] = np.stack([obs[key] for obs in batch_obs], axis=0)

        return batched

    def _parse_state_string(self, state_str: str) -> Dict:
        """
        Parse poke-engine state string into structured data.

        Note: This is a simplified parser. The actual poke-engine state
        format is more complex and would require proper deserialization.
        """
        # Placeholder parsing
        # In production, would use poke_engine.State.from_string()
        return {
            "side_one": {
                "active": {"species": "pikachu", "hp": 100, "max_hp": 100, "moves": ["thunderbolt", "quickattack"]},
                "bench": []
            },
            "side_two": {
                "active": {"species": "charizard", "hp": 150, "max_hp": 150, "moves": ["flamethrower"]},
                "bench": []
            },
            "weather": "none",
            "terrain": "none",
            "trick_room": False
        }

    def _encode_team(self, team_data: Dict, visible: bool) -> np.ndarray:
        """
        Encode team information.

        Args:
            team_data: Team data dictionary
            visible: Whether all Pokemon are visible (player's team) or partially hidden (opponent)

        Returns:
            Encoded team tensor
        """
        # Placeholder encoding
        # Shape: [max_team_size, feature_dim]
        # Features: species_id, hp_ratio, stat_mods, status, etc.

        max_team_size = 6
        feature_dim = 32  # Simplified

        team_tensor = np.zeros((max_team_size, feature_dim), dtype=np.float32)

        # Encode active Pokemon
        active = team_data.get("active", {})
        if active:
            team_tensor[0, 0] = 1.0  # Active flag
            team_tensor[0, 1] = active.get("hp", 0) / max(active.get("max_hp", 1), 1)  # HP ratio
            # ... (would encode moves, stats, etc.)

        # Encode bench
        bench = team_data.get("bench", [])
        for i, pkmn in enumerate(bench[:5]):  # Max 5 bench Pokemon
            team_tensor[i+1, 1] = pkmn.get("hp", 0) / max(pkmn.get("max_hp", 1), 1)
            if not visible:
                # For opponent, mark as unrevealed
                team_tensor[i+1, 2] = 0.0  # Unrevealed flag

        return team_tensor

    def _encode_field_conditions(self, state_data: Dict) -> np.ndarray:
        """Encode weather, terrain, and other field conditions."""
        # Placeholder: [weather_id, terrain_id, trick_room, ...]
        field_tensor = np.zeros(10, dtype=np.float32)

        weather_map = {"none": 0, "sun": 1, "rain": 2, "sand": 3, "hail": 4}
        field_tensor[0] = weather_map.get(state_data.get("weather", "none"), 0)

        terrain_map = {"none": 0, "electric": 1, "grassy": 2, "psychic": 3, "misty": 4}
        field_tensor[1] = terrain_map.get(state_data.get("terrain", "none"), 0)

        field_tensor[2] = 1.0 if state_data.get("trick_room", False) else 0.0

        return field_tensor

    def _encode_metadata(self, state_data: Dict) -> np.ndarray:
        """Encode battle metadata (turn number, etc.)."""
        metadata = np.zeros(5, dtype=np.float32)
        metadata[0] = state_data.get("turn", 0) / 100.0  # Normalized turn number
        return metadata

    def _encode_legal_actions(self, legal_actions: List[int]) -> np.ndarray:
        """
        Encode legal action mask.

        Args:
            legal_actions: List of legal action indices

        Returns:
            Binary mask array of length 13 (Metamon action space size)
        """
        mask = np.zeros(13, dtype=np.float32)
        for action_idx in legal_actions:
            if 0 <= action_idx < 13:
                mask[action_idx] = 1.0
        return mask

    def get_cache_stats(self) -> Dict:
        """Return cache statistics for monitoring."""
        return {
            "cache_size": len(self._cache),
            "cache_capacity": 1000,  # Could be configurable
        }

    def clear_cache(self):
        """Clear the translation cache."""
        self._cache.clear()
        logger.debug("Cleared state translation cache")
