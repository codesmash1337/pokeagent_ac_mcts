"""
Battle Adapter for Foul-Play → Metamon Compatibility

Wraps foul-play Battle objects to make them compatible with Metamon's
UniversalState.from_Battle() which expects poke-env Battle objects.
"""

import logging

logger = logging.getLogger(__name__)


class PokemonAdapter:
    """
    Adapter that makes foul-play Pokemon compatible with Metamon's expectations.

    Metamon expects Pokemon.moves to be a dict, but foul-play stores it as a list.
    """

    def __init__(self, foul_play_pokemon):
        """Wrap a foul-play Pokemon object."""
        self._pokemon = foul_play_pokemon

    @property
    def moves(self):
        """
        Return moves as a dict (Metamon expects this).
        Foul-play stores moves as a list.
        """
        if not hasattr(self._pokemon, 'moves') or not self._pokemon.moves:
            return {}

        # If already a dict, return as-is
        if isinstance(self._pokemon.moves, dict):
            return self._pokemon.moves

        # Convert list to dict using move ID as key
        # Metamon expects move IDs as keys
        moves_dict = {}
        for move in self._pokemon.moves:
            # Handle both Move objects and strings
            if hasattr(move, 'id'):
                key = move.id
            elif hasattr(move, 'name'):
                key = move.name
            else:
                key = str(move)
            moves_dict[key] = move

        return moves_dict

    def __getattr__(self, name):
        """Forward all other attributes to the wrapped Pokemon object."""
        return getattr(self._pokemon, name)


class BattleAdapter:
    """
    Adapter that makes foul-play Battle compatible with Metamon's expectations.

    This is a lightweight wrapper that provides the attributes Metamon needs
    without requiring full poke-env Battle compatibility.
    """

    def __init__(self, foul_play_battle):
        """
        Wrap a foul-play Battle object.

        Args:
            foul_play_battle: A foul-play Battle object from fp.battle.Battle
        """
        self._battle = foul_play_battle

    @property
    def fields(self):
        """
        Metamon expects 'fields' (plural), foul-play has 'field' (singular).
        Return as a dict to match poke-env format.
        """
        # Poke-env uses a dict like: {Field.ELECTRIC_TERRAIN: 1}
        # Foul-play uses a string like: "electricterrain"
        if self._battle.field:
            # Return as dict with field name as key, turns remaining as value
            return {self._battle.field: self._battle.field_turns_remaining or 0}
        return {}

    @property
    def battle_tag(self):
        return self._battle.battle_tag

    @property
    def weather(self):
        return self._battle.weather

    @property
    def side_conditions(self):
        """
        Foul-play doesn't track side conditions the same way.
        Return empty dict for now (can be enhanced later).
        """
        # TODO: Map foul-play's side condition tracking to poke-env format
        return {}

    @property
    def opponent_side_conditions(self):
        """
        Foul-play doesn't track opponent side conditions the same way.
        Return empty dict for now (can be enhanced later).
        """
        return {}

    @property
    def active_pokemon(self):
        """
        Return the active Pokemon (wrapped in PokemonAdapter).
        Foul-play uses battle.user.active
        """
        if not self._battle.user or not self._battle.user.active:
            return None
        return PokemonAdapter(self._battle.user.active)

    @property
    def opponent_active_pokemon(self):
        """
        Return the opponent's active Pokemon (wrapped in PokemonAdapter).
        Foul-play uses battle.opponent.active
        """
        if not self._battle.opponent or not self._battle.opponent.active:
            return None
        return PokemonAdapter(self._battle.opponent.active)

    @property
    def reviving(self):
        """Foul-play doesn't have reviving mechanics - always False"""
        return False

    @property
    def team(self):
        """
        Return player's team as a dict with wrapped Pokemon.
        Foul-play stores team differently - need to construct dict.
        """
        team = {}
        if self._battle.user:
            if self._battle.user.active:
                active = self._battle.user.active
                # Handle both Pokemon objects and strings
                name = active.name if hasattr(active, 'name') else str(active)
                team[name] = PokemonAdapter(active)
            for idx, pokemon in enumerate(self._battle.user.reserve):
                if pokemon:
                    # Handle both Pokemon objects and strings
                    name = pokemon.name if hasattr(pokemon, 'name') else str(pokemon)
                    team[name] = PokemonAdapter(pokemon)
        return team

    @property
    def opponent_team(self):
        """
        Return opponent's team as a dict with wrapped Pokemon.
        """
        team = {}
        if self._battle.opponent:
            if self._battle.opponent.active:
                active = self._battle.opponent.active
                # Handle both Pokemon objects and strings
                name = active.name if hasattr(active, 'name') else str(active)
                team[name] = PokemonAdapter(active)
            for idx, pokemon in enumerate(self._battle.opponent.reserve):
                if pokemon:
                    # Handle both Pokemon objects and strings
                    name = pokemon.name if hasattr(pokemon, 'name') else str(pokemon)
                    team[name] = PokemonAdapter(pokemon)
        return team

    @property
    def teampreview_opponent_team(self):
        """Return list of opponent Pokemon seen in team preview"""
        # Foul-play doesn't store this explicitly
        return []

    @property
    def force_switch(self):
        return self._battle.force_switch

    @property
    def won(self):
        """Battle won status - foul-play doesn't track this"""
        return False

    @property
    def lost(self):
        """Battle lost status - foul-play doesn't track this"""
        return False

    @property
    def can_tera(self):
        """Whether terastallization is available - foul-play doesn't track this"""
        return None

    @property
    def available_moves(self):
        """Get available moves for the active Pokemon"""
        if not self.active_pokemon:
            return []
        # Foul-play stores moves as a list, not a dict
        if not hasattr(self.active_pokemon, 'moves'):
            return []
        moves = self.active_pokemon.moves
        # Handle both list and dict cases
        if isinstance(moves, dict):
            return list(moves.values())
        elif isinstance(moves, list):
            return moves
        return []

    @property
    def available_switches(self):
        """Get available switches (reserve Pokemon)"""
        if not self._battle.user:
            return []
        # Return reserve Pokemon that aren't fainted
        return [p for p in self._battle.user.reserve if p and (not hasattr(p, 'fainted') or not p.fainted)]

    def __getattr__(self, name):
        """
        Fallback: forward any unknown attributes to the wrapped battle object.
        This allows accessing foul-play-specific attributes directly.
        """
        return getattr(self._battle, name)


def adapt_battle(foul_play_battle):
    """
    Convenience function to wrap a foul-play Battle.

    Args:
        foul_play_battle: A foul-play Battle object

    Returns:
        BattleAdapter instance
    """
    return BattleAdapter(foul_play_battle)
