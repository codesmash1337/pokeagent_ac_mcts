"""
How to get UniversalState from a live Pokemon Showdown battle.

This shows exactly what you need for live battles.
"""

from poke_env.player import Player
from metamon.interface import UniversalState


class ExamplePlayer(Player):
    """
    Example player showing how to get UniversalState from a live battle.

    This is what you need:
    1. A poke-env Player class
    2. Access to the 'battle' object in choose_move()
    3. Call UniversalState.from_Battle(battle)

    That's it!
    """

    def choose_move(self, battle):
        """
        This method is called every turn when you need to choose an action.

        Args:
            battle: A poke-env Battle object representing the current game state

        Returns:
            A BattleOrder (move or switch command)
        """
        # THIS IS THE KEY LINE - Convert live battle to UniversalState
        state = UniversalState.from_Battle(battle)

        # Now 'state' is a complete UniversalState with all the fields:
        # - state.format (e.g., "gen9ou")
        # - state.player_active_pokemon (your active Pokemon)
        # - state.opponent_active_pokemon (opponent's active Pokemon)
        # - state.available_switches (Pokemon you can switch to)
        # - state.can_tera (can you terastallize?)
        # - state.opponent_teampreview (opponent's team in Gen 9)
        # - state.player_prev_move
        # - state.opponent_prev_move
        # - state.weather, state.player_conditions, etc.

        # You can print it to see what it looks like:
        print(f"Format: {state.format}")
        print(f"Player: {state.player_active_pokemon.name} ({state.player_active_pokemon.hp_pct:.1%})")
        print(f"Opponent: {state.opponent_active_pokemon.name} ({state.opponent_active_pokemon.hp_pct:.1%})")
        print(f"Can Tera: {state.can_tera}")

        # Now you can use this state with your AI/model
        # For example, convert to observation for Abra:
        # obs = observation_space(state)
        # action = model.predict(obs)

        # For this example, just return a random legal move
        return self.choose_random_move(battle)


def dummy_action_selector(obs: dict) -> int:
    """
    Dummy function that selects an action based on the observation.

    For now, just returns index 0 (first available action).
    In a real implementation, this would call your model to predict the best action.

    Args:
        obs: The observation dictionary (with 'text_tokens' and 'numbers' keys)

    Returns:
        Action index (int)
    """
    # TODO: Replace with actual model inference
    # action = model.predict(obs)
    _ = obs  # Unused for now, but will be used when model is integrated
    return 0


class LiveBattleRunner(Player):
    """
    A player that runs live battles using the observation state pipeline.

    This demonstrates the full flow:
    1. Receive battle state from websocket
    2. Convert to UniversalState via from_Battle()
    3. Convert to observation via observation_space()
    4. Pass to action selector (dummy function for now)
    5. Execute action and continue battle
    """

    def __init__(self, observation_space=None, *args, **kwargs):
        """
        Args:
            observation_space: The observation space to use (e.g., TokenizedObservationSpace).
                             If None, will use the default TeamPreviewObservationSpace.
        """
        super().__init__(*args, **kwargs)

        # Set up observation space
        if observation_space is None:
            from metamon.interface import get_observation_space, TokenizedObservationSpace
            from metamon.tokenizer import get_tokenizer

            base_obs_space = get_observation_space("TeamPreviewObservationSpace")
            tokenizer = get_tokenizer("DefaultObservationSpace-v1")
            self.observation_space = TokenizedObservationSpace(
                base_obs_space=base_obs_space,
                tokenizer=tokenizer,
            )
        else:
            self.observation_space = observation_space

        # Track battle stats
        self.turn_count = 0
        self.total_battles = 0

    def choose_move(self, battle):
        """
        Called every turn to choose an action.

        This is where the magic happens:
        - Convert battle -> UniversalState -> observation
        - Get action from dummy function
        - Convert action index to BattleOrder
        """
        try:
            # Step 1: Convert live battle to UniversalState
            state = UniversalState.from_Battle(battle)

            # Step 2: Convert UniversalState to observation
            obs = self.observation_space(state)

            # Step 3: Get action from dummy selector
            action_idx = dummy_action_selector(obs)

            # Step 4: Convert action index to BattleOrder
            # For now, just use the available moves/switches
            available_moves = battle.available_moves
            available_switches = battle.available_switches

            # Simple mapping: try to use move at index 0 if available
            if available_moves:
                move = available_moves[min(action_idx, len(available_moves) - 1)]
                self.turn_count += 1
                print(f"Turn {self.turn_count}: Choosing {move.id}")
                return self.create_order(move)
            elif available_switches:
                switch = available_switches[0]
                self.turn_count += 1
                print(f"Turn {self.turn_count}: Switching to {switch.species}")
                return self.create_order(switch)
            else:
                # Fallback to random move if nothing is available
                return self.choose_random_move(battle)

        except Exception as e:
            print(f"Error in choose_move: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to random move on error
            return self.choose_random_move(battle)

    def _battle_finished_callback(self, battle):
        """Called when a battle finishes."""
        super()._battle_finished_callback(battle)
        self.total_battles += 1
        result = "won" if battle.won else ("lost" if battle.lost else "tied")
        print(f"\nBattle {self.total_battles} {result} after {self.turn_count} turns")
        print(f"Rating: {self.rating if hasattr(self, 'rating') else 'N/A'}")
        print("-" * 60)
        self.turn_count = 0


async def run_live_battle(
    battle_format: str = "gen9ou",
    num_battles: int = 1,
    server_url: str = None,
    username: str = None,
    password: str = None,
):
    """
    Run live battles on Pokemon Showdown.

    Args:
        battle_format: The battle format (e.g., "gen9ou", "gen9randombattle")
        num_battles: Number of battles to play
        server_url: Optional custom server URL
        username: Optional username (if not provided, will be auto-generated)
        password: Optional password
    """
    print("=" * 60)
    print("Live Battle Runner")
    print("=" * 60)
    print(f"Format: {battle_format}")
    print(f"Battles: {num_battles}")
    print(f"Server: {server_url or 'showdown (default)'}")
    print("=" * 60)

    # Create player
    player_kwargs = {"battle_format": battle_format}
    if server_url:
        player_kwargs["server_configuration"] = {"url": server_url}
    if username:
        player_kwargs["username"] = username
    if password:
        player_kwargs["password"] = password

    player = LiveBattleRunner(**player_kwargs)

    # Battle on ladder
    await player.ladder(n_games=num_battles)

    print("\n" + "=" * 60)
    print("All battles completed!")
    print("=" * 60)


if __name__ == "__main__":
    import asyncio
    import sys

    # Simple CLI to run the live battle runner
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        # Parse arguments
        battle_format = sys.argv[2] if len(sys.argv) > 2 else "gen9randombattle"
        num_battles = int(sys.argv[3]) if len(sys.argv) > 3 else 1

        # Run the battle
        asyncio.run(run_live_battle(
            battle_format=battle_format,
            num_battles=num_battles,
        ))
    else:
        print("""
==================================================================
Live Battle Runner - Test Observation Pipeline with Real Battles
==================================================================

USAGE:
    python test_showdown_state.py run [format] [num_battles]

EXAMPLES:
    # Run 1 random battle (easiest for testing)
    python test_showdown_state.py run gen9randombattle 1

    # Run 3 Gen 9 OU battles
    python test_showdown_state.py run gen9ou 3

WHAT THIS DOES:
    1. Connects to Pokemon Showdown via websocket
    2. For each turn:
       - Gets battle state from websocket
       - Converts to UniversalState via from_Battle()
       - Converts to observation via observation_space()
       - Passes to dummy_action_selector() (returns index 0)
       - Executes the action in the battle
    3. Continues until battle ends
    4. Repeats for num_battles

CUSTOMIZING THE ACTION SELECTOR:
    Edit the dummy_action_selector() function to use your model:

    def dummy_action_selector(obs: dict) -> int:
        # Load your model
        action = model.predict(obs)
        return action

==================================================================
Original Documentation
==================================================================

How to Get UniversalState from Live Battles:

You need a poke-env Battle object. Here's what to provide:

1. BATTLE OBJECT - This comes from poke-env Player class

   from poke_env.player import Player
   from metamon.interface import UniversalState

   class MyPlayer(Player):
       def choose_move(self, battle):
           # 'battle' is the Battle object - this is what you need!
           state = UniversalState.from_Battle(battle)

           # Now you have a complete UniversalState
           return self.choose_random_move(battle)

==================================================================
The Battle Object Contains:
==================================================================

The 'battle' parameter in choose_move() has everything:
  - battle.active_pokemon (your active Pokemon)
  - battle.opponent_active_pokemon
  - battle.available_switches
  - battle.available_moves
  - battle.can_tera
  - battle.team (your full team)
  - battle.opponent_team
  - battle.weather
  - battle.side_conditions
  - etc.

UniversalState.from_Battle(battle) converts all of this into the
metamon UniversalState format.

==================================================================
""")
