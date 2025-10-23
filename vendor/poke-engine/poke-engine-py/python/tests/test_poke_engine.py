import os
import pathlib
import sys

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import pytest
import torch

from metamon.interface import get_observation_space
from poke_engine import NeuralInferenceRunner

from poke_engine import (
    State,
    Side,
    Move,
    Pokemon,
    PokemonIndex,
    monte_carlo_tree_search,
    generate_instructions,
    calculate_damage,
    iterative_deepening_expectiminimax,
)

state = State(
    side_one=Side(
        pokemon=[
            Pokemon(
                id="squirtle",
                level=100,
                types=("water", "typeless"),
                base_types=("water", "rock"),
                hp=100,
                maxhp=100,
                attack=100,
                defense=100,
                special_attack=100,
                special_defense=100,
                speed=100,
                status="none",
                moves=[
                    Move(id="watergun", pp=32),
                    Move(id="tackle", pp=32),
                    Move(id="quickattack", pp=32),
                    Move(id="leer", pp=32),
                ],
            ),
        ]
    ),
    side_two=Side(
        pokemon=[
            Pokemon(
                id="charmander",
                level=100,
                types=("fire", "typeless"),
                hp=100,
                maxhp=100,
                attack=100,
                defense=100,
                special_attack=100,
                special_defense=100,
                speed=100,
                status="none",
                moves=[
                    Move(id="ember", pp=32),
                    Move(id="tackle", pp=32),
                    Move(id="quickattack", pp=32),
                    Move(id="leer", pp=32),
                ],
            ),
        ]
    ),
    weather="none",
    weather_turns_remaining=-1,
    terrain="none",
    terrain_turns_remaining=-1,
    trick_room=False,
    trick_room_turns_remaining=-1,
)


class _DemoPolicy:
    action_dim = 13


class _DemoInference:
    def __init__(self, action_dim: int) -> None:
        self.action_dim = action_dim

    def __call__(self, obs, legal_actions, gamma_idx=-1):
        probs = torch.zeros(self.action_dim)
        if legal_actions:
            mass = 1.0 / len(legal_actions)
            for idx in legal_actions:
                probs[idx] = mass
        q_values = torch.arange(self.action_dim, dtype=torch.float32)
        state_value = torch.tensor(float(len(legal_actions)))
        return probs, q_values, state_value

    def reset(self):  # pragma: no cover - demo helper is stateless
        pass

    def update(self, reward: float, action: int, done: bool):  # pragma: no cover
        pass


def _build_demo_state() -> State:
    def _make_pokemon(name: str) -> Pokemon:
        return Pokemon(
            id=name,
            level=100,
            types=("water", "typeless"),
            base_types=("water", "typeless"),
            hp=100,
            maxhp=100,
            attack=100,
            defense=100,
            special_attack=100,
            special_defense=100,
            speed=100,
            status="none",
            moves=[
                Move(id="watergun", pp=32),
                Move(id="tackle", pp=32),
                Move(id="quickattack", pp=32),
                Move(id="leer", pp=32),
            ],
        )

    side_one_species = [
        "Squirtle",
        "Wartortle",
        "Blastoise",
        "Pikachu",
        "Raichu",
        "Lapras",
    ]
    side_two_species = [
        "Charmander",
        "Charmeleon",
        "Charizard",
        "Vulpix",
        "Ninetales",
        "Growlithe",
    ]

    side_one = Side(
        pokemon=[_make_pokemon(species.lower()) for species in side_one_species],
        active_index=PokemonIndex.P0,
    )
    side_two = Side(
        pokemon=[_make_pokemon(species.lower()) for species in side_two_species],
        active_index=PokemonIndex.P0,
    )

    return State(
        side_one=side_one,
        side_two=side_two,
        weather="none",
        weather_turns_remaining=-1,
        terrain="none",
        terrain_turns_remaining=-1,
        trick_room=False,
        trick_room_turns_remaining=-1,
    )


def test_state_can_be_converted_to_and_from_a_string():
    print(state)
    serialized = state.to_string()
    State.from_string(serialized)
    serialized_again = state.to_string()
    assert serialized == serialized_again


def test_monte_carlo_search():
    monte_carlo_tree_search(state, 10)


def test_iterative_deepening_search():
    iterative_deepening_expectiminimax(state, 10)


def test_get_instructions():
    generate_instructions(state, "watergun", "ember")


def test_calculate_damage():
    calculate_damage(state, "watergun", "ember", True)


def test_generate_instructions_errors_when_move_does_not_exist():
    with pytest.raises(ValueError):
        generate_instructions(state, "not_a_move", "ember")


def test_neural_runner_demo(capsys):
    observation_space = get_observation_space("TeamPreviewObservationSpace")
    runner = NeuralInferenceRunner(
        policy=_DemoPolicy(),
        observation_space=observation_space,
        device=torch.device("cpu"),
        inference=_DemoInference(_DemoPolicy().action_dim),
    )

    demo_state = _build_demo_state()
    result = runner.infer(demo_state, battle_format="gen9ou")
    print("Neural runner demo action_probs:", result.action_probs.tolist())
    print("Neural runner demo state_value:", result.state_value.item())

    captured = capsys.readouterr()
    assert "Neural runner demo action_probs" in captured.out


@pytest.mark.skipif(
    not os.environ.get("METAMON_CACHE_DIR"),
    reason="Requires METAMON_CACHE_DIR to download Abra",
)
def test_neural_runner_with_abra_prints_outputs(capsys):
    runner = NeuralInferenceRunner.from_pretrained("Abra")
    demo_state = _build_demo_state()

    result = runner.infer(demo_state, battle_format="gen9ou")
    print("Abra runner action_probs:", result.action_probs.tolist())
    print("Abra runner state_value:", result.state_value.item())

    captured = capsys.readouterr()
    assert "Abra runner action_probs" in captured.out
