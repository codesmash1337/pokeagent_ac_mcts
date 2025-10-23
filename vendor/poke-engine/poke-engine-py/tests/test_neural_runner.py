import pathlib
import sys
import unittest

import torch

TEST_ROOT = pathlib.Path(__file__).resolve()
VENDOR_ROOT = TEST_ROOT.parents[3]
METAMON_ROOT = VENDOR_ROOT / "metamon"
if str(METAMON_ROOT) not in sys.path:
    sys.path.insert(0, str(METAMON_ROOT))

from dataclasses import dataclass
from typing import List, Tuple

from metamon.backend.replay_parser.str_parsing import clean_name
from metamon.backend.showdown_dex import Dex
from metamon.interface import UniversalAction, UniversalState, get_observation_space
from poke_engine import (
    Move as PEMove,
    Pokemon as PEPokemon,
    PokemonIndex,
    Side as PESide,
    State as PEState,
)
from poke_engine.neural_runner import InferenceResult, NeuralInferenceRunner


@dataclass
class PokemonSpec:
    species: str
    move_ids: List[str]
    move_pps: List[int]
    ability: str
    tera_type: str
    hp_pct: float
    item: str = "unknownitem"
    status: str = "nostatus"
    terastallized: bool = False


def _status_to_poke_engine(status: str) -> str:
    mapping = {
        "nostatus": "none",
        "brn": "burn",
        "slp": "sleep",
        "frz": "freeze",
        "par": "paralyze",
        "psn": "poison",
        "tox": "toxic",
        "fnt": "none",
    }
    status = status.lower()
    return mapping.get(status, status)


def _tera_type_to_poke_engine(tera_type: str) -> str:
    tera_type = clean_name(tera_type)
    if tera_type in {"notype", "typeless"}:
        return "typeless"
    return tera_type


def _types_from_dex(
    dex: Dex, species: str, terastallized: bool, tera_type: str
) -> Tuple[str, str]:
    entry = dex.get_pokedex_entry(species)
    base_types = [clean_name(t) for t in entry["types"]]
    while len(base_types) < 2:
        base_types.append("typeless")
    if terastallized:
        return (
            _tera_type_to_poke_engine(tera_type),
            "typeless",
        )
    return (base_types[0], base_types[1])


def _create_pokemon(dex: Dex, spec: PokemonSpec) -> PEPokemon:
    entry = dex.get_pokedex_entry(spec.species)
    if len(spec.move_ids) != 4 or len(spec.move_pps) != 4:
        raise ValueError(
            f"PokemonSpec for '{spec.species}' must provide exactly four moves"
        )

    moves = [
        PEMove(id=clean_name(move_id), pp=pp, disabled=False)
        for move_id, pp in zip(spec.move_ids, spec.move_pps)
    ]

    max_hp = 400
    hp = int(round(spec.hp_pct * max_hp))
    stats = entry["baseStats"]

    return PEPokemon(
        id=clean_name(spec.species),
        level=100,
        types=_types_from_dex(dex, spec.species, spec.terastallized, spec.tera_type),
        base_types=_types_from_dex(dex, spec.species, False, spec.tera_type),
        hp=hp,
        maxhp=max_hp,
        ability=clean_name(spec.ability),
        base_ability=clean_name(spec.ability),
        item=clean_name(spec.item),
        nature="serious",
        evs=(0, 0, 0, 0, 0, 0),
        attack=stats["atk"],
        defense=stats["def"],
        special_attack=stats["spa"],
        special_defense=stats["spd"],
        speed=stats["spe"],
        status=_status_to_poke_engine(spec.status),
        rest_turns=0,
        sleep_turns=0,
        weight_kg=float(entry.get("weightkg", 50)),
        moves=moves,
        terastallized=spec.terastallized,
        tera_type=_tera_type_to_poke_engine(spec.tera_type),
    )


def build_poke_engine_side(
    dex: Dex,
    specs: List[PokemonSpec],
    *,
    last_move_index: int,
) -> PESide:
    if len(specs) != 6:
        raise ValueError("Exactly six PokemonSpec entries are required per side")

    pokemon = [_create_pokemon(dex, spec) for spec in specs]
    return PESide(
        pokemon=pokemon,
        active_index=PokemonIndex.P0,
        last_used_move=f"move:{last_move_index}",
    )


def build_poke_engine_state_from_specs(
    dex: Dex,
    player_specs: List[PokemonSpec],
    opponent_specs: List[PokemonSpec],
    *,
    player_prev_index: int,
    opponent_prev_index: int,
) -> PEState:
    side_one = build_poke_engine_side(dex, player_specs, last_move_index=player_prev_index)
    side_two = build_poke_engine_side(dex, opponent_specs, last_move_index=opponent_prev_index)
    return PEState(
        side_one=side_one,
        side_two=side_two,
        weather="none",
        weather_turns_remaining=0,
        terrain="none",
        terrain_turns_remaining=0,
        trick_room=False,
        trick_room_turns_remaining=0,
        team_preview=False,
    )


class _FakePolicy:
    action_dim = 13


class _FakeInference:
    def __init__(self, action_dim: int) -> None:
        self.action_dim = action_dim
        self.calls = []
        self.reset_calls = 0
        self.update_calls = []

    def __call__(self, obs, legal_actions, gamma_idx=-1):
        self.calls.append(
            {
                "keys": set(obs.keys()),
                "legal_actions": list(legal_actions),
                "gamma_idx": gamma_idx,
            }
        )
        action_probs = torch.full((self.action_dim,), 1.0 / self.action_dim)
        q_values = torch.arange(self.action_dim, dtype=torch.float32)
        state_value = torch.tensor(0.5)
        return action_probs, q_values, state_value

    def reset(self):
        self.reset_calls += 1

    def update(self, reward: float, action: int, done: bool):
        self.update_calls.append((reward, action, done))


def _make_test_state():
    dex = Dex.from_gen(9)
    player_specs = [
        PokemonSpec(
            species="Iron Valiant",
            move_ids=["moonblast", "nightslash", "nuzzle", "thunderbolt"],
            move_pps=[16, 15, 20, 24],
            ability="Quark Drive",
            tera_type="Fairy",
            hp_pct=0.75,
            item="choicespecs",
        ),
        PokemonSpec(
            species="Corviknight",
            move_ids=["roost", "bravebird", "uturn", "bodypress"],
            move_pps=[16, 16, 32, 16],
            ability="Pressure",
            tera_type="Flying",
            hp_pct=1.0,
            item="leftovers",
        ),
        PokemonSpec(
            species="Toxapex",
            move_ids=["scald", "recover", "haze", "toxicspikes"],
            move_pps=[16, 16, 48, 24],
            ability="Regenerator",
            tera_type="Water",
            hp_pct=0.0,
            item="blacksludge",
            status="fnt",
        ),
        PokemonSpec(
            species="Dragapult",
            move_ids=["dragondarts", "shadowball", "uturn", "thunderbolt"],
            move_pps=[16, 16, 32, 24],
            ability="Infiltrator",
            tera_type="Ghost",
            hp_pct=1.0,
            item="choicespecs",
        ),
        PokemonSpec(
            species="Great Tusk",
            move_ids=["headlongrush", "closecombat", "rapidspin", "stealthrock"],
            move_pps=[16, 8, 40, 32],
            ability="Protosynthesis",
            tera_type="Ground",
            hp_pct=0.5,
            item="leftovers",
        ),
        PokemonSpec(
            species="Gholdengo",
            move_ids=["makeitrain", "shadowball", "focusblast", "nastyplot"],
            move_pps=[8, 24, 8, 32],
            ability="Good as Gold",
            tera_type="Steel",
            hp_pct=0.0,
            item="airballoon",
            status="fnt",
        ),
    ]

    opponent_specs = [
        PokemonSpec(
            species="Garchomp",
            move_ids=["earthquake", "swordsdance", "stoneedge", "dragonclaw"],
            move_pps=[16, 32, 8, 15],
            ability="Rough Skin",
            tera_type="Dragon",
            hp_pct=0.6,
            item="lumberry",
        ),
        PokemonSpec(
            species="Rotom-Wash",
            move_ids=["hydropump", "voltswitch", "willowisp", "protect"],
            move_pps=[8, 16, 24, 16],
            ability="Levitate",
            tera_type="Water",
            hp_pct=0.5,
            item="leftovers",
        ),
        PokemonSpec(
            species="Kingambit",
            move_ids=["kowtowcleave", "suckerpunch", "ironhead", "swordsdance"],
            move_pps=[15, 8, 15, 20],
            ability="Supreme Overlord",
            tera_type="Dark",
            hp_pct=1.0,
            item="blackglasses",
        ),
        PokemonSpec(
            species="Amoonguss",
            move_ids=["spore", "sludgebomb", "gigadrain", "pollenpuff"],
            move_pps=[15, 16, 16, 24],
            ability="Regenerator",
            tera_type="Water",
            hp_pct=0.7,
            item="rockyhelmet",
        ),
        PokemonSpec(
            species="Ting-Lu",
            move_ids=["ruination", "stompingtantrum", "spikes", "whirlwind"],
            move_pps=[10, 10, 32, 32],
            ability="Vessel of Ruin",
            tera_type="Ground",
            hp_pct=0.3,
            item="leftovers",
        ),
        PokemonSpec(
            species="Blissey",
            move_ids=["softboiled", "seismictoss", "thunderwave", "teleport"],
            move_pps=[16, 32, 32, 32],
            ability="Natural Cure",
            tera_type="Normal",
            hp_pct=0.0,
            item="leftovers",
            status="fnt",
        ),
    ]

    return build_poke_engine_state_from_specs(
        dex,
        player_specs,
        opponent_specs,
        player_prev_index=0,
        opponent_prev_index=1,
    )


class NeuralInferenceRunnerTests(unittest.TestCase):
    def setUp(self):
        observation_space = get_observation_space("TeamPreviewObservationSpace")
        fake_policy = _FakePolicy()
        fake_inference = _FakeInference(fake_policy.action_dim)
        self.runner = NeuralInferenceRunner(
            policy=fake_policy,
            observation_space=observation_space,
            device=torch.device("cpu"),
            inference=fake_inference,
        )
        self.fake_inference = fake_inference
        self.test_state = _make_test_state()

    def test_infer_produces_inference_result(self):
        result = self.runner.infer(
            self.test_state,
            battle_format="gen9ou",
        )

        self.assertIsInstance(result, InferenceResult)
        self.assertIsInstance(result.universal_state, UniversalState)
        self.assertTrue({"numbers", "text"}.issubset(result.observation.keys()))

        expected_legal = sorted(
            action.action_idx
            for action in UniversalAction.maybe_valid_actions(result.universal_state)
        )
        self.assertEqual(result.legal_actions, expected_legal)
        self.assertEqual(self.fake_inference.calls[-1]["legal_actions"], expected_legal)

        self.assertEqual(result.action_probs.shape[0], self.runner.action_dim)
        self.assertEqual(result.q_values.shape[0], self.runner.action_dim)
        self.assertAlmostEqual(result.action_probs.sum().item(), 1.0, places=6)
        self.assertAlmostEqual(result.state_value.item(), 0.5, places=6)

    def test_reset_and_update_flow(self):
        self.runner.reset()
        self.assertEqual(self.fake_inference.reset_calls, 1)

        self.runner.update(reward=0.25, action=2, done=False)
        self.assertIn((0.25, 2, False), self.fake_inference.update_calls)

    def test_custom_legal_actions(self):
        custom_actions = [0, 4, 5]
        result = self.runner.infer(
            self.test_state,
            battle_format="gen9ou",
            legal_actions=custom_actions,
        )
        self.assertEqual(result.legal_actions, custom_actions)
        self.assertEqual(
            self.fake_inference.calls[-1]["legal_actions"],
            custom_actions,
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
