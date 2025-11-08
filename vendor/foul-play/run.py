import asyncio
import json
import logging
import math
import os
import traceback
from copy import deepcopy
from pathlib import Path
from datetime import datetime

from config import FoulPlayConfig, init_logging, BotModes

from teams.load_team import TEAM_DIR, load_team
from fp.run_battle import pokemon_battle
from fp.websocket_client import PSWebsocketClient

from data import all_move_json
from data import pokedex
from data.mods.apply_mods import apply_mods


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METAMON_CACHE = REPO_ROOT / "metamon_cache"
os.environ.setdefault("METAMON_CACHE_DIR", str(DEFAULT_METAMON_CACHE))
DEFAULT_METAMON_CACHE.mkdir(parents=True, exist_ok=True)


logger = logging.getLogger(__name__)

TEAM_STATS_FILE = os.path.join(TEAM_DIR, "team_stats.json")
DEFAULT_UCB_EXPLORATION = 1.4


class TeamSelector:
    def __init__(
        self,
        team_group,
        stats_file=TEAM_STATS_FILE,
        exploration_constant=DEFAULT_UCB_EXPLORATION,
        half_life_games=12,
        exploit_threshold=40,
    ):
        self.team_group = team_group
        self.stats_file = stats_file
        self.exploration_constant = exploration_constant
        self.half_life_games = half_life_games
        self.exploit_threshold = exploit_threshold
        safe_group_name = (
            self.team_group.replace(os.sep, "_").replace("/", "_").strip("_")
            or "default"
        )
        self.log_file = os.path.join(TEAM_DIR, f"{safe_group_name}_game_log.jsonl")
        self._stats_data = self._load_stats_file()
        self._group_stats = self._stats_data.setdefault(self.team_group, {})
        self.team_keys = self._discover_team_files()
        selector_base_path = os.path.join(TEAM_DIR, self.team_group)
        logger.info(
            f"TeamSelector using directory {selector_base_path} "
            f"(absolute path: {os.path.abspath(selector_base_path)})"
        )
        logger.info(f"TeamSelector game log path: {self.log_file}")
        logger.info(
            f"TeamSelector initialized for {self.team_group}; "
            f"found {len(self.team_keys)} team files (stats file: {self.stats_file})"
        )
        self._ensure_team_entries()
        self._persist_stats()

    def _discover_team_files(self):
        base_path = os.path.join(TEAM_DIR, self.team_group)
        if os.path.isdir(base_path):
            team_files = []
            for file_name in sorted(os.listdir(base_path)):
                if file_name.startswith("."):
                    continue
                full_path = os.path.join(base_path, file_name)
                if os.path.isfile(full_path):
                    team_files.append(os.path.relpath(full_path, TEAM_DIR))
        elif os.path.isfile(base_path):
            team_files = [os.path.relpath(base_path, TEAM_DIR)]
        else:
            raise ValueError(
                "Path must be file or dir relative to teams: {}".format(self.team_group)
            )

        if not team_files:
            raise ValueError(
                "No team files found under {}".format(os.path.abspath(base_path))
            )

        logger.info(
            f"Discovered team files for {self.team_group}: {', '.join(team_files)} "
            f"(base path: {os.path.abspath(base_path)})"
        )

        return team_files

    def _load_stats_file(self):
        if os.path.exists(self.stats_file):
            try:
                with open(self.stats_file, "r") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return data
                    logger.warning(
                        "Team stats file %s malformed, resetting.", self.stats_file
                    )
            except (json.JSONDecodeError, OSError):
                logger.warning(
                    "Failed to read team stats file %s, resetting.", self.stats_file
                )
        return {}

    def _ensure_team_entries(self):
        updated = False
        for key in self.team_keys:
            if key not in self._group_stats:
                self._group_stats[key] = {"wins": 0, "losses": 0, "games": 0}
                updated = True
        if updated:
            self._stats_data[self.team_group] = self._group_stats

    def _persist_stats(self):
        os.makedirs(os.path.dirname(self.stats_file), exist_ok=True)
        tmp_path = "{}.tmp".format(self.stats_file)
        with open(tmp_path, "w") as f:
            json.dump(self._stats_data, f, indent=2, sort_keys=True)
        os.replace(tmp_path, self.stats_file)

    def select_team(self):
        team_key = self._select_team_key()
        team_packed, team_dict, team_file_name = load_team(team_key)
        stats = self._group_stats.get(team_key, {"wins": 0, "losses": 0, "games": 0})
        logger.info(
            f"TeamSelector selected {team_key} "
            f"(wins={stats.get('wins', 0)} losses={stats.get('losses', 0)} "
            f"games={stats.get('games', 0)})"
        )
        return team_key, team_packed, team_dict, team_file_name

    def _select_team_key(self):
        if self._should_exploit():
            logger.info(
                f"TeamSelector exploiting: wins={self._total_wins()} "
                f"losses={self._total_losses()} threshold={self.exploit_threshold}"
            )
            return self._select_highest_win_rate()

        total_games = sum(
            self._group_stats[key].get("games", 0) for key in self.team_keys
        )
        total_games = max(total_games, 1)
        best_key = None
        best_score = float("-inf")

        for key in self.team_keys:
            stats = self._group_stats[key]
            games = stats.get("games", 0)
            if games == 0:
                score = float("inf")
            else:
                win_rate = stats.get("wins", 0) / games
                exploration = math.sqrt(math.log(total_games) / games)
                decayed_c = self.exploration_constant * math.pow(
                    0.5, games / self.half_life_games
                )
                score = win_rate + decayed_c * exploration

            if score > best_score:
                best_score = score
                best_key = key

        if best_key is None:
            raise RuntimeError(
                "Failed to select a team for group {}".format(self.team_group)
            )

        return best_key

    def record_result(self, team_key, did_win):
        stats = self._group_stats.setdefault(
            team_key, {"wins": 0, "losses": 0, "games": 0}
        )
        stats["games"] += 1
        if did_win:
            stats["wins"] += 1
        else:
            stats["losses"] += 1

        self._group_stats[team_key] = stats
        self._stats_data[self.team_group] = self._group_stats
        self._persist_stats()
        logger.info(
            f"Updated stats for {team_key} -> wins={stats['wins']} "
            f"losses={stats['losses']} games={stats['games']}"
        )
        self._append_log(team_key, did_win)

    def _total_wins(self):
        return sum(self._group_stats[key].get("wins", 0) for key in self.team_keys)

    def _total_losses(self):
        total_games = sum(
            self._group_stats[key].get("games", 0) for key in self.team_keys
        )
        return total_games - self._total_wins()

    def _should_exploit(self):
        return (
            self._total_wins() >= self.exploit_threshold
            or self._total_losses() >= self.exploit_threshold
        )

    def _select_highest_win_rate(self):
        best_key = None
        best_rate = float("-inf")
        best_games = -1

        for key in self.team_keys:
            stats = self._group_stats[key]
            games = stats.get("games", 0)
            wins = stats.get("wins", 0)
            if games == 0:
                win_rate = 0.0
            else:
                win_rate = wins / games

            if (
                win_rate > best_rate
                or (win_rate == best_rate and games > best_games)
                or best_key is None
            ):
                best_key = key
                best_rate = win_rate
                best_games = games

        if best_key is None:
            logger.warning(
                "No teams available for exploitation; defaulting to first team"
            )
            best_key = self.team_keys[0]
        logger.info(
            f"Exploitation selected {best_key} with win_rate={best_rate:.4f} "
            f"games={best_games}"
        )
        return best_key

    def _append_log(self, team_key, did_win):
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        stats = self._group_stats[team_key]
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "team_key": team_key,
            "result": "win" if did_win else "loss",
            "wins": stats.get("wins", 0),
            "losses": stats.get("losses", 0),
            "games": stats.get("games", 0),
            "total_wins": self._total_wins(),
            "total_losses": self._total_losses(),
        }
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry))
            f.write("\n")
        logger.info(
            f"Appended game result to log: {entry['team_key']} {entry['result']} "
            f"(wins={entry['wins']} losses={entry['losses']} games={entry['games']})"
        )


def check_dictionaries_are_unmodified(original_pokedex, original_move_json):
    # The bot should not modify the data dictionaries
    # This is a "just-in-case" check to make sure and will stop the bot if it mutates either of them
    if original_move_json != all_move_json:
        logger.critical(
            "Move JSON changed!\nDumping modified version to `modified_moves.json`"
        )
        with open("modified_moves.json", "w") as f:
            json.dump(all_move_json, f, indent=4)
        exit(1)
    else:
        logger.debug("Move JSON unmodified!")

    if original_pokedex != pokedex:
        logger.critical(
            "Pokedex JSON changed!\nDumping modified version to `modified_pokedex.json`"
        )
        with open("modified_pokedex.json", "w") as f:
            json.dump(pokedex, f, indent=4)
        exit(1)
    else:
        logger.debug("Pokedex JSON unmodified!")


async def run_foul_play():
    logger.info("Initializing Foul Play run")
    FoulPlayConfig.configure()
    logger.info(
        f"Configuration loaded: username={FoulPlayConfig.username} "
        f"format={FoulPlayConfig.pokemon_format} "
        f"bot_mode={FoulPlayConfig.bot_mode.name} "
        f"team_name={FoulPlayConfig.team_name} "
        f"run_count={FoulPlayConfig.run_count}"
    )
    init_logging(FoulPlayConfig.log_level, FoulPlayConfig.log_to_file)
    apply_mods(FoulPlayConfig.pokemon_format)
    logger.info(f"Applied format mods for {FoulPlayConfig.pokemon_format}")

    original_pokedex = deepcopy(pokedex)
    original_move_json = deepcopy(all_move_json)
    logger.info("Copied original pokedex and move data for mutation checks")

    ps_websocket_client = await PSWebsocketClient.create(
        FoulPlayConfig.username, FoulPlayConfig.password, FoulPlayConfig.websocket_uri
    )
    logger.info(
        f"Websocket client created for {FoulPlayConfig.username} "
        f"(uri={FoulPlayConfig.websocket_uri})"
    )

    FoulPlayConfig.user_id = await ps_websocket_client.login()
    logger.info(f"Logged in with user_id {FoulPlayConfig.user_id}")

    if FoulPlayConfig.avatar is not None:
        await ps_websocket_client.avatar(FoulPlayConfig.avatar)
        logger.info(f"Avatar configured to {FoulPlayConfig.avatar}")

    battles_run = 0
    wins = 0
    losses = 0
    team_file_name = "None"
    team_dict = None
    team_selector = None
    if FoulPlayConfig.requires_team():
        logger.info(
            f"Format requires a team; initializing selector for {FoulPlayConfig.team_name}"
        )
        team_selector = TeamSelector(FoulPlayConfig.team_name)
    else:
        logger.info("Format does not require a preset team (random/battlefactory)")
    while True:
        logger.info(f"=== Preparing battle {battles_run + 1} ===")
        selected_team_key = None
        if team_selector:
            (
                selected_team_key,
                team_packed,
                team_dict,
                team_file_name,
            ) = team_selector.select_team()
            team_path = os.path.join(TEAM_DIR, selected_team_key)
            logger.info(
                f"Battle {battles_run + 1}: loaded team from {selected_team_key} "
                f"(file {team_file_name}, path {team_path})"
            )
            await ps_websocket_client.update_team(team_packed)
            logger.info(f"Battle {battles_run + 1}: team update sent to server")
        else:
            team_dict = None
            await ps_websocket_client.update_team("None")
            logger.info(
                f"Battle {battles_run + 1}: using no preset team (random/battlefactory format)"
            )

        if FoulPlayConfig.bot_mode == BotModes.challenge_user:
            logger.info(
                f"Battle {battles_run + 1}: challenging user "
                f"{FoulPlayConfig.user_to_challenge} in format {FoulPlayConfig.pokemon_format}"
            )
            await ps_websocket_client.challenge_user(
                FoulPlayConfig.user_to_challenge,
                FoulPlayConfig.pokemon_format,
            )
        elif FoulPlayConfig.bot_mode == BotModes.accept_challenge:
            logger.info(
                f"Battle {battles_run + 1}: accepting challenges in room "
                f"{FoulPlayConfig.room_name} for format {FoulPlayConfig.pokemon_format}"
            )
            await ps_websocket_client.accept_challenge(
                FoulPlayConfig.pokemon_format, FoulPlayConfig.room_name
            )
        elif FoulPlayConfig.bot_mode == BotModes.search_ladder:
            logger.info(
                f"Battle {battles_run + 1}: searching ladder for format {FoulPlayConfig.pokemon_format}"
            )
            await ps_websocket_client.search_for_match(FoulPlayConfig.pokemon_format)
        else:
            raise ValueError("Invalid Bot Mode: {}".format(FoulPlayConfig.bot_mode))

        logger.info(f"Battle {battles_run + 1}: awaiting pokemon_battle resolution")
        winner = await pokemon_battle(
            ps_websocket_client, FoulPlayConfig.pokemon_format, team_dict
        )
        logger.info(
            f"Battle {battles_run + 1}: pokemon_battle completed with winner {winner}"
        )
        if winner == FoulPlayConfig.username:
            wins += 1
            logger.info("Won with team: {}".format(team_file_name))
            if team_selector and selected_team_key is not None:
                team_selector.record_result(selected_team_key, True)
        else:
            losses += 1
            logger.info("Lost with team: {}".format(team_file_name))
            if team_selector and selected_team_key is not None:
                team_selector.record_result(selected_team_key, False)

        logger.info("W: {}\tL: {}".format(wins, losses))
        check_dictionaries_are_unmodified(original_pokedex, original_move_json)

        battles_run += 1
        if battles_run >= FoulPlayConfig.run_count:
            break
    await ps_websocket_client.close()


if __name__ == "__main__":
    try:
        asyncio.run(run_foul_play())
    except Exception:
        logger.error(traceback.format_exc())
        raise
