"""Simple Pokemon Showdown tournament using ladder battles."""

import argparse
import json
import os
import sys
import time
import uuid
import shutil
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import csv

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Repo setup
REPO_ROOT = Path(__file__).resolve().parent
VENDOR_ROOT = REPO_ROOT / "vendor" / "metamon"
if str(VENDOR_ROOT) not in sys.path:
    sys.path.append(str(VENDOR_ROOT))

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("METAMON_CACHE_DIR", str(REPO_ROOT / "metamon_cache"))

from metamon.rl.pretrained import get_pretrained_model  # type: ignore
from metamon.rl.evaluate import pretrained_vs_local_ladder  # type: ignore
from metamon.env.wrappers import TeamSet  # type: ignore


def copy_single_team_to_dir(team_file: Path, dest_root: Path) -> Path:
    """Copy a team file to a directory structure expected by TeamSet."""
    dest_root.mkdir(parents=True, exist_ok=True)
    dest_dir = dest_root / team_file.stem
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_dir / team_file.name
    shutil.copy2(team_file, dest_file)
    return dest_dir


def generate_username(prefix: str, use_uuid: bool = True) -> str:
    """Generate username for Showdown."""
    base = prefix.replace("-", "").replace("_", "")[:10]
    if use_uuid:
        suffix = uuid.uuid4().hex[:6]
        return f"{base}{suffix}".lower()
    else:
        return base.lower()


def resolve_team_argument(value: Optional[str]) -> Tuple[Optional[Path], Optional[str]]:
    """Return (path_if_exists, team_name) for a CLI team argument."""
    if not value:
        return None, None

    candidate = Path(value)
    if candidate.exists():
        resolved = candidate.resolve()
        return resolved, resolved.stem

    return None, candidate.stem


def dedupe_paths(paths: List[Path]) -> List[Path]:
    """Remove duplicate paths while preserving order."""
    seen = set()
    deduped = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        deduped.append(resolved)
        seen.add(resolved)
    return deduped


def run_agent_on_ladder(
    username: str, battle_format: str, team_dir: Path, battles: int, out_dir: Path
):
    """Run a single agent on ladder."""
    print(
        f"Parameters: username={username}, battle_format={battle_format}, team_dir={team_dir}, battles={battles}, out_dir={out_dir}"
    )
    pm = get_pretrained_model("Abra")
    team_set = TeamSet(str(team_dir), battle_format)
    pretrained_vs_local_ladder(
        pretrained_model=pm,
        username=username,
        battle_format=battle_format,
        team_set=team_set,
        total_battles=battles,
        avatar=None,
        checkpoint=None,
        battle_backend="poke-env",
        save_trajectories_to=None,
        save_team_results_to=str(out_dir),
        log_to_wandb=False,
    )


class SimplePokemonTournament:
    """Round-robin tournament where each team plays every other team N times."""

    def __init__(
        self,
        team_files: List[Path],
        battle_format: str,
        results_dir: Path,
        use_uuid: bool = True,
        use_ratings: bool = False,
        ratings_file: Optional[Path] = None,
        backup_dir: Optional[Path] = None,
        focus_team: Optional[str] = None,
        opponent_team: Optional[str] = None,
    ):
        self.team_files = team_files
        self.team_names = [f.stem for f in team_files]
        self.battle_format = battle_format
        self.results_dir = results_dir
        self.use_uuid = use_uuid
        self.use_ratings = use_ratings
        self.backup_dir = backup_dir or (REPO_ROOT / "backups")
        self.focus_team = focus_team
        self.opponent_team = opponent_team
        self.results_dir.mkdir(parents=True, exist_ok=True)

        if self.opponent_team and not self.focus_team:
            raise ValueError("opponent_team specified without focus_team")

        if self.focus_team and self.focus_team not in self.team_names:
            raise ValueError(f"focus_team '{self.focus_team}' not found in loaded teams")

        if self.opponent_team and self.opponent_team not in self.team_names:
            raise ValueError(
                f"opponent_team '{self.opponent_team}' not found in loaded teams"
            )

        if self.focus_team and self.opponent_team and self.focus_team == self.opponent_team:
            raise ValueError("focus_team and opponent_team must be different")

        # Tournament tracking
        n = len(team_files)
        self.battles = [[0 for _ in range(n)] for _ in range(n)]
        self.wins = [[0 for _ in range(n)] for _ in range(n)]
        self.pairings_completed = 0

        # ELO rating system (optional)
        if self.use_ratings:
            self.ratings_file = ratings_file or (results_dir / "ratings.json")
            self.elo_ratings = self._load_ratings()
        else:
            self.elo_ratings = None

        # Results CSV
        self.results_csv = results_dir / "results.csv"
        if not self.results_csv.exists():
            self.results_csv.write_text(
                "team_a_name,team_b_name,team_a_wins,team_a_losses,team_b_wins,team_b_losses\n"
            )

        # Load existing battle data from CSV
        self._load_existing_battle_data()

    def generate_pairings(self, target_battles: int):
        """Generate team pairings with remaining battles needed."""
        pairings = []  # (team_a_idx, team_b_idx, battles_needed)
        skipped_pairings = []

        for i in range(len(self.team_files)):
            for j in range(i + 1, len(self.team_files)):
                team_a_name = self.team_files[i].stem
                team_b_name = self.team_files[j].stem

                if not self._pairing_allowed(team_a_name, team_b_name):
                    continue

                # Check existing battle count for this pairing
                existing_battles = self.battles[i][j]  # Stored in lower triangle

                if existing_battles < target_battles:
                    battles_needed = target_battles - existing_battles
                    pairings.append((i, j, battles_needed))
                else:
                    skipped_pairings.append((i, j, existing_battles))

        return pairings, skipped_pairings

    def run_tournament(self, battles_per_pairing: int = 5):
        """Run complete round-robin tournament."""
        pairings, skipped_pairings = self.generate_pairings(battles_per_pairing)
        total_pairings = len(pairings)
        total_skipped = len(skipped_pairings)

        # Calculate total battles needed
        total_battles_needed = sum(battles_needed for _, _, battles_needed in pairings)

        print("Starting round-robin tournament:")
        print(f"  {len(self.team_files)} teams")
        print(f"  {total_pairings} pairings need more battles")
        if total_skipped > 0:
            print(f"  {total_skipped} pairings already completed (skipped)")
        print(f"  {battles_per_pairing} target battles per pairing")
        print(f"  {total_battles_needed} total battles to run")
        print()

        if self.focus_team and self.opponent_team:
            print(
                f"🔍 Restricted to matchup: {self.focus_team} vs {self.opponent_team}"
            )
            print()
        elif self.focus_team:
            print(f"🔍 Restricted to pairings involving: {self.focus_team}")
            print()

        if total_skipped > 0:
            print("Skipped pairings (already completed):")
            for i, j, count in skipped_pairings:
                team_a = self.team_files[i].stem
                team_b = self.team_files[j].stem
                print(f"  {team_a} vs {team_b}: {count} battles completed")
            print()

        if total_pairings == 0:
            print("🎉 All pairings already completed! No new battles to run.")
            return

        for pairing_num, (team_a_idx, team_b_idx, battles_needed) in enumerate(
            pairings, 1
        ):
            team_a_file = self.team_files[team_a_idx]
            team_b_file = self.team_files[team_b_idx]

            print(
                f"Pairing {pairing_num}/{total_pairings}: {team_a_file.stem} vs {team_b_file.stem} ({battles_needed} battles needed)"
            )

            self._run_pairing(team_a_file, team_b_file, battles_needed, pairing_num)
            self.pairings_completed += 1

            print(f"Completed pairing {pairing_num}/{total_pairings}")
            print()

        print("🎉 Tournament complete!")

        # Save backup of results CSV
        self._save_results_backup()

    def _run_pairing(
        self, team_a_file: Path, team_b_file: Path, battles: int, pairing_num: int
    ):
        """Run battles between exactly two teams."""
        # Create temporary directories for this pairing
        tmp_root = self.results_dir / f"tmp_pairing_{pairing_num}"
        tmp_root.mkdir(parents=True, exist_ok=True)

        try:
            # Set up team directories
            team_a_dir = copy_single_team_to_dir(team_a_file, tmp_root)
            team_b_dir = copy_single_team_to_dir(team_b_file, tmp_root)

            # Generate usernames
            username_a = generate_username(team_a_file.stem, use_uuid=self.use_uuid)
            username_b = generate_username(team_b_file.stem, use_uuid=self.use_uuid)

            username_to_team = {
                username_a: team_a_file.stem,
                username_b: team_b_file.stem,
            }

            print(f"  Queuing {username_a} and {username_b} for {battles} battles")

            def run_agent_wrapper(args):
                username, team_dir = args
                try:
                    run_agent_on_ladder(
                        username,
                        self.battle_format,
                        team_dir,
                        battles,
                        self.results_dir,
                    )
                    return f"✅ {username} completed"
                except Exception as e:
                    return f"❌ {username} failed: {e}"

            # Run exactly 2 agents concurrently
            start_time = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                future_a = executor.submit(run_agent_wrapper, (username_a, team_a_dir))
                time.sleep(0.5)  # Small stagger
                future_b = executor.submit(run_agent_wrapper, (username_b, team_b_dir))

                # Wait for both to complete
                result_a = future_a.result()
                result_b = future_b.result()

                print(f"  {result_a}")
                print(f"  {result_b}")

            elapsed = time.time() - start_time
            print(f"  Battles completed in {elapsed:.1f} seconds")

            # Process results
            self._process_battle_logs(username_to_team)

        finally:
            # Cleanup
            try:
                shutil.rmtree(tmp_root, ignore_errors=True)
            except Exception:
                pass

    def _process_battle_logs(self, username_to_team: Dict[str, str]):
        """Process battle log files and update tournament state."""
        print("Processing battle logs...")

        # Find all battle log files from this round
        log_files = list(
            self.results_dir.glob(f"battle_log_*_{self.battle_format}.csv")
        )

        battles_processed = 0
        pair_results = {}  # (team_a, team_b) -> [wins_a, losses_a, wins_b, losses_b]

        for log_file in log_files:
            try:
                with open(log_file, "r") as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    if not header:
                        continue

                    for row in reader:
                        if len(row) < 4:
                            continue

                        # player_username = row[0].strip()  # Not used
                        team_file = row[1].strip()
                        opponent_username = row[2].strip()
                        result = row[3].strip().upper()

                        # Get team names
                        player_team = Path(team_file).stem
                        opponent_team = username_to_team.get(opponent_username)

                        if (
                            player_team not in self.team_names
                            or opponent_team not in self.team_names
                        ):
                            continue

                        # Ensure consistent ordering for pair tracking
                        team_a, team_b = sorted([player_team, opponent_team])

                        if (team_a, team_b) not in pair_results:
                            pair_results[(team_a, team_b)] = [
                                0,
                                0,
                                0,
                                0,
                            ]  # a_wins, a_losses, b_wins, b_losses

                        if player_team == team_a:
                            if result == "WIN":
                                pair_results[(team_a, team_b)][0] += 1  # team_a wins
                                pair_results[(team_a, team_b)][3] += 1  # team_b losses
                            elif result == "LOSS":
                                pair_results[(team_a, team_b)][1] += 1  # team_a losses
                                pair_results[(team_a, team_b)][2] += 1  # team_b wins
                        else:  # player_team == team_b
                            if result == "WIN":
                                pair_results[(team_a, team_b)][2] += 1  # team_b wins
                                pair_results[(team_a, team_b)][1] += 1  # team_a losses
                            elif result == "LOSS":
                                pair_results[(team_a, team_b)][3] += 1  # team_b losses
                                pair_results[(team_a, team_b)][0] += 1  # team_a wins

                        battles_processed += 1

            except Exception as e:
                print(f"Error processing {log_file}: {e}")
                continue

        print(f"Processed {battles_processed} battles from {len(log_files)} log files")

        adjusted_pair_results = {}
        for (team_a, team_b), results in pair_results.items():
            adj = []
            warn = False
            for value in results:
                if value % 2 != 0:
                    warn = True
                adj.append(value // 2)
            if warn:
                print(
                    f"Warning: uneven counts detected for {team_a} vs {team_b}; "
                    "rounding down after duplicate log merge."
                )
            adjusted_pair_results[(team_a, team_b)] = tuple(adj)

        # Update ELO ratings using de-duplicated counts
        if self.use_ratings:
            for (team_a, team_b), results in adjusted_pair_results.items():
                a_wins, a_losses, b_wins, b_losses = results
                idx_a = self.team_names.index(team_a)
                idx_b = self.team_names.index(team_b)

                for _ in range(a_wins):
                    self._update_elo(idx_a, idx_b, 1.0)
                for _ in range(b_wins):
                    self._update_elo(idx_a, idx_b, 0.0)

            self._save_ratings()

        # Update tournament matrices
        for (team_a, team_b), results in adjusted_pair_results.items():
            a_wins, a_losses, b_wins, b_losses = results
            total_games = a_wins + a_losses + b_wins + b_losses

            if total_games == 0:
                continue

            # Get team indices
            idx_a = self.team_names.index(team_a)
            idx_b = self.team_names.index(team_b)

            # Calculate actual number of matches between these teams
            # Note: a_wins + a_losses should equal b_wins + b_losses (total matches from each perspective)
            matches_played = a_wins + a_losses  # or equivalently: b_wins + b_losses

            # Update battle matrices (store only once to avoid double counting)
            # We'll store battles in the lower triangle of the matrix
            if idx_a < idx_b:
                self.battles[idx_a][idx_b] += matches_played
                self.wins[idx_a][idx_b] += a_wins
                self.wins[idx_b][idx_a] += b_wins
            else:
                self.battles[idx_b][idx_a] += matches_played
                self.wins[idx_b][idx_a] += a_wins
                self.wins[idx_a][idx_b] += b_wins

            print(
                f"  {team_a} vs {team_b}: {matches_played} matches ({a_wins}-{a_losses} from {team_a} perspective)"
            )

            # Write to results CSV
            with open(self.results_csv, "a") as f:
                f.write(f"{team_a},{team_b},{a_wins},{a_losses},{b_wins},{b_losses}\n")

        # Clean up log files
        for log_file in log_files:
            try:
                log_file.unlink()
            except Exception:
                pass

        print(
            f"Updated tournament state with {len(adjusted_pair_results)} team pairings"
        )

    def _pairing_allowed(self, team_a_name: str, team_b_name: str) -> bool:
        """Return True if this pairing should be scheduled under the focus filters."""
        if not self.focus_team and not self.opponent_team:
            return True

        if self.focus_team and self.opponent_team:
            return {team_a_name, team_b_name} == {self.focus_team, self.opponent_team}

        if self.focus_team:
            return self.focus_team in (team_a_name, team_b_name)

        # Should not reach here, but allow pairing if filters are inconsistent
        return True

    def _load_ratings(self) -> List[float]:
        """Load ELO ratings from file, or initialize with default values."""
        default_elo = 1500.0

        if self.ratings_file.exists():
            try:
                with open(self.ratings_file, "r") as f:
                    ratings_data = json.load(f)

                # Load existing ratings, initialize new teams to default
                ratings = []
                for team_name in self.team_names:
                    ratings.append(ratings_data.get(team_name, default_elo))

                print(f"Loaded ratings from {self.ratings_file}")
                return ratings
            except Exception as e:
                print(f"Error loading ratings: {e}, using defaults")

        # Initialize all teams with default ELO
        print(f"Initializing all teams with ELO {default_elo}")
        return [default_elo] * len(self.team_names)

    def _save_ratings(self):
        """Save current ELO ratings to file."""
        if not self.use_ratings:
            return

        ratings_data = {}
        for i, team_name in enumerate(self.team_names):
            ratings_data[team_name] = self.elo_ratings[i]

        try:
            # Save to primary location
            with open(self.ratings_file, "w") as f:
                json.dump(ratings_data, f, indent=2)
            print(f"Saved ratings to {self.ratings_file}")

            # Save timestamped backup to permanent location
            self._save_ratings_backup(ratings_data)

        except Exception as e:
            print(f"Error saving ratings: {e}")

    def _save_ratings_backup(self, ratings_data):
        """Save timestamped backup of ratings to permanent location."""
        try:
            # Create permanent backup directory
            ratings_backup_dir = self.backup_dir / "ratings"
            ratings_backup_dir.mkdir(parents=True, exist_ok=True)

            # Generate timestamp
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Create backup filename
            backup_file = ratings_backup_dir / f"ratings_{timestamp}.json"

            # Add metadata to backup
            backup_data = {
                "timestamp": datetime.now().isoformat(),
                "battle_format": self.battle_format,
                "num_teams": len(self.team_names),
                "team_names": self.team_names,
                "ratings": ratings_data,
            }

            with open(backup_file, "w") as f:
                json.dump(backup_data, f, indent=2)

            print(f"Saved ratings backup to {backup_file}")

        except Exception as e:
            print(f"Warning: Could not save ratings backup: {e}")

    def _save_results_backup(self):
        """Save timestamped backup of results.csv to permanent location."""
        try:
            # Create permanent backup directory
            results_backup_dir = self.backup_dir / "results"
            results_backup_dir.mkdir(parents=True, exist_ok=True)

            # Generate timestamp
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Create backup filename
            backup_file = (
                results_backup_dir / f"results_{self.battle_format}_{timestamp}.csv"
            )

            # Copy the results CSV if it exists
            if self.results_csv.exists():
                import shutil

                shutil.copy2(self.results_csv, backup_file)
                print(f"Saved results backup to {backup_file}")

        except Exception as e:
            print(f"Warning: Could not save results backup: {e}")

    def _update_elo(self, team1_idx: int, team2_idx: int, result: float):
        """Update ELO ratings for a single battle result.

        Args:
            team1_idx: Index of first team
            team2_idx: Index of second team
            result: 1.0 if team1 wins, 0.0 if team2 wins, 0.5 for draw
        """
        k_factor = 32  # Standard ELO K-factor

        pre_elo1 = self.elo_ratings[team1_idx]
        pre_elo2 = self.elo_ratings[team2_idx]

        # Calculate expected scores
        expected1 = 1 / (1 + 10 ** ((pre_elo2 - pre_elo1) / 400))
        expected2 = 1 - expected1

        # Update ratings
        elo_change1 = k_factor * (result - expected1)
        elo_change2 = k_factor * ((1 - result) - expected2)

        self.elo_ratings[team1_idx] += elo_change1
        self.elo_ratings[team2_idx] += elo_change2

    def _load_existing_battle_data(self):
        """Load existing battle data from results.csv to populate battle matrices."""
        if not self.results_csv.exists():
            return

        print(f"Loading existing battle data from {self.results_csv}")

        try:
            import csv

            battles_loaded = 0

            with open(self.results_csv, "r") as f:
                reader = csv.reader(f)
                header = next(reader, None)  # Skip header
                if not header:
                    return

                for row in reader:
                    if len(row) < 6:
                        continue

                    team_a_name = row[0].strip()
                    team_b_name = row[1].strip()
                    a_wins = int(row[2])
                    a_losses = int(row[3])
                    b_wins = int(row[4])
                    # b_losses = int(row[5])  # Not needed for calculation

                    # Check if both teams are in current tournament
                    if (
                        team_a_name not in self.team_names
                        or team_b_name not in self.team_names
                    ):
                        continue

                    idx_a = self.team_names.index(team_a_name)
                    idx_b = self.team_names.index(team_b_name)

                    # Calculate total matches from this CSV row
                    matches_played = a_wins + a_losses  # Should equal b_wins + b_losses

                    # Update battle matrices (store only in lower triangle)
                    if idx_a < idx_b:
                        self.battles[idx_a][idx_b] += matches_played
                        self.wins[idx_a][idx_b] += a_wins
                        self.wins[idx_b][idx_a] += b_wins
                    else:
                        self.battles[idx_b][idx_a] += matches_played
                        self.wins[idx_b][idx_a] += a_wins
                        self.wins[idx_a][idx_b] += b_wins

                    battles_loaded += matches_played

            if battles_loaded > 0:
                print(f"Loaded {battles_loaded} historical battles from CSV")
            else:
                print("No historical battle data found for current teams")

        except Exception as e:
            print(f"Error loading existing battle data: {e}")

    def get_standings(self):
        """Get current tournament standings."""
        standings = []
        for i, team_name in enumerate(self.team_names):
            # Count battles and wins correctly with new storage pattern
            total_battles = 0
            total_wins = 0
            for j in range(len(self.team_names)):
                if i != j:
                    # Check both positions since we store only in lower triangle for battles
                    if i < j:
                        total_battles += self.battles[i][j]
                        total_wins += self.wins[i][j]
                    else:
                        total_battles += self.battles[j][i]
                        total_wins += self.wins[i][j]

            win_rate = total_wins / total_battles if total_battles > 0 else 0.0

            team_data = {
                "team": team_name,
                "wins": total_wins,
                "losses": total_battles - total_wins,
                "total_battles": total_battles,
                "win_rate": win_rate,
            }

            # Add ELO rating if enabled
            if self.use_ratings:
                team_data["elo"] = self.elo_ratings[i]

            standings.append(team_data)

        # Sort by ELO if enabled, otherwise by win rate
        if self.use_ratings:
            standings.sort(key=lambda x: x["elo"], reverse=True)
        else:
            standings.sort(key=lambda x: (x["win_rate"], x["wins"]), reverse=True)
        return standings

    def generate_heatmap(self, save_path: Optional[Path] = None):
        """Generate win rate heatmap."""
        n = len(self.team_files)
        win_rates = np.full((n, n), 0.5)

        for i in range(n):
            for j in range(n):
                if i != j:
                    # Get battle count from correct location (lower triangle storage)
                    battles_count = self.battles[min(i, j)][max(i, j)]
                    if battles_count > 0:
                        win_rates[i, j] = self.wins[i][j] / battles_count

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            win_rates,
            xticklabels=self.team_names,
            yticklabels=self.team_names,
            vmin=0,
            vmax=1,
            cmap="RdYlGn",
            center=0.5,
            annot=True,
            fmt=".3f",
        )
        plt.title("Team vs Team Win Rate Heatmap")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"Heatmap saved to {save_path}")
        else:
            plt.show()
        plt.close()

    def print_summary(self):
        """Print tournament summary."""
        standings = self.get_standings()

        print("\n" + "=" * 70)
        print("TOURNAMENT STANDINGS")
        print("=" * 70)

        if self.use_ratings:
            print(
                f"{'Rank':<4} {'Team':<20} {'ELO':<8} {'W-L':<12} {'Win%':<8} {'Games':<6}"
            )
            print("-" * 70)
            for rank, team in enumerate(standings, 1):
                print(
                    f"{rank:<4} {team['team']:<20} {team['elo']:<8.1f} "
                    f"{team['wins']}-{team['losses']:<12} {team['win_rate']:<8.3f} {team['total_battles']:<6}"
                )
        else:
            print(f"{'Rank':<4} {'Team':<20} {'W-L':<12} {'Win%':<8} {'Games':<6}")
            print("-" * 55)
            for rank, team in enumerate(standings, 1):
                print(
                    f"{rank:<4} {team['team']:<20} "
                    f"{team['wins']}-{team['losses']:<12} {team['win_rate']:<8.3f} {team['total_battles']:<6}"
                )


def main():
    # python run_tournament.py --teams-dir filtered_teams/curr_gauntlet_winners --battles-per-pairing 3 --results-dir results/tournament_gauntlet
    parser = argparse.ArgumentParser(description="Simple Pokemon Showdown Tournament")
    parser.add_argument("--format", default="gen9ou", help="Battle format")
    parser.add_argument(
        "--teams-dir", default=str(REPO_ROOT / "metamon_cache/teams/competitive/gen9ou")
    )
    parser.add_argument(
        "--results-dir", default=str(REPO_ROOT / "results/simple_showdown")
    )
    parser.add_argument(
        "--top-n", type=int, default=1000, help="Number of teams to use"
    )
    parser.add_argument(
        "--battles-per-pairing",
        type=int,
        default=5,
        help="Number of battles per pairing",
    )
    parser.add_argument(
        "--no-uuid",
        action="store_true",
        help="Use consistent usernames without UUID suffix",
    )
    parser.add_argument(
        "--use-ratings",
        action="store_true",
        help="Enable ELO rating system with persistent storage",
    )
    parser.add_argument(
        "--ratings-file",
        type=str,
        help="Path to ratings file (default: results_dir/ratings.json)",
    )
    parser.add_argument(
        "--backup-dir",
        type=str,
        help="Directory for permanent backups (default: repo_root/backups)",
    )
    parser.add_argument(
        "--focus-team",
        type=str,
        help=(
            "Team (path or stem) that must be included; pairings will be limited "
            "to matchups involving this team"
        ),
    )
    parser.add_argument(
        "--opponent-team",
        type=str,
        help=(
            "Optional opponent (path or stem). When set, only the focus vs opponent "
            "matchup will be played"
        ),
    )

    args = parser.parse_args()

    if args.opponent_team and not args.focus_team:
        parser.error("--opponent-team requires --focus-team")

    # Load teams
    teams_dir = Path(args.teams_dir)
    team_files = sorted([p for p in teams_dir.glob(f"*.{args.format}_team")])[
        : args.top_n
    ]
    team_files = [p.resolve() for p in team_files]
    team_files = dedupe_paths(team_files)

    focus_team_path, focus_team_name = resolve_team_argument(args.focus_team)
    opponent_team_path, opponent_team_name = resolve_team_argument(args.opponent_team)

    for extra_path in [focus_team_path, opponent_team_path]:
        if extra_path and extra_path not in team_files:
            team_files.append(extra_path)

    team_files = dedupe_paths(team_files)

    if len(team_files) < 2:
        raise RuntimeError("Need at least 2 teams")

    team_names = [team.stem for team in team_files]

    if focus_team_name and focus_team_name not in team_names:
        parser.error(
            f"Could not find focus team '{args.focus_team}' in the provided teams"
        )

    if opponent_team_name and opponent_team_name not in team_names:
        parser.error(
            f"Could not find opponent team '{args.opponent_team}' in the provided teams"
        )

    if focus_team_name and opponent_team_name and focus_team_name == opponent_team_name:
        parser.error("--focus-team and --opponent-team must refer to different teams")

    print(f"Selected {len(team_files)} teams:")
    for team in team_files:
        print(f"  - {team.stem}")

    # Run tournament
    results_dir = Path(args.results_dir)
    use_uuid = not args.no_uuid  # Invert the flag

    # Set up ratings file path if specified
    ratings_file = Path(args.ratings_file) if args.ratings_file else None
    backup_dir = Path(args.backup_dir) if args.backup_dir else None

    tournament = SimplePokemonTournament(
        team_files,
        args.format,
        results_dir,
        use_uuid=use_uuid,
        use_ratings=args.use_ratings,
        ratings_file=ratings_file,
        backup_dir=backup_dir,
        focus_team=focus_team_name,
        opponent_team=opponent_team_name,
    )

    if args.no_uuid:
        print("Using consistent usernames (no UUID suffix)")
    else:
        print("Using unique usernames with UUID suffix")

    if args.use_ratings:
        print("ELO rating system enabled")
    else:
        print("Using simple win rate rankings")

    # Run tournament
    tournament.run_tournament(battles_per_pairing=args.battles_per_pairing)

    # Final results
    print(f"\n{'=' * 50}")
    print("FINAL RESULTS")
    print(f"{'=' * 50}")
    tournament.print_summary()

    # Generate visualizations
    heatmap_path = results_dir / "win_rate_heatmap.png"
    tournament.generate_heatmap(save_path=heatmap_path)

    # Save summary
    summary_path = results_dir / "tournament_summary.json"
    summary_data = {
        "standings": tournament.get_standings(),
        "pairings_completed": tournament.pairings_completed,
        "total_battles": sum(
            sum(row) for row in tournament.battles
        ),  # Now stored only once in lower triangle
    }

    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\nSummary saved to: {summary_path}")
    print(f"Results directory: {results_dir}")


if __name__ == "__main__":
    main()
