#!/usr/bin/env python3
"""Minimal Bradley–Terry estimator for offline battle logs.

Reads a CSV containing head-to-head results and computes Bradley–Terry scores
(and Wald confidence intervals) for each team. Intended for quick analysis of
files like `results/sample_teams/results.csv`.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple
import shutil

import numpy as np
import pandas as pd


@dataclass
class TeamStats:
    rating: float
    ci_lower: float
    ci_upper: float
    matches: int


def parse_results(csv_path: Path) -> Tuple[pd.DataFrame, Dict[str, int]]:
    df = pd.read_csv(csv_path)
    required_cols = {
        "team_a_name",
        "team_b_name",
        "team_a_wins",
        "team_a_losses",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {sorted(missing)}")
    teams = sorted(set(df["team_a_name"]) | set(df["team_b_name"]))
    index = {name: idx for idx, name in enumerate(teams)}
    return df, index


def fit_bradley_terry(
    df: pd.DataFrame, index: Dict[str, int], max_iter: int = 1000, tol: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(index)
    ability = np.ones(n, dtype=float)
    wins = np.zeros((n, n), dtype=float)
    matches = np.zeros((n, n), dtype=float)

    for _, row in df.iterrows():
        i = index[row["team_a_name"]]
        j = index[row["team_b_name"]]
        w_ij = float(row["team_a_wins"])
        l_ij = float(row["team_a_losses"])
        wins[i, j] += w_ij
        wins[j, i] += l_ij
        total = w_ij + l_ij
        matches[i, j] += total
        matches[j, i] += total

    for _ in range(max_iter):
        updated = np.copy(ability)
        for i in range(n):
            numer = wins[i].sum()
            denom = 0.0
            for j in range(n):
                if i == j:
                    continue
                m_ij = matches[i, j]
                if m_ij <= 0:
                    continue
                denom += m_ij / (ability[i] + ability[j])
            if numer == 0 and denom == 0:
                updated[i] = ability[i]
            else:
                updated[i] = numer / max(denom, 1e-12)
        if np.max(np.abs(updated - ability)) < tol:
            ability = updated
            break
        ability = updated

    log_ratings = np.log(np.maximum(ability, 1e-12))

    fisher = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            m_ij = matches[i, j]
            if m_ij <= 0:
                continue
            diff = log_ratings[i] - log_ratings[j]
            exp_diff = math.exp(diff)
            weight = m_ij * exp_diff / (1 + exp_diff) ** 2
            fisher[i, i] += weight
            fisher[i, j] -= weight

    cov = np.linalg.pinv(fisher)
    var = np.diag(cov)
    se = np.sqrt(np.maximum(var, 0.0))
    return log_ratings, se, matches


def summarize(csv_path: Path) -> Dict[str, TeamStats]:
    df, index = parse_results(csv_path)
    ratings, se, matches_matrix = fit_bradley_terry(df, index)
    stats: Dict[str, TeamStats] = {}
    z = 1.96
    total_matches = np.zeros(len(index), dtype=int)
    for i in range(len(index)):
        total_matches[i] = int(matches_matrix[i].sum())
    for name, idx in index.items():
        r = ratings[idx]
        ci = z * se[idx]
        stats[name] = TeamStats(
            rating=r,
            ci_lower=r - ci,
            ci_upper=r + ci,
            matches=int(total_matches[idx]),
        )
    return stats


def export_near_top(
    output_dir: Path, teams_dir: Path, stats: Dict[str, TeamStats]
) -> None:
    if not stats:
        print("No stats available; skipping export.")
        return

    ranked = sorted(stats.items(), key=lambda kv: kv[1].rating, reverse=True)
    best_name, best_stats = ranked[0]
    threshold = best_stats.ci_lower

    output_dir.mkdir(parents=True, exist_ok=True)
    eligible = [name for name, st in ranked if st.ci_upper >= threshold]

    print(
        f"Exporting {len(eligible)} teams with CI upper >= {threshold:.3f} to {output_dir}"
    )

    for name in eligible:
        src = teams_dir / f"{name}.gen9ou_team"
        if not src.exists():
            print(f"  Warning: team file not found for {name} in {teams_dir}")
            continue
        dest = output_dir / src.name
        shutil.copy2(src, dest)
        print(f"  Copied {src.name}")


def main() -> None:
    # python offline_bt.py results/simple_showdown/results.csv
    parser = argparse.ArgumentParser(description="Offline Bradley–Terry ranking")
    parser.add_argument("csv", type=Path, help="CSV with head-to-head results")
    parser.add_argument(
        "--top",
        type=int,
        default=0,
        help="Show only top N teams (default: show all)",
    )
    parser.add_argument(
        "--export-near-top",
        type=Path,
        help=(
            "Directory to copy teams whose upper CI bounds overlap the top team's "
            "lower bound (for further experiments)"
        ),
    )
    parser.add_argument(
        "--teams-dir",
        type=Path,
        default=Path("filtered_teams/all_teams"),
        help="Directory containing .gen9ou_team files for export",
    )
    args = parser.parse_args()

    stats = summarize(args.csv)
    ranked = sorted(stats.items(), key=lambda kv: kv[1].rating, reverse=True)
    if args.top > 0:
        ranked = ranked[: args.top]

    print(f"Bradley–Terry ranking for {args.csv}:")
    for rank, (team, st) in enumerate(ranked, start=1):
        print(
            f"{rank:2d}. {team:<60} rating={st.rating: .3f} "
            f"CI=({st.ci_lower:.3f}, {st.ci_upper:.3f}) matches={st.matches}"
        )

    if args.export_near_top:
        export_near_top(args.export_near_top, args.teams_dir, stats)


if __name__ == "__main__":
    # Usage: python offline_bt.py results/simple_showdown/results.csv
    main()
