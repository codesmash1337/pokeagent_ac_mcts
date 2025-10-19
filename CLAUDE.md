# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Pokémon AI agent project that combines Actor-Critic methods with Monte Carlo Tree Search (MCTS) for competitive Pokémon battling. The project integrates multiple vendored dependencies as git subtrees:

- **AMAGO** (`vendor/amago`): Scalable in-context RL framework for training adaptive agents with transformers
- **Metamon** (`vendor/metamon`): Pokémon Showdown RL environment, datasets, and baselines
- **Poke Engine** (`vendor/poke-engine`): Rust-based Pokémon battle simulator with Python bindings
- **Foul Play** (`vendor/foul-play`): Pokémon battle bot framework

## Architecture

### Key Components

**Battle Simulation Stack:**
- Poke Engine (Rust) provides the core battle simulator for generations 4-9
- Supports expectiminimax, iterative deepening, and MCTS search algorithms
- Python bindings via poke-engine-py allow integration with RL frameworks

**RL Environment:**
- Metamon provides gymnasium-compatible environments for Pokémon battles
- Supports multiple battle formats (gen1ou through gen9ou)
- Observation spaces are tokenized for transformer models
- Action space: 13 discrete actions (4 moves + 5 switches + 4 terastallization moves)

**Training Framework:**
- AMAGO handles large-scale RL training with transformers
- Supports offline RL on human battle datasets
- Multi-GPU training with asynchronous rollouts
- Off-policy learning with large replay buffers stored on disk

**Battle Backend Options:**
- `"poke-env"`: Standard poke-env backend (faster, more stable)
- `"metamon"`: Experimental replay-accurate backend (reduces sim2sim gap)

## Development Commands

### Environment Setup

```bash
# Install metamon (the main environment package)
pip install -e vendor/metamon

# Install poke-engine for a specific generation (requires Rust)
cd vendor/poke-engine
make gen9  # or gen4, gen5, gen6, gen7, gen8
./target/release/poke-engine  # Run the engine directly

# Or install via foul-play Makefile
cd vendor/foul-play
make poke_engine GEN=gen9

# Install AMAGO for RL training
pip install -e vendor/amago
# Optional: pip install -e vendor/amago[flash] for FlashAttention
# Optional: pip install -e vendor/amago[mamba] for Mamba sequence models
```

### Running Tests

```bash
# Test the Metamon environment
python -m metamon.env --battle_format gen1ou --episodes 10 --team_set competitive

# Run foul-play tests
cd vendor/foul-play
make test  # Runs ruff check and pytest

# Test poke-engine
cd vendor/poke-engine
cargo test
```

### Local Pokémon Showdown Server

Metamon requires a local Showdown server for battles:

```bash
cd vendor/metamon/server/pokemon-showdown
npm install
node pokemon-showdown start --no-security
# --no-security removes battle speed throttling and password requirements
```

### Training Models

```bash
# Train an imitation learning model from scratch
cd vendor/metamon
python -m metamon.rl.train \
  --run_name MyAgent \
  --model_gin_config small_agent.gin \
  --train_gin_config il.gin \
  --save_dir ~/checkpoints/ \
  --log

# Train an RL model
python -m metamon.rl.train \
  --run_name MyRLAgent \
  --model_gin_config small_agent.gin \
  --train_gin_config exp_rl.gin \
  --save_dir ~/checkpoints/ \
  --log

# Finetune from a pretrained HuggingFace model
python -m metamon.rl.finetune_from_hf \
  --finetune_from_model SmallRL \
  --run_name MyFinetuned \
  --save_dir ~/checkpoints/ \
  --formats gen9ou \
  --epochs 10 \
  --log
```

### Evaluation

```bash
# Evaluate against heuristic baselines
python -m metamon.rl.evaluate \
  --eval_type heuristic \
  --agent SyntheticRLV2 \
  --gens 1 \
  --formats ou \
  --total_battles 100

# Evaluate on local ladder
python -m metamon.rl.evaluate \
  --eval_type ladder \
  --agent SyntheticRLV2 \
  --gens 1 \
  --formats ou \
  --total_battles 50 \
  --username MyAgent \
  --team_set competitive

# Compete two baselines
cd vendor/metamon
python -m metamon.baselines.compete \
  --battle_format gen2ou \
  --player GymLeader \
  --opponent RandomBaseline \
  --battles 10
```

### Interactive Poke Engine Usage

```bash
cd vendor/poke-engine
./target/release/poke-engine --state <state-string>

# Available commands:
# - expectiminimax <depth> [ab-prune]: Run expectiminimax search
# - monte-carlo-tree-search <time-ms>: Run MCTS
# - iterative-deepening <time-ms>: Run iterative deepening
# - calculate-damage <s1-move> <s2-move>: Get damage rolls
# - generate-instructions <s1-move> <s2-move>: See instruction tree
```

### Linting and Formatting

```bash
# Foul-play
cd vendor/foul-play
make fmt   # Format with ruff
make lint  # Lint and auto-fix with ruff
```

## Important Technical Details

### Poke Engine Generation Features

The poke-engine must be built with generation-specific features. When installing poke-engine-py:

```bash
pip install -v --force-reinstall --no-cache-dir poke-engine \
  --config-settings="build-args=--features poke-engine/gen9 --no-default-features"
```

Available generations: gen4, gen5, gen6, gen7, gen8, gen9

### Observation Spaces

- `DefaultObservationSpace`: Original text/numerical space from the paper
- `ExpandedObservationSpace`: Improved version with tera types for Gen 9
- `TeamPreviewObservationSpace`: Adds opponent team preview
- `OpponentMoveObservationSpace`: Includes revealed opponent moves

All observation spaces should be wrapped with `TokenizedObservationSpace` for use with transformer models.

### Action Space

Universal action space of 13 discrete actions:
- 0-3: Use moves (alphabetical order)
- 4-8: Switch Pokémon (alphabetical order)
- 9-12: Use moves with terastallization (Gen 9 only)

### Reward Functions

- `DefaultShapedReward`: +/-100 for win/loss plus light shaping for damage/healing
- `BinaryReward`: Only +/-100 for win/loss
- `AggressiveShapedReward`: +200 for win, 0 for loss

### Dataset Management

```bash
# Set cache directory for large datasets
export METAMON_CACHE_DIR=/path/to/storage

# Download datasets in advance
python -m metamon.data.download parsed-replays
python -m metamon.data.download teams
python -m metamon.data.download usage-stats
python -m metamon.data.download revealed-teams
```

### Team Sets

Available team sets via `get_metamon_teams(battle_format, set_name)`:
- `"competitive"`: Human-made expert teams (< 30 per format)
- `"modern_replays"`: 8k-20k predicted teams from recent replays (OU only)
- `"paper_variety"`: 1k procedurally generated teams (Gen 1-4)
- `"paper_replays"`: 1k replay-based teams (Gen 1-4 OU)

### Baseline Opponents

Access via `get_baseline(name)`:
- `BugCatcher`: Actively bad agent (picks worst moves)
- `RandomBaseline`: Random legal actions
- `Gen1BossAI`: Emulates Gen 1 game AI
- `Grunt`: Maximally offensive (picks highest damage)
- `GymLeader`: Improved Grunt with stat boosts/healing
- `EmeraldKaizo`: ROM hack AI with extensive move rules
- `BaseRNN`: Simple RNN imitation learning baseline

### Sim2Sim Gap

There is a mismatch between battle trajectories observed live vs. replay data because replays are from a spectator POV while live battles have player-only information. Replay data is best treated as pretraining data for offline-to-online finetuning.

## Logging and Monitoring

```bash
# Configure wandb for training runs
export METAMON_WANDB_PROJECT="my_project"
export METAMON_WANDB_ENTITY="my_username"

# Use --log flag when training to enable wandb logging
```

## Git Subtree Management

This project uses git subtrees for vendored dependencies. When updating:

```bash
# Update a subtree
git subtree pull --prefix vendor/metamon <remote-url> main --squash

# Add a new subtree
git subtree add --prefix vendor/new-dep <remote-url> main --squash
```

## Instructions for Claude
- Prioritize performance optimizations
- Limit code changes to a minimum
- Rate limit all endpoints
- Only make changes with 95% certainty of achieving the desired output
- Update ac_boosted_mcts_prd.md when there are changes to the product requirements
- Disagree when proposed solution is not optimal
- Use the virtual environment venv