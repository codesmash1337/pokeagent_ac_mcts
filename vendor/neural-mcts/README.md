# Neural-Guided MCTS for Pokemon Battling

A hybrid Pokemon battle AI that combines Monte Carlo Tree Search (MCTS) from poke-engine with neural policy priors from Metamon's Actor-Critic models.

## Overview

This project implements a Neural-Guided MCTS agent that improves upon baseline MCTS by using learned policy priors from transformer-based models to guide tree search. The architecture bridges Rust-based battle simulation (poke-engine) with Python-based neural networks (Metamon).

### Key Features

- **Neural Policy Priors**: Uses pretrained Metamon models (Minikazam, Abra) to provide learned move probabilities
- **State Translation**: Converts between poke-engine's Rust state representation and Metamon's tokenized observation space
- **Hybrid Search**: Combines MCTS visit counts with neural policy probabilities using PUCT-inspired formula
- **Graceful Fallback**: Falls back to vanilla MCTS if neural policy server is unavailable
- **HTTP-based IPC**: Lightweight communication between Rust simulator and Python neural network
- **Battle-tested**: Integrates with foul-play framework for Pokemon Showdown battles

## Architecture

```
┌─────────────────┐
│ Pokemon Showdown│
│     Server      │
└────────┬────────┘
         │
    ┌────▼────┐
    │Foul-Play│
    └────┬────┘
         │
    ┌────▼──────────┐
    │ Neural-MCTS   │
    │  Search Layer │
    └────┬────┬─────┘
         │    │
    ┌────▼────▼─────┐         ┌──────────────┐
    │  Poke-Engine  │◄────────┤    Neural    │
    │     MCTS      │  HTTP   │Policy Server │
    │   (Rust)      │────────►│  (Python)    │
    └───────────────┘         └──────┬───────┘
                                     │
                              ┌──────▼──────┐
                              │   Metamon   │
                              │Transformer  │
                              └─────────────┘
```

## Installation

### Prerequisites

1. **Poke-engine** (Rust-based Pokemon simulator)
```bash
cd ../poke-engine
make gen9
```

2. **Metamon** (RL environment and models)
```bash
pip install -e ../metamon
```

3. **Foul-play** (Battle framework)
```bash
cd ../foul-play
make poke_engine GEN=gen9
```

4. **Neural-MCTS** (this package)
```bash
pip install -e .
```

### Environment Setup

```bash
# Set cache directory for Metamon models and datasets
export METAMON_CACHE_DIR=/path/to/cache

# Ensure Python path includes all dependencies
export PYTHONPATH="src:../metamon:../foul-play:$PYTHONPATH"
```

## Quick Start

### 1. Start the Neural Policy Server

The policy server loads a pretrained Metamon model and serves policy predictions via HTTP:

```bash
# Using default settings (Minikazam model on CPU, port 5000)
./scripts/start_policy_server.sh

# Or with custom settings
MODEL_NAME=Abra DEVICE=cuda PORT=5000 ./scripts/start_policy_server.sh
```

The server exposes these endpoints:
- `GET /health` - Health check
- `POST /policy` - Single state policy query
- `POST /policy/batch` - Batch policy queries
- `POST /clear_cache` - Clear state translation cache

### 2. Run a Hybrid Agent

In a separate terminal, launch a hybrid MCTS agent:

```bash
# Basic usage
./scripts/run_hybrid_agent.sh

# Custom configuration
AGENT_NAME=HybridBot1 \
SEARCH_TIME_MS=1000 \
POKEMON_FORMAT=gen9ou \
./scripts/run_hybrid_agent.sh
```

### 3. Test Against Baseline

Run baseline Metamon RL agent for comparison:

```bash
cd ../../  # Return to project root
export METAMON_CACHE_DIR=.metamon_cache
venv/bin/python -m metamon.rl.evaluate \
  --eval_type ladder \
  --agent Minikazam \
  --gens 9 \
  --formats randombattle \
  --total_battles 20 \
  --username MetamonBaseline
```

Run baseline MCTS agent:

```bash
cd vendor/foul-play
PYTHONPATH=.:$PYTHONPATH python run.py \
  --websocket-uri ws://localhost:8000/showdown/websocket \
  --ps-username MCTSBaseline \
  --ps-password "" \
  --bot-mode search_ladder \
  --pokemon-format gen9randombattle \
  --search-time-ms 500 \
  --search-parallelism 2 \
  --run-count 20
```

## Configuration

### Policy Server Settings

Edit `config/neural_mcts_config.yaml`:

```yaml
policy_server:
  model_name: "Minikazam"  # or "Abra", "SmallRL", etc.
  device: "cpu"            # or "cuda"
  port: 5000
  cache_size: 1000
  host: "127.0.0.1"
```

### Neural Search Parameters

```yaml
neural_search:
  c_puct: 1.0              # Exploration constant
  neural_weight: 0.5       # Weight for neural prior vs MCTS prior
  fallback_on_error: true  # Use vanilla MCTS if policy server fails
  timeout_ms: 100          # HTTP request timeout
```

### MCTS Settings

```yaml
mcts:
  search_time_ms: 500      # Time budget per move
  parallelism: 2           # Number of parallel searches
  use_iterative_deepening: false
```

## API Reference

### StateTranslator

Converts poke-engine state strings to Metamon observations:

```python
from neural_mcts.state_translator import StateTranslator

translator = StateTranslator(observation_space_type="TeamPreviewObservationSpace")

# Single state
obs = translator.translate(state_str, legal_actions=[0, 1, 2, 3])

# Batch translation
obs_batch = translator.batch_translate(state_list, legal_actions_list)
```

### PolicyClient

Query the neural policy server:

```python
from neural_mcts.policy_client import PolicyClient

client = PolicyClient(server_url="http://localhost:5000")

# Get policy for a state
policy = client.get_policy(state_str, legal_actions=[0, 1, 2, 3])
# Returns np.ndarray of shape (13,) with probabilities

# Get statistics
stats = client.get_stats()
# Returns: {"queries": N, "cache_hits": M, "errors": K, "avg_latency_ms": X}
```

### NeuralPolicyServer

Serve neural policy predictions:

```python
from neural_mcts.policy_server import NeuralPolicyServer

server = NeuralPolicyServer(
    model_name="Minikazam",
    device="cpu",
    cache_size=1000,
    port=5000
)

server.start_server(host="127.0.0.1")
```

## Project Structure

```
vendor/neural-mcts/
├── README.md                      # This file
├── setup.py                       # Package configuration
├── config/
│   └── neural_mcts_config.yaml    # Configuration file
├── src/neural_mcts/
│   ├── __init__.py
│   ├── state_translator.py       # Rust state → Python obs
│   ├── policy_server.py           # HTTP server for neural policy
│   ├── policy_client.py           # Client for policy queries
│   └── neural_search.py           # Hybrid MCTS implementation
├── scripts/
│   ├── start_policy_server.sh     # Launch policy server
│   └── run_hybrid_agent.sh        # Launch hybrid agent
├── tests/
│   ├── test_state_translator.py
│   ├── test_policy_server.py
│   └── test_neural_search.py
└── examples/
    ├── basic_usage.py
    ├── compare_baselines.py
    └── tune_hyperparameters.py
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

```bash
# Format code
ruff format src/ tests/

# Lint
ruff check src/ tests/ --fix
```

### Performance Monitoring

Monitor policy server performance:

```bash
# Get server statistics
curl http://localhost:5000/stats

# Clear cache
curl -X POST http://localhost:5000/clear_cache
```

## Performance Targets

Based on PRD requirements:

- **State Translation**: < 5ms per state
- **Neural Policy Query**: < 50ms per batch (batch size 10)
- **Total Search Overhead**: < 10% compared to vanilla MCTS
- **Win Rate Improvement**: > 5% vs baseline Metamon RL model

## Troubleshooting

### Policy Server Won't Start

Check that Metamon models are downloaded:
```bash
python -m metamon.data.download parsed-replays
```

Verify PYTHONPATH includes Metamon:
```bash
echo $PYTHONPATH  # Should include ../metamon
```

### Agent Connection Fails

Ensure Pokemon Showdown server is running:
```bash
cd ../metamon/server/pokemon-showdown
node pokemon-showdown start --no-security
```

Check websocket URI matches server (default: `ws://localhost:8000/showdown/websocket`)

### Slow Performance

- Increase `cache_size` in policy server config
- Use GPU for neural inference: `DEVICE=cuda`
- Reduce `search_time_ms` or `parallelism`
- Use smaller model (Minikazam vs Abra)

### Neural Prior Not Being Used

Check policy server logs for errors:
```bash
tail -f policy_server.log
```

Verify `fallback_on_error: false` in config to catch errors

## Related Projects

- **Metamon**: https://github.com/spktrm/pokesim (RL environment and baselines)
- **Poke-Engine**: https://github.com/pmariglia/poke-engine (Rust battle simulator)
- **Foul-Play**: https://github.com/pmariglia/foul-play (Pokemon Showdown bot framework)
- **AMAGO**: https://github.com/UT-Austin-RPL/amago (In-context RL framework)

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{neural-mcts-pokemon,
  title={Neural-Guided Monte Carlo Tree Search for Pokemon Battling},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/pokeagent_ac_mcts}
}
```

## License

This project inherits licenses from its dependencies:
- Metamon: MIT License
- Poke-Engine: MIT License
- Foul-Play: MIT License
