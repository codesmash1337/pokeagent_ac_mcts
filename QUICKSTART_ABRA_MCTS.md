# Quick Start: Abra-Boosted MCTS

**Updated:** October 19, 2025
**Model:** Abra (medium transformer, Gen1-9 support)

---

## Prerequisites

```bash
# Activate virtual environment
source venv/bin/activate

# Verify installations
python -c "import metamon; print('Metamon OK')"
python -c "import torch; print('PyTorch OK')"
python -c "from poke_engine import mcts; print('Poke-engine OK')"
```

---

## Quick Test

### 1. Test Abra Integration

```bash
python test_abra_integration.py
```

**Expected output:**
```
Abra Loading              ✓ PASS
Abra Policy Provider      ✓ PASS
Abra Inference            ✓ PASS
Abra NeuralSearch         ✓ PASS
```

### 2. Run a Test Battle

```bash
# Start local Pokemon Showdown server (in separate terminal)
cd vendor/metamon/server/pokemon-showdown
node pokemon-showdown start --no-security

# Run neural-guided MCTS bot (in main terminal)
python -m fp.run \
    --websocket-uri ws://localhost:8000/showdown/websocket \
    --ps-username AbraMCTSBot \
    --bot-mode search_ladder \
    --pokemon-format gen9randombattle \
    --use-neural-mcts \
    --neural-c-puct 1.0
```

---

## Configuration Options

### Model Selection

To switch between models, edit `vendor/foul-play/fp/search/neural_guided.py:50`:

```python
# Abra (default - medium transformer, Gen1-9)
model_name="Abra"

# Minikazam (small RNN, faster but less capable)
model_name="Minikazam"

# Other options (untested but should work):
# model_name="SyntheticRLV2"  # Largest model, best performance
```

### MCTS Parameters

```bash
# Adjust search time (ms)
--mcts-search-time 500  # Default, good balance

# Adjust neural weighting
--neural-c-puct 1.0  # Default, equal MCTS/neural weight
--neural-c-puct 2.0  # More neural influence
--neural-c-puct 0.5  # More MCTS influence

# Disable neural guidance (vanilla MCTS)
# Just remove the --use-neural-mcts flag
```

---

## Understanding the System

### How It Works

```
1. MCTS runs for 500ms on poke-engine
   ├─ Explores ~200-1000 game tree nodes
   └─ Builds visit count distribution

2. For each sampled battle state:
   ├─ StateTranslator converts to Metamon observation
   ├─ Abra model infers policy distribution
   └─ Combines with MCTS visits (geometric mean)

3. Select final move from boosted distribution
```

### What Gets Boosted

The Abra model influences move selection by:
- **Reinforcing** moves that both MCTS and Abra prefer
- **Penalizing** moves that only MCTS explores (but Abra disagrees)
- **Surfacing** strong moves that MCTS may have under-explored

---

## Performance Expectations

### Inference Time

| Component | Time | Notes |
|-----------|------|-------|
| MCTS Search | ~500ms | Configurable |
| State Translation | <5ms | Very fast |
| Abra Inference | ~40-70ms | Medium transformer |
| **Total** | **~550-575ms** | Well under 1000ms budget |

### Win Rate Goals

- **Baseline MCTS**: ~50% vs random
- **Abra (pure RL)**: ~50% GXE in Gen9OU
- **Target (Abra-MCTS)**: >55% vs Abra baseline

---

## Troubleshooting

### "Model not loaded" Error

```python
# Check if Abra downloads correctly
from metamon.rl.pretrained import get_pretrained_model
abra = get_pretrained_model("Abra")
agent = abra.initialize_agent(checkpoint=None, log=False)
```

If this fails:
- Check `METAMON_CACHE_DIR` environment variable
- Verify HuggingFace Hub access
- Check disk space for model download (~500MB)

### "FlashAttention not available" Warning

This is expected and safe to ignore. The system automatically falls back to VanillaAttention.

### Slow Inference

If Abra inference is too slow:
- Switch to Minikazam (faster RNN)
- Reduce MCTS search time
- Enable GPU if available: `device="cuda"`

---

## Advanced Usage

### GPU Acceleration

Edit `vendor/foul-play/fp/search/neural_guided.py:52`:

```python
_policy_provider = LocalPolicyProvider(
    model_name="Abra",
    device="cuda"  # Changed from "cpu"
)
```

### Custom Observation Space

Edit `vendor/neural-mcts/src/neural_mcts/state_translator.py:21`:

```python
# Current (default for Abra)
observation_space_type="TeamPreviewObservationSpace"

# Alternative options:
# observation_space_type="OpponentMoveObservationSpace"  # For Minikazam
# observation_space_type="ExpandedObservationSpace"
```

### Ensemble Multiple Models

Not currently supported, but could be added:

```python
# Future enhancement
models = ["Abra", "Minikazam", "SyntheticRLV2"]
policies = [get_policy(model, obs) for model in models]
ensemble_policy = np.mean(policies, axis=0)
```

---

## Evaluation Protocol

### Quick Evaluation (10 battles)

```bash
python -m fp.run \
    --websocket-uri ws://localhost:8000/showdown/websocket \
    --ps-username TestBot \
    --bot-mode search_random \
    --pokemon-format gen9randombattle \
    --use-neural-mcts \
    --battles 10
```

### Full Evaluation (500 battles)

See [PRD Section 7.3](PRD_AC_MCTS_HYBRID.md#73-evaluation-protocol) for complete evaluation protocol.

---

## Next Steps

1. **Verify Installation**: Run `python test_abra_integration.py`
2. **Test Battle**: Run a single battle with `--use-neural-mcts`
3. **Baseline Comparison**: Run 100 battles with and without neural guidance
4. **Full Evaluation**: 500-battle head-to-head vs Abra baseline

---

## Getting Help

- **Documentation**: See [PRD_AC_MCTS_HYBRID.md](PRD_AC_MCTS_HYBRID.md)
- **Implementation Details**: See [ABRA_UPGRADE_SUMMARY.md](ABRA_UPGRADE_SUMMARY.md)
- **Original System**: See [NEURAL_INTEGRATION_COMPLETE.md](NEURAL_INTEGRATION_COMPLETE.md)

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `vendor/foul-play/fp/search/neural_guided.py` | Main integration, model selection |
| `vendor/neural-mcts/src/neural_mcts/local_policy.py` | Abra model loading |
| `vendor/neural-mcts/src/neural_mcts/neural_search.py` | MCTS+neural combination |
| `vendor/neural-mcts/src/neural_mcts/state_translator.py` | State conversion |
| `test_abra_integration.py` | Integration tests |
| `PRD_AC_MCTS_HYBRID.md` | Full requirements document |
