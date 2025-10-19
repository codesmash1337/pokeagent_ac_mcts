# PUCT Implementation Plan

## Overview

Implementing PUCT (Polynomial Upper Confidence Trees) integration as outlined in the PRD requires modifying the poke-engine Rust codebase to accept neural policy priors during MCTS node selection.

## Architecture

### Current Flow
```
Python (foul-play) → Rust (poke-engine MCTS) → Return results → Python
```

### PUCT Flow
```
Python (foul-play)
   ↓
Initialize Python policy provider
   ↓
Rust (poke-engine MCTS)
   ├→ Node expansion
   ├→ Query Python for neural prior via PyO3 callback
   ├→ Store prior in MoveNode
   ├→ Use PUCT formula: Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
   └→ Return results
   ↓
Python (foul-play)
```

## Implementation Steps

### 1. Modify MoveNode Structure

**File**: `vendor/poke-engine/src/mcts.rs`

Add neural prior field to MoveNode:

```rust
#[derive(Debug)]
pub struct MoveNode {
    pub move_choice: MoveChoice,
    pub total_score: f32,
    pub visits: u32,
    pub neural_prior: f32,  // NEW: Neural policy prior
}
```

### 2. Add PUCT Method to MoveNode

```rust
impl MoveNode {
    pub fn ucb1(&self, parent_visits: u32) -> f32 {
        // Existing UCB1 formula
    }

    pub fn puct(&self, parent_visits: u32, c_puct: f32) -> f32 {
        // NEW: PUCT formula
        if self.visits == 0 {
            return f32::INFINITY;
        }
        let q_value = self.total_score / self.visits as f32;
        let u_value = c_puct * self.neural_prior * (parent_visits as f32).sqrt() / (1.0 + self.visits as f32);
        q_value + u_value
    }
}
```

### 3. Create Python Callback Interface in PyO3

**File**: `vendor/poke-engine/poke-engine-py/src/lib.rs`

Add callback struct:

```rust
use pyo3::types::PyFunction;

pub struct PolicyCallback {
    pub callback_fn: Py<PyFunction>,
}

impl PolicyCallback {
    pub fn query_policy(&self, state_str: &str, legal_moves: Vec<String>) -> Vec<f32> {
        Python::with_gil(|py| {
            let result = self.callback_fn.call1(py, (state_str, legal_moves)).unwrap();
            result.extract::<Vec<f32>>(py).unwrap()
        })
    }
}
```

### 4. Modify MCTS to Accept Callback

**File**: `vendor/poke-engine/src/mcts.rs`

```rust
pub fn perform_mcts_with_neural_priors(
    state: &mut State,
    s1_options: Vec<MoveChoice>,
    s2_options: Vec<MoveChoice>,
    duration: Duration,
    policy_callback: Option<PolicyCallback>,  // NEW
    c_puct: f32,  // NEW
) -> MctsResult {
    // ... existing code ...

    // When populating node, query neural priors
    if let Some(callback) = &policy_callback {
        let state_str = state.serialize();
        let legal_move_strs: Vec<String> = s1_options.iter().map(|m| m.to_string()).collect();
        let priors = callback.query_policy(&state_str, legal_move_strs);

        // Apply priors to MoveNodes
        for (node, prior) in s1_options_vec.iter_mut().zip(priors.iter()) {
            node.neural_prior = *prior;
        }
    } else {
        // Uniform prior
        let uniform_prior = 1.0 / s1_options.len() as f32;
        for node in s1_options_vec.iter_mut() {
            node.neural_prior = uniform_prior;
        }
    }
}
```

### 5. Update Node Selection to Use PUCT

**File**: `vendor/poke-engine/src/mcts.rs`

```rust
pub fn maximize_ucb_for_side(&self, side_map: &[MoveNode], c_puct: f32, use_puct: bool) -> usize {
    let mut choice = 0;
    let mut best_score = f32::MIN;
    for (index, node) in side_map.iter().enumerate() {
        let score = if use_puct {
            node.puct(self.times_visited, c_puct)
        } else {
            node.ucb1(self.times_visited)
        };
        if score > best_score {
            best_score = score;
            choice = index;
        }
    }
    choice
}
```

### 6. Expose New Function via PyO3

**File**: `vendor/poke-engine/poke-engine-py/src/lib.rs`

```rust
#[pyfunction]
fn mcts_with_neural_priors(
    py_state: PyState,
    duration_ms: u64,
    policy_callback: &PyFunction,
    c_puct: f32,
) -> PyResult<PyMctsResult> {
    let mut state: State = py_state.into();
    let duration = Duration::from_millis(duration_ms);
    let (s1_options, s2_options) = state.root_get_all_options();

    // Create callback wrapper
    let callback = PolicyCallback {
        callback_fn: policy_callback.into(),
    };

    let mcts_result = perform_mcts_with_neural_priors(
        &mut state,
        s1_options,
        s2_options,
        duration,
        Some(callback),
        c_puct,
    );

    let py_mcts_result = PyMctsResult::from_mcts_result(mcts_result, &state);
    Ok(py_mcts_result)
}

// Register in module
#[pymodule]
fn poke_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(mcts_with_neural_priors, m)?)?;
    // ... existing functions ...
    Ok(())
}
```

### 7. Python Integration

**File**: `vendor/foul-play/fp/search/puct_main.py`

```python
from poke_engine import State as PokeEngineState, mcts_with_neural_priors
from neural_mcts.puct_policy_provider import get_policy_for_rust, initialize_puct_policy_provider

# Initialize policy provider
initialize_puct_policy_provider(model_name="Minikazam")

# Run MCTS with neural priors
def get_result_from_puct_mcts(state_str: str, search_time_ms: int, c_puct: float) -> MctsResult:
    poke_engine_state = PokeEngineState.from_string(state_str)

    # Run MCTS with neural callback
    res = mcts_with_neural_priors(
        poke_engine_state,
        search_time_ms,
        get_policy_for_rust,  # Python callback function
        c_puct
    )

    return res
```

## Challenges

### 1. Performance
- **Problem**: Calling Python from Rust during MCTS is slow
- **Solution**:
  - Cache policy results for identical states
  - Batch queries where possible
  - Use async/await if supported

### 2. State Serialization
- **Problem**: Need to serialize state efficiently for callback
- **Solution**:
  - Use existing `state.serialize()` method (already implemented)
  - Pass string representation to Python
  - Python parses or uses cached translations

### 3. Thread Safety
- **Problem**: PyO3 requires GIL (Global Interpreter Lock)
- **Solution**:
  - Use `Python::with_gil()` for each callback
  - Accept performance penalty for correctness
  - Consider thread-local policy caching

## Testing Strategy

1. **Unit Tests**: Test PUCT formula calculation
2. **Integration Tests**: Test Rust → Python callback
3. **Performance Tests**: Measure overhead of neural queries
4. **Battle Tests**: Compare PUCT vs vanilla MCTS win rates

## Estimated Effort

- Rust modifications: 4-6 hours
- PyO3 integration: 2-3 hours
- Python callback implementation: 2-3 hours
- Testing and debugging: 4-6 hours
- **Total**: 12-18 hours

## Alternative: Simpler Approach

If full PUCT proves too complex, consider:

1. **Pre-compute priors**: Query all possible states before MCTS starts
2. **Lookup table**: Store priors in hashmap, MCTS looks up by state string
3. **Hybrid**: Use post-search reranking (already implemented)

This avoids callback complexity while still using neural guidance.

## Decision

Given the complexity and performance concerns of full PUCT integration with PyO3 callbacks during tree search, I recommend:

**Option 1 (Recommended)**: Continue with **post-search reranking** (already implemented)
- Simpler
- Faster
- Already working
- Can evaluate performance first

**Option 2 (Advanced)**: Implement **PUCT with pre-computed priors**
- Compute neural priors for root node only
- Use uniform priors for child nodes
- Balances complexity and benefit

**Option 3 (Full)**: Implement **PUCT with callbacks** (this document)
- Most aligned with PRD
- Highest complexity
- May not provide significant benefit over Option 1
- Performance overhead unknown

## Recommendation

Start with Option 1 (already done), evaluate results, then upgrade to Option 2 or 3 only if needed based on empirical performance data.
