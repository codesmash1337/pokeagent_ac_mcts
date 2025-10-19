# Product Requirements Document (PRD)
## Neural-Guided MCTS for Pokémon Battling

**Version:** 1.0
**Date:** October 18, 2025
**Status:** Draft
**Author:** Product Team

---

## 1. Executive Summary

### 1.1 Overview
This document outlines the requirements for developing a hybrid Pokémon battling agent that combines the strategic planning capabilities of Monte Carlo Tree Search (MCTS) with the learned policy priors from the Metamon Actor-Critic reinforcement learning model. The goal is to create a new model that outperforms the baseline Metamon Actor-Critic model while maintaining the existing baseline intact for comparison and evaluation.

### 1.2 Problem Statement
Current Pokémon AI agents face a trade-off:
- **Metamon Actor-Critic (RL) Model**: Fast inference, learned from human replays and self-play, but limited lookahead and strategic planning
- **Poke-Engine MCTS Model**: Sophisticated lookahead search and tactical evaluation, but relies on hand-crafted heuristics for move selection probabilities

The hybrid approach aims to leverage the strengths of both: using the learned policy network from Metamon as a neural prior to guide MCTS exploration, replacing the current heuristic-based policy.

### 1.3 Success Criteria
- **Primary**: New hybrid model achieves >5% higher win rate than baseline Metamon Actor-Critic model in head-to-head matches
- **Secondary**: Model demonstrates improved strategic play in complex battle scenarios (multi-turn sequences, switch decisions)
- **Performance**: Inference time remains within 2x of current MCTS model (with 500ms search budget)
- **Compatibility**: Successfully integrates with Pokemon Showdown server for live battles

---

## 2. Background & Context

### 2.1 Current System Architecture

#### 2.1.1 Metamon Actor-Critic Model (Python)
- **Framework**: AMAGO (transformer-based in-context RL)
- **State Representation**: Tokenized observations from poke-env
  - Text-based features (Pokémon names, moves, abilities)
  - Numerical features (HP, stats, type effectiveness)
  - Contextual history via transformer attention
- **Action Space**: 13 discrete actions (4 moves, 5 switches, 4 tera-moves)
- **Output**: Action probability distribution over legal moves
- **Training**: Offline IL on human replays + online RL with self-play
- **Inference**: ~50-100ms per decision

#### 2.1.2 Poke-Engine MCTS Model (Rust)
- **Framework**: Custom MCTS implementation in poke-engine
- **State Representation**: Structured state object
  - `Side` structs with Pokémon arrays (HP, stats, moves, abilities)
  - Global state (weather, terrain, trick room)
  - Deterministic game rules engine
- **Search Algorithm**: UCB1-based tree search
- **Policy Prior**: Uniform distribution or simple heuristics
- **Value Function**: Minimax evaluation with damage calculations
- **Search Budget**: 500ms default (configurable)
- **Inference**: Parallelized search across multiple sampled states

#### 2.1.3 Integration Layer (Foul-Play)
- **Role**: Bridges poke-engine MCTS with Pokemon Showdown
- **Language**: Python
- **Functions**:
  - Converts Showdown battle state to poke-engine state
  - Handles uncertainty sampling for unrevealed Pokémon
  - Orchestrates parallel MCTS searches
  - Aggregates results into final move selection

### 2.2 Key Technical Challenges

#### 2.2.1 State Representation Mismatch
| Aspect | Metamon (Python) | Poke-Engine (Rust) |
|--------|------------------|-------------------|
| **Format** | Tokenized text + numerical tensors | Structured binary state |
| **Pokémon** | String names → token IDs | Enum indices |
| **Moves** | String names → token IDs | Enum indices |
| **Stats** | Normalized float arrays | Integer values |
| **History** | Sequence of timesteps | Single state snapshot |
| **Uncertainty** | Implicit in observations | Explicit state sampling |

#### 2.2.2 Language Barrier
- Metamon model: Python (PyTorch)
- Poke-engine MCTS: Rust
- Requires inter-process communication or FFI (Foreign Function Interface)

#### 2.2.3 Performance Constraints
- MCTS must evaluate thousands of states within 500ms budget
- Neural network inference adds latency
- Need efficient batching and caching strategies

---

## 3. Requirements

### 3.1 Functional Requirements

#### FR-1: State Translation Layer
**Priority**: P0 (Critical)

**Description**: Bidirectional translation between Metamon observation format and poke-engine state format.

**Requirements**:
- **FR-1.1**: Convert poke-engine `State` struct to Metamon observation format
  - Map Pokémon from enum indices to species names/tokens
  - Map moves from enum indices to move names/tokens
  - Extract and format HP, stats, status conditions
  - Include field conditions (weather, terrain, etc.)
  - Handle team preview state

- **FR-1.2**: Handle uncertainty in state translation
  - Support partially observed states (unrevealed opponent Pokémon)
  - Generate appropriate observation masks for unknown information
  - Maintain consistency with Metamon's training data format

- **FR-1.3**: Optimize translation performance
  - Cache frequently accessed mappings (species/move name lookups)
  - Minimize memory allocations
  - Target <5ms per state translation

**Acceptance Criteria**:
- Translation produces valid Metamon observations that match the model's input schema
- Translated states produce meaningful policy outputs from Metamon model
- Unit tests verify correctness across diverse battle states
- Performance benchmarks meet <5ms target

---

#### FR-2: Neural Policy Interface
**Priority**: P0 (Critical)

**Description**: Interface for querying Metamon model to obtain move probability distributions.

**Requirements**:
- **FR-2.1**: Model loading and initialization
  - Load pretrained Metamon checkpoint (specify which model: Abra, Minikazam, SyntheticRLV2, etc.)
  - Initialize model in evaluation mode (no gradients)
  - Support GPU acceleration if available
  - Handle model versioning and compatibility

- **FR-2.2**: Batch policy inference
  - Accept batch of translated states
  - Return action probability distributions for each state
  - Support dynamic batch sizes (1 to N states)
  - Handle illegal action masking (align with MCTS legal moves)

- **FR-2.3**: Caching and optimization
  - Cache policy outputs for identical states
  - Implement simple memoization for recently seen positions
  - Support asynchronous inference for pipeline parallelism

**Acceptance Criteria**:
- Successfully loads all available Metamon models
- Produces valid probability distributions (sums to 1, non-negative)
- Respects illegal action masks from MCTS
- Achieves <20ms inference time for single state on CPU
- Achieves <50ms inference time for batch of 10 states on CPU

---

#### FR-3: Modified MCTS Algorithm
**Priority**: P0 (Critical)

**Description**: Integrate neural policy priors into poke-engine MCTS node selection and expansion.

**Requirements**:
- **FR-3.1**: Neural prior integration in UCB formula
  - Modify node selection to use AlphaGo-style PUCT formula:
    ```
    UCB(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))
    ```
  - `P(s, a)` = neural policy prior from Metamon
  - `Q(s, a)` = empirical value from MCTS rollouts
  - `c_puct` = exploration constant (tunable hyperparameter)

- **FR-3.2**: Prior initialization at node expansion
  - When expanding a new node, query neural policy interface
  - Initialize child node priors with policy probabilities
  - Fall back to uniform prior if neural inference fails

- **FR-3.3**: Hybrid evaluation
  - Combine neural value estimates with minimax evaluation
  - Support configurable blending ratio
  - Maintain backward compatibility with pure MCTS mode

- **FR-3.4**: Configuration and tuning
  - Expose hyperparameters: `c_puct`, blend ratio, cache size
  - Support A/B testing different configurations
  - Provide sensible defaults based on empirical tuning

**Acceptance Criteria**:
- Modified MCTS produces valid move selections
- Neural priors demonstrably influence search behavior
- Hyperparameters can be adjusted without code changes
- Performance regression <2x compared to baseline MCTS

---

#### FR-4: Inter-Process Communication
**Priority**: P0 (Critical)

**Description**: Efficient communication mechanism between Rust MCTS and Python neural policy.

**Requirements**:
- **FR-4.1**: Communication protocol
  - Define serialization format for state exchange (JSON, MessagePack, or protobuf)
  - Design request/response protocol for policy queries
  - Support batch requests to amortize IPC overhead
  - Include error handling and timeouts

- **FR-4.2**: Implementation options (choose one or support multiple)
  - **Option A**: Python server with Rust client (via HTTP or gRPC)
  - **Option B**: Shared memory with semaphores
  - **Option C**: PyO3 FFI bindings (Rust calling Python directly)
  - **Option D**: Separate process with Unix sockets/pipes

- **FR-4.3**: Performance optimization
  - Minimize serialization/deserialization overhead
  - Support connection pooling or persistent connections
  - Implement request batching and pipelining
  - Graceful degradation if neural server is unavailable

**Acceptance Criteria**:
- Round-trip latency <10ms for single state query
- Supports 100+ queries/second throughput
- Handles network/process failures gracefully
- Well-documented API contract

---

#### FR-5: Baseline Preservation
**Priority**: P0 (Critical)

**Description**: Ensure original Metamon and MCTS models remain functional and unmodified.

**Requirements**:
- **FR-5.1**: Code organization
  - Hybrid model lives in separate module/directory
  - No breaking changes to existing Metamon or poke-engine code
  - Shared utilities extracted to common library

- **FR-5.2**: Configuration-based switching
  - Command-line flags or config files to select model type
  - Defaults to baseline models if hybrid not configured
  - Clear documentation on how to run each model variant

- **FR-5.3**: Versioning and compatibility
  - Hybrid model compatible with same Pokemon Showdown interface
  - Can run all three models (Metamon, MCTS, Hybrid) side-by-side
  - Version tags and changelogs tracked separately

**Acceptance Criteria**:
- Existing Metamon evaluation scripts run unchanged
- Existing MCTS (foul-play) scripts run unchanged
- No regressions in baseline model performance
- Documentation clearly explains model variants

---

### 3.2 Non-Functional Requirements

#### NFR-1: Performance
- **Inference Latency**: Hybrid model completes move selection within 1000ms (2x MCTS baseline)
- **Throughput**: Support 10+ concurrent battles on consumer hardware
- **Memory**: Peak memory usage <4GB per battle instance
- **Scalability**: Efficiently utilize multi-core CPUs for parallelism

#### NFR-2: Reliability
- **Uptime**: Model should handle 1000+ consecutive battles without crashes
- **Error Handling**: Graceful fallbacks if neural inference fails
- **Determinism**: Reproducible results given same random seed
- **Recovery**: Automatically recover from transient failures

#### NFR-3: Maintainability
- **Code Quality**: Follow project style guides (Rust: `cargo fmt`, Python: `ruff`)
- **Documentation**: Comprehensive inline comments and architecture docs
- **Testing**: >80% unit test coverage for new code
- **Modularity**: Clean interfaces between components

#### NFR-4: Usability
- **Configuration**: YAML or TOML config files for hyperparameters
- **Logging**: Structured logs with configurable verbosity
- **Monitoring**: Metrics for search depth, cache hit rate, neural query count
- **Debugging**: Tools to visualize search trees and policy distributions

---

## 4. System Architecture

### 4.1 High-Level Design

```
┌─────────────────────────────────────────────────────────────┐
│                   Pokemon Showdown Server                    │
└─────────────────────────────────────────────────────────────┘
                            ▲ │
                            │ │ WebSocket (battle messages)
                            │ ▼
┌─────────────────────────────────────────────────────────────┐
│              Foul-Play Integration Layer (Python)            │
│  - Parse Showdown messages                                   │
│  - Convert to poke-engine state                              │
│  - Sample unrevealed Pokemon                                 │
│  - Coordinate parallel searches                              │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ State translation
                            ▼
┌─────────────────────────────────────────────────────────────┐
│         Hybrid MCTS Engine (Rust + Python Bridge)            │
│                                                               │
│  ┌───────────────────────┐      ┌──────────────────────┐    │
│  │   MCTS Core (Rust)    │◄────►│ Neural Policy Server │    │
│  │   - Tree search       │ IPC  │    (Python)          │    │
│  │   - UCB selection     │      │  - State translator  │    │
│  │   - Rollout sims      │      │  - Metamon inference │    │
│  │   - Value backup      │      │  - Policy cache      │    │
│  └───────────────────────┘      └──────────────────────┘    │
│                                                               │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ Move selection
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  Battle Decision Output                      │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Component Breakdown

#### 4.2.1 State Translation Module (Python)
**Location**: `hybrid_mcts/state_translator.py`

**Responsibilities**:
- Convert `poke_engine.State` (from Rust) to Metamon observation format
- Handle enum → string mappings for species and moves
- Extract relevant features for transformer input
- Apply tokenization using Metamon's tokenizer
- Generate observation masks for partial observability

**Key Classes**:
```python
class StateTranslator:
    def __init__(self, tokenizer: PokemonTokenizer, observation_space: ObservationSpace)
    def translate(self, state: State) -> Dict[str, np.ndarray]
    def batch_translate(self, states: List[State]) -> Dict[str, np.ndarray]
```

#### 4.2.2 Neural Policy Server (Python)
**Location**: `hybrid_mcts/policy_server.py`

**Responsibilities**:
- Load pretrained Metamon model
- Serve policy inference requests via IPC
- Batch requests for efficiency
- Cache policy outputs
- Handle timeouts and errors

**Key Classes**:
```python
class NeuralPolicyServer:
    def __init__(self, model_name: str, device: str = "cpu")
    def get_policy(self, state: State) -> np.ndarray
    def get_policy_batch(self, states: List[State]) -> np.ndarray
    def start_server(self, port: int)
    def shutdown(self)
```

#### 4.2.3 MCTS Integration Layer (Rust)
**Location**: `vendor/poke-engine/src/neural_mcts.rs`

**Responsibilities**:
- Extend existing MCTS with neural prior support
- Communicate with Python policy server
- Implement PUCT formula
- Manage neural prior cache
- Fallback to heuristic if neural unavailable

**Key Structs/Functions**:
```rust
pub struct NeuralNode {
    pub base: Node,
    pub policy_prior: Option<Vec<f32>>,
    pub neural_value: Option<f32>,
}

pub struct NeuralMCTS {
    pub policy_client: Option<PolicyClient>,
    pub c_puct: f32,
    pub use_neural_prior: bool,
}

impl NeuralMCTS {
    pub fn query_policy(&mut self, state: &State) -> Vec<f32>;
    pub fn puct_selection(&self, node: &NeuralNode) -> usize;
    pub fn search(&mut self, state: &State, time_budget: Duration) -> MoveChoice;
}
```

#### 4.2.4 IPC Bridge (Rust ↔ Python)
**Location**: `hybrid_mcts/ipc_bridge.rs` and `hybrid_mcts/ipc_client.py`

**Responsibilities**:
- Serialize poke-engine state to JSON/MessagePack
- Send HTTP/gRPC requests to policy server
- Deserialize policy distributions
- Handle connection pooling and retries

---

## 5. Data Flow

### 5.1 Training Phase (Offline)
```
Human Replays + Self-Play Data
           │
           ▼
   Metamon RL Training
    (Actor-Critic)
           │
           ▼
  Pretrained Checkpoints
   (Abra, Minikazam, etc.)
```

### 5.2 Inference Phase (Online)

```
1. Battle State from Showdown
   │
   ▼
2. Foul-Play converts to poke-engine State
   │
   ▼
3. Sample N possible states (handle uncertainty)
   │
   ▼
4. For each sampled state:
   │
   ├─► 4a. MCTS root node created
   │   │
   │   ▼
   │   4b. MCTS selection (PUCT with neural prior)
   │   │
   │   ├─► 4b.1: Query Neural Policy Server
   │   │          - Translate state
   │   │          - Inference Metamon model
   │   │          - Return P(a|s)
   │   │
   │   ▼
   │   4c. Node expansion with neural priors
   │   │
   │   ▼
   │   4d. Rollout simulation
   │   │
   │   ▼
   │   4e. Backup values (combine neural + empirical)
   │   │
   │   └─► Repeat 4b-4e until time budget exhausted
   │
   ▼
5. Aggregate results across sampled states
   │
   ▼
6. Select move with highest visit count / value
   │
   ▼
7. Send move to Showdown
```

---

## 6. Implementation Plan

### 6.1 Milestones

#### Milestone 1: State Translation & Neural Interface (2 weeks)
**Deliverables**:
- State translator with unit tests
- Neural policy server with API
- Benchmark performance of translation and inference

**Success Metrics**:
- Translation accuracy: 100% on test suite
- Inference latency: <20ms single state, <50ms batch of 10

#### Milestone 2: IPC Communication (1 week)
**Deliverables**:
- IPC bridge implementation (HTTP or PyO3)
- Integration tests for Rust ↔ Python communication
- Performance profiling and optimization

**Success Metrics**:
- Round-trip latency: <10ms
- Throughput: >100 queries/second
- Zero failures in 10,000 request stress test

#### Milestone 3: MCTS Algorithm Modification (2 weeks)
**Deliverables**:
- Neural MCTS implementation in Rust
- PUCT formula with configurable hyperparameters
- Unit tests for search algorithm correctness

**Success Metrics**:
- Search correctness: matches AlphaGo-style PUCT behavior
- Performance: <2x slowdown vs baseline MCTS
- Hyperparameter tuning framework in place

#### Milestone 4: Integration & Testing (2 weeks)
**Deliverables**:
- End-to-end integration with Foul-Play
- Hybrid model runs on Pokemon Showdown
- Battle logging and analysis tools

**Success Metrics**:
- Successfully completes 100 consecutive battles
- No crashes or hangs
- Logging captures search statistics

#### Milestone 5: Evaluation & Tuning (2 weeks)
**Deliverables**:
- Head-to-head evaluation vs Metamon baseline
- Hyperparameter tuning experiments
- Performance optimization based on profiling

**Success Metrics**:
- >5% win rate improvement over Metamon baseline
- <1000ms average move decision time
- Identified optimal hyperparameters (c_puct, search time, etc.)

### 6.2 Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| State translation errors | Medium | High | Extensive unit tests, manual verification of edge cases |
| IPC latency bottleneck | High | High | Profile early, consider PyO3 FFI if HTTP too slow |
| Neural model incompatibility | Low | Medium | Validate against multiple Metamon checkpoints |
| MCTS search quality degrades | Medium | High | A/B test with pure MCTS, tune c_puct carefully |
| Memory leaks in long runs | Medium | Medium | Stress testing, profiling with valgrind/heaptrack |
| Model doesn't improve over baseline | Low | High | Fallback: analyze failure modes, iterate on design |

---

## 7. Testing Strategy

### 7.1 Unit Tests
- **State Translator**: Test all Pokémon species, moves, abilities, field conditions
- **Neural Server**: Test model loading, batching, caching, error handling
- **IPC Bridge**: Test serialization, network failures, timeouts
- **MCTS Algorithm**: Test PUCT formula, prior initialization, value backup

### 7.2 Integration Tests
- **End-to-End**: Run full battles with hybrid model on test scenarios
- **Baseline Comparison**: Verify Metamon and pure MCTS still work
- **Stress Testing**: 1000+ battle marathon, memory profiling

### 7.3 Evaluation Protocol

#### 7.3.1 Baseline Comparisons
Run each model for 200 battles on gen9randombattle:
1. **Metamon (Minikazam)**: Pure RL baseline
2. **MCTS (Foul-Play)**: Pure search baseline
3. **Hybrid (Neural-MCTS)**: New model

#### 7.3.2 Head-to-Head Matches
- Hybrid vs Metamon: 500 battles (expect >55% win rate for hybrid)
- Hybrid vs MCTS: 500 battles (assess trade-offs)
- All models vs heuristic baselines: validate general competence

#### 7.3.3 Ablation Studies
- Neural prior only (no MCTS search)
- MCTS only (no neural prior)
- Varying c_puct values: [0.5, 1.0, 2.0, 5.0]
- Varying search time: [250ms, 500ms, 1000ms, 2000ms]

#### 7.3.4 Metrics
- **Win Rate**: Primary metric
- **Average Turn Length**: Measure battle efficiency
- **Move Quality**: Analyze decisions in critical positions
- **Search Statistics**: Depth reached, nodes expanded, cache hit rate
- **Latency**: P50, P95, P99 decision times

---

## 8. Success Metrics & KPIs

### 8.1 Primary Success Metrics
1. **Win Rate vs Baseline**: Hybrid achieves ≥55% win rate against Metamon (Minikazam)
2. **Inference Performance**: Average decision time ≤1000ms
3. **Stability**: Zero crashes in 1000-battle stress test

### 8.2 Secondary Metrics
1. **Strategic Play Quality**: Improved performance in complex scenarios (e.g., endgame positions, team preview)
2. **Search Efficiency**: Higher average search depth than pure MCTS
3. **Neural Utilization**: >90% of MCTS nodes use neural priors
4. **Cache Effectiveness**: >50% cache hit rate for policy queries

### 8.3 Long-Term Metrics
1. **Generalization**: Performance across multiple generations (gen8, gen7, etc.)
2. **Team Diversity**: Win rate across different team compositions
3. **Robustness**: Performance against novel opponents (not seen in training)

---

## 9. Open Questions & Future Work

### 9.1 Open Questions
1. **Which Metamon model to use?**
   - Minikazam (small, fast)
   - Abra (medium, Gen 9 specialist)
   - SyntheticRLV2 (large, best performance)

2. **How to handle value function blending?**
   - Pure MCTS rollout values
   - Pure neural value estimates
   - Weighted combination (tunable)

3. **Should we retrain Metamon with MCTS targets?**
   - Use MCTS search results as training signal
   - Iterative improvement (AlphaZero-style)

4. **How to handle unrevealed opponent Pokémon?**
   - Sample multiple possible teams
   - Average policy across samples
   - Use opponent modeling

### 9.2 Future Enhancements
1. **AlphaZero-style Self-Play Training**
   - Use hybrid model to generate training data
   - Retrain Metamon on MCTS-improved games
   - Iterate to convergence

2. **Adaptive Search Budget**
   - Allocate more time to critical decisions
   - Use fast heuristics for trivial moves

3. **Multi-Agent Coordination**
   - Ensemble of different Metamon models
   - Mixture-of-experts approach

4. **Interpretability Tools**
   - Visualize search trees
   - Highlight high-value branches
   - Explain move choices to users

---

## 10. Dependencies & Constraints

### 10.1 Technical Dependencies
- **Python**: ≥3.11 (for Metamon and integration layer)
- **Rust**: ≥1.70 (for poke-engine)
- **PyTorch**: ≥2.0 (for Metamon model)
- **poke-engine**: Gen 9 build with MCTS support
- **AMAGO**: For Metamon model loading
- **poke-env**: For battle state parsing

### 10.2 Hardware Requirements
- **Minimum**: 4-core CPU, 8GB RAM
- **Recommended**: 8-core CPU, 16GB RAM, GPU (for faster inference)
- **Distributed**: Multiple machines for large-scale evaluation

### 10.3 Constraints
- **Backward Compatibility**: Must not break existing Metamon or MCTS workflows
- **Showdown Compliance**: Respect 1000ms move timer (with buffer)
- **Resource Usage**: Peak memory <4GB per battle instance
- **Code Licensing**: Maintain compatibility with existing open-source licenses

---

## 11. Stakeholders & Communication

### 11.1 Stakeholders
- **Primary**: Development team implementing the hybrid model
- **Secondary**: Researchers evaluating model performance
- **Tertiary**: Community users testing on Pokemon Showdown

### 11.2 Communication Plan
- **Weekly Standups**: Progress updates, blockers, next steps
- **Milestone Reviews**: Demo functionality, discuss results
- **Documentation**: Keep CLAUDE.md updated with new architecture
- **Public Updates**: Share evaluation results with community (if applicable)

---

## 12. Appendix

### 12.1 Glossary
- **MCTS**: Monte Carlo Tree Search
- **PUCT**: Polynomial Upper Confidence Trees (UCT variant)
- **RL**: Reinforcement Learning
- **IL**: Imitation Learning
- **IPC**: Inter-Process Communication
- **FFI**: Foreign Function Interface
- **UCB**: Upper Confidence Bound

### 12.2 References
- [Metamon Paper](https://arxiv.org/abs/2410.17207)
- [AlphaGo Paper](https://www.nature.com/articles/nature16961)
- [Poke-Engine Repository](https://github.com/pmariglia/poke-engine)
- [AMAGO Framework](https://github.com/UT-Austin-RPL/amago)
- [Foul-Play Repository](https://github.com/pmariglia/foul-play)

### 12.3 Version History
| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-10-18 | Product Team | Initial draft |

---

## 13. Approval

**Reviewed By**:
- [ ] Engineering Lead
- [ ] Research Lead
- [ ] Product Manager

**Approved By**:
- [ ] Technical Director
- [ ] Project Sponsor

**Approval Date**: _________________
