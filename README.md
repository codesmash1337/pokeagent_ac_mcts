# Pokeagent project
Repo for AC-boosted MCTS

Repository looks as follows:

- vendor/
- metamon/
  - This repo holds code related to the creation and training of metamon models, along with their interface to pokemon showdown
- amago/
  - This repo gives insight into how metamon models run inference  
- foul-play/
  - Python-based MCTS orchestrator
- poke-engine/
  - Rust-based pokemon simulator that also houses MCTS implementation

# TODO
Model Side:
1. Add inference for value() and policy() given a metamon actor critic object (should be one pass)
2. Properly parse a state from poke-engine into an observation state that a metamon AC model can interpret

MCTS Side:
1. Augment MCTS to be AC-boosted
2. Expose low-latency functionality to call the value/policy heads that sends a poke-engine state and takes the policy/actor values

Orchestrator Side:
1. Successfully connect to a Showdown port and run the AC-boosted MCTS

Scaling:
1. Run the MCTS with distributed resources for GPU + greater parallelism

# Update vendors

Make sure we're up to date with the remote
```git fetch amago-upstream```
Pull new commits from upstream into your vendored copy
```git subtree pull --prefix third_party/amago amago-upstream "$BRANCH" --squash```
Compile rust via
```
 cargo build --release \
    --no-default-features \
    --features "gen9 terastallization neural"
```



> Neural Hook-Up

  - Added neural_evaluate.rs with a Python-backed evaluator that lazily
  spins up NeuralInferenceRunner, caches the Abra runner, and turns its
  state_value into a 0–1 score (maps [-1, 1] → [0, 1]). POKEENGINE_MODEL,
  POKEENGINE_CHECKPOINT, and POKEENGINE_FORMAT env vars control which runner/
  format/checkpoint to load (vendor/poke-engine/src/neural_evaluate.rs).
  - Swapped the MCTS evaluation pipeline over to the neural path when it’s
  available: every rollout now calls evaluate_with_fallback, and only falls
  back to the old heuristic when the Python call fails (vendor/poke-engine/
  src/mcts.rs).
  - Exported the helper via lib.rs and introduced a neural feature that
  pulls in pyo3/once_cell so the base crate stays lean when you don’t need
  the neural bridge (vendor/poke-engine/Cargo.toml, vendor/poke-engine/src/
  lib.rs).
  - Tweaked the Python tests so the pytest suite can import our in-tree module
  before any globally installed wheel (vendor/poke-engine/poke-engine-py/
  python/tests/test_poke_engine.py).

  What You Need To Run It

  1. Build with the neural feature:

     cargo build --release --features gen9,neural
OR
    cd /Users/danielchristl/pokeagent_ac_mcts/vendor/poke-engine/poke-engine-py
  maturin develop --no-default-features --features "poke-engine/gen9 poke-engine/neural"  
  2. Ensure Python can import the package:

     python -m pip install -e vendor/poke-engine/poke-engine-py
     export PYTHONPATH="$PWD/vendor/poke-engine/poke-engine-py/python:
  $PYTHONPATH"
  3. Configure the runner:

     export METAMON_CACHE_DIR=/path/to/metamon_cache
     export POKEENGINE_MODEL=Abra          # optional (default Abra)
     export POKEENGINE_CHECKPOINT=40       # optional
     export POKEENGINE_FORMAT=gen9ou       # optional

  Quick Local Smoke Test

  # Python-side confirmation with Abra
  pytest vendor/poke-engine/poke-engine-py/python/tests/
  test_poke_engine.py::test_neural_runner_with_abra_prints_outputs -s

  # Rust-side sanity check (falls back to neural when env is set)
  cargo test --features neural mcts -- --nocapture

  Connecting To A Live Server
  If your next step is running this against a real ladder, wire the Python
  package into whatever client you use to talk to Showdown (e.g., foul-play or
  poke-env):

  1. Install poke-engine-py (step 2 above) in the same venv that runs your
  3. Start your client with the neural feature enabled on the Rust side
  (compiled binary) so the engine can pull values from the Python runner while
  it talks to the live server.

  That’s it—the Rust MCTS now defers to the Python neural runner for value
  estimates, while staying drop-in compatible when the neural feature is
  disabled.


  RUST_BACKTRACE=1 python vendor/foul-play/run.py \
    --websocket-uri ws://localhost:8000/showdown/websocket \
    --ps-username PAC-haunter \
    --ps-password willmom \
    --bot-mode search_ladder \
    --pokemon-format gen9ou \
    --team-name gen9/ou/team0 \
    --search-time-ms 10 \
    --search-parallelism 1



    Current flow for testing MCTS:

    Recompile Rust with
    maturin develop --no-default-features --features "poke-engine/gen9 poke-engine/terastallization poke-engine/neural"

    Run with
RUST_BACKTRACE=1 python vendor/foul-play/run.py \   
    --websocket-uri ws://localhost:8000/showdown/websocket \   
    --ps-username PAC-haunter \   
    --ps-password willmom \   
    --bot-mode search_ladder \   
    --pokemon-format gen9ou \   
    --team-name gen9/ou/team0 \   
    --search-time-ms 10 \
    --search-parallelism 1