# pokeagent_ac_mcts
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