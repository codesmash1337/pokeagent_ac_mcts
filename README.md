# Pokeagent Project

This repository includes all the work related to my submission for the [pokeagent](https://pokeagent.github.io/) competition.

---

## Foreword

This project wouldn't have been possible without borrowing a tremendous amount from my vendors:

- **amago** and **metamon** (courtesy of Jake Grigsby, one of the organizers of the pokeagent competition)
- **foulplay** and **pokeengine** (courtesy of pmariglia, a long-time Pokemon AI programmer and winner of the gen9ou portion)

I thank them both for their resources and brilliant repositories. I would also like to thank my fellow contributors in the challenge, particularly psriram4 and SirNeural, for their insight, commitment, and perseverance.

---

## Summary of Approach

I would breakdown my approach into 3 categories: **teambuilding**, **battling**, and **tournament-time adjustments**. Although battling takes the majority of the glory in the literature (and in my approach), I strongly believe that the future relies on a balance between the three pillars.

---

## Teambuilding

My teambuilding implementation is relatively simple and can be further broken down into two categories: creation and ranking.

### Creation

For "creating" the teams themselves, I rely on a combination of:
- Scraping the Smogon forums (sample teams and high elo performers)
- Reconstructing teams from battles
- Creating my own teams (PREMO)

This allows me to obtain a population of teams. I've also used my domain knowledge to acquire what I believe is a balance of teams across different styles and tiers.

### Ranking

To rank the teams, I run a round-robin style tournament (`run_tournament.py`) amongst a selected population of teams and use Bradley-Terry (BT) across a log of all games to rank my teams (`offline_bt.py`).

The battler must be the same in a given log file for consistent results. To be precise, this gives a ranking of how good one model is against itself; obviously this is only a proxy for actual team strength, but in practice I've found this method gives a **~200 ELO gain** over simple random selection.

BT is the preferred algorithm to rank my k-dueling bandits because it is:
- Order-agnostic (unlike ELO)
- Provides lower bounds and upper bounds
- Has nice statistical guarantees

---

## Battling

Battling is where the majority of the engineering effort and novelty comes from. In short, my goal is to create an **AlphaZero-style implementation** of Pokemon through a combination of actor-critic agents from metamon and a modification to the Pokemon MCTS implementation in foul play.

**Reference:** PUCT-based MCTS chooses its next action by balancing:
- **Exploitation** (proportional to the critic's value)
- **Exploration** (proportional to the actor's policy)

This paradigm informs points 4-7 below.

### Key Implementations

1. **Move Metamon observation state to Rust**  
   Move the creation of the Metamon observation state (originally in Python from PokeEnv) to Rust from PokeEngine. This sounds simple but was actually where a lot of our time was spent and from where many headaches originated.  
   📁 `vendor/poke-engine/poke-engine-py/src/observation.rs`

2. **Dissect Metamon architecture**  
   Return state values and probability distributions in the modified architecture.  
   📁 `vendor/metamon/metamon/stateful_inference.py`

3. **Modify vanilla MCTS**  
   Retrieve the (policy, value) for each side. Use the priors for search in classic PUCT fashion and replace the MCTS rollout/heuristic with the value.  
   📁 `vendor/poke-engine/poke-engine-py/src/mcts.rs`

4. **Virtual loss**  
   Because it is extremely slow to run a single inference at a time, we need to batch our packaged states before sending them over to Python to run inference. However, vanilla MCTS requires us to backprop before selecting the next node. To remedy this, we artificially mark a selected node as a loss (hence, virtual loss) and add it to a list. Once our list hits the given batch size, we send the batch over, process in parallel, and undo the virtual loss. This has the added benefit of encouraging more exploration.

5. **Smoothen priors**  
   One major problem was that our actor-critic model was highly overconfident in one option (mean of 0.9 confidence across ~50 battles across ~13 options). Obviously, one option is to retrain the model with some algorithm that encourages less peaked priors; however, for the sake of time, we decided to go with a simpler approach by injecting temperature into the priors before returning to MCTS. This gave us a more smoothened exploration policy that helped us avoid exactly mimicking the baseline priors' choices.

6. **Value overconfidence and strengthening exploitation**  
   Still another major problem revolved around a highly overconfident value from the model. Its range is from -1100 to 1100, but it had a mean of ~900. To combat this, we took the value from both sides' perspectives and subtracted s1-s2 to get the advantage. But at early-mid stages of the game, the advantage differentiation between two similar states tended to be extremely small (especially relative to the exploration term).  
   
   Rather than simply adjusting the constant in front of exploration, which has the drawback of being static throughout the game, we decided to standardize the exploitation term across all actions at a given node via z-value. With small standard deviations, however, this z-value can explode; therefore we use the *local* mean and the *global* standard deviation to achieve a balance of local differentiality and global stability.

7. **Strengthening exploration**  
   Finally, we run into a problem where the exploitation term can dominate post-standardization; this is partially because exploration isn't on the same scale as exploitation. For instance, if we have an extremely high exploitation value, exploration can never negate it because it is strictly non-negative. This gives us an asymmetric situation where one action can be particularly strong and always be chosen for expansion.  
   
   We need to move one to the same scale as the other! As a remedy, we introduce adaptive U-gain in PopArt style on the exploration term, which gives balance between exploration and exploitation.

> **Note:** We also have some other techniques as optional flags, but those listed above are the main ones on our hot path.

---

## Tournament Time

Another underexplored area, I took a relatively simple approach: take my top n teams and use UCB at tournament time to choose the next team. The pokeagent tournament was best-of-99, so I used an aggressive half-life in front of my UCB exploration term to help the model choice rapidly favor exploitation.

**Future work:** The initial selection of teams should probably balance raw strength and team variety.

---

## Conclusion

And that's about it! For future work in this direction, I definitely think training a model with MCTS in mind is a crucial step; that way, you don't run into some of the nasty problems we had to grapple with this challenge.

Furthermore, I think doing some sort of joint training between teambuilding and battling, then collecting a population of diverse teams to use at tournament time would be a great way to tie everything together rather than having three separate systems.

---

## Engineering Guide

### Repository Structure

```
vendor/
├── metamon/          # Creation and training of metamon models + interface to Pokemon Showdown
├── amago/            # Insight into how metamon models run inference
├── foul-play/        # Python-based MCTS orchestrator
└── poke-engine/      # Rust-based Pokemon simulator that also houses MCTS implementation
```

---

### Update Vendors

Make sure we're up to date with the remote:

```bash
git fetch amago-upstream
```

Pull new commits from upstream into your vendored copy:

```bash
git subtree pull --prefix third_party/amago amago-upstream "$BRANCH" --squash
```

---

### Running the Project

#### 1. Compile Rust

```bash
cd poke-engine/poke-engine-py
maturin develop --no-default-features --features "poke-engine/gen9 poke-engine/terastallization poke-engine/neural"
```

#### 2. Run the Bot

```bash
export METAMON_CACHE_DIR=metamon_cache 
RUST_BACKTRACE=full python vendor/foul-play/run.py \
  --websocket-uri ws://localhost:8000/showdown/websocket \
  --ps-username username \
  --ps-password password \
  --bot-mode search_ladder \
  --pokemon-format gen9ou \
  --team-name gen9/ou/team0 \
  --search-time-ms 5000 \
  --search-parallelism 1 \
  --run-count 1 > output_22.txt 2>&1
```

> **⚠️ Note:** On MPS you may need to run `TORCH_COMPILE_DISABLE=1`
> 
> **⚠️ Important:** Make sure to set all hyperparameters appropriately in MCTS!