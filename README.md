# Icy Grid World

Reusable Python 3.10+ / Gymnasium 1.x implementation of the CIS 5930 HW1 MDP.

## Install

```sh
git clone https://github.com/acharland1/CIS5930_GroupCompetition.git
cd CIS5930_GroupCompetition
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` in Windows PowerShell or
`source .venv/bin/activate` on macOS/Linux. Then run:

```sh
python -m pip install -r requirements.txt
python -m pip install -e .
```

Alternatively, `python -m pip install -e ".[test]"` installs the package and tests
in one command. For runtime only, use `python -m pip install .`.

## Test and demonstrate

```sh
python -m pytest -q
python demo.py --seed 42 --episodes 3 --policy optimal --render ansi
python demo.py --seed 42 --episodes 3 --policy random --render none
```

The default demo computes a policy by value iteration using the exact transition
helper and gamma=0.99. This solves the discounted MDP; it does not optimize a
finite 500-step horizon. ANSI mode shows the final board, human mode prints every
frame, and none prints episode summaries only. Environment and action-space
seeds are set separately; later episodes continue the existing random streams.

## Python usage

```python
import gymnasium as gym
import icy_gridworld

env = gym.make("IcyGridWorld-v0", render_mode="ansi")
observation, info = env.reset(seed=42)
env.action_space.seed(42)
while True:
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        break
print(env.render())
env.close()
```

`env.unwrapped.transition_probabilities(380, 2)` returns
`{380: 0.9, 381: 0.1}` (up to floating-point representation). It is a pure query,
valid before reset, and rejects obstacle IDs, invalid states, and invalid actions.
Use `IcyGridWorldEnv` directly for a base environment without a time limit.

## MDP and implementation

- Coordinates are `(row, column)`, top-left `(0, 0)`. The fixed start is `(19, 0)`
  and the goal is `(0, 19)`. The obstacle equations are in `icy_gridworld/env.py`;
  `obstacles` is a frozenset, and `start` and `goal` are tuples, read-only by convention.
- Observations encode `20 * row + column`: start 380, goal 19. Discrete(400)
  includes 48 obstacle IDs that are never emitted; the MDP has 352 states.
- Actions are 0 up, 1 right, 2 down, 3 left. Requested motion occurs with
  probability 0.80; each perpendicular direction occurs with probability 0.10.
  There is no backward slip. Only `self.np_random` samples slips.
- Edges and rocks leave the position unchanged and still count as a step.
  Both `step` and the exact query share direction probabilities and collision logic;
  the query merges probabilities when multiple outcomes reach the same state.
- Entering gold earns +100 and terminates. Every other transition earns -1,
  including collisions. The learning algorithm uses gamma=0.99; step returns
  immediate rewards only. Expected reward is `101 * P(goal) - 1` outside gold.
- The mathematical goal self-loops with probability 1 and reward 0. Runtime
  callers must reset after termination; the base class raises `ResetNeeded`
  before the initial reset and after termination.
- Registration applies Gymnasium's external TimeLimit at transition 500.
  The base always returns `truncated=False`. Gold on transition 500 can yield
  both `terminated=True` and `truncated=True`. Reset after either flag.
- Step info contains position, steps, executed_action, slipped, and blocked.
  Reset ignores options and returns the fixed start and zero steps.
- Rendering produces exactly 20 rows of 20 characters: A agent, G gold,
  # rock, . empty, and * agent on gold. No graphics dependency is needed.

Tests include an independent map/oracle for all 1,408 state-action queries,
every possible nonterminal executed outcome, 240,000 fixed-seed sampling trials,
Gymnasium's checker, reward examples, invalid inputs, RNG continuity and purity,
rendering, and separate/simultaneous termination and truncation.

API references: [Gymnasium Env](https://gymnasium.farama.org/api/env/) and
[registration](https://gymnasium.farama.org/api/registry/). HW1.pdf is the
authoritative assignment specification. See AI_LOG.md for AI assistance and
verification details. The repository URL still needs to be submitted through
the course system; a private repository requires instructor access.
