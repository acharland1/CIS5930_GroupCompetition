# AI assistance log

## 2026-09-13: requirements and implementation

- The user asked Codex to implement HW1.pdf and upload the necessary files to
  acharland1/CIS5930_GroupCompetition.
- Codex extracted all six pages of the handout and checked the official Gymnasium
  environment and registration API documentation.
- Codex implemented the fixed obstacle equations, row-major observations, shared
  collision/outcome logic, seeded slip sampling, exact probability queries,
  rendering, and registration with an external 500-step TimeLimit.
- The mathematical goal is absorbing in the query helper. The runtime rejects
  step before reset or after termination, where the handout requires a reset.
- No human review or course submission is claimed by this log. The student remains
  responsible for reviewing and explaining every submitted line.

## 2026-09-13: tests, demonstration, and review

- Codex authored tests using the handout's ASCII map as an independent obstacle
  oracle and separate displacement/collision calculations. Tests enumerate all
  352 states and four actions, including the absorbing goal, and force all three
  possible executed directions for every nonterminal state/action pair.
- Statistical tests sample 20,000 transitions for each of four actions at each of
  three representative positions (interior, start corner, and goal neighbor):
  240,000 transitions total. They compare both direction and destination frequencies
  against the kernel with absolute tolerance 0.015 and verify no backward slips.
- Contract tests cover invalid inputs, Python and NumPy integer inputs, seeded
  trajectory equality, reset(seed=None) RNG continuity, pure transition queries,
  all rendering symbols, counters, collision rewards, goal rewards, and the
  Gymnasium environment checker.
- Wrapper tests verify truncation exactly at step 500, both flags when entering
  gold on step 500, and no time limit in the base environment.
- Codex added a CLI demo with random actions and exact-model value iteration
  (gamma=0.99), installation metadata, and user documentation.
- Verification environment: Windows, Python 3.12, Gymnasium 1.3.0, pytest 9.1.1.
  Editable installation via `python -m pip install -e ".[test]"` succeeded.
- `python -m pytest -q`: **82 passed** (2.04 seconds).
- `python demo.py --seed 42 --episodes 3 --policy optimal --render none`:
  44, 40, and 46 steps; rewards 57, 61, and 55; each terminated at gold.
- The same random-policy demo produced three 500-step truncations, each reward -500.
- `git diff --check` passed. Review confirmed no backwards slips, no internal
  truncation, and no global RNG use. Generated caches and the virtual environment
  are excluded from version control.
- These are Codex's automated checks and review, not independent human validation.
