"""The discounted episodic MDP specified in CIS 5930 HW1."""

import gymnasium as gym
from gymnasium import spaces
from gymnasium.error import ResetNeeded


class IcyGridWorldEnv(gym.Env):
    """A fixed map with 80% intended motion and two 10% perpendicular slips.

    Observations are row-major IDs, including unused IDs for obstacle cells.
    The base environment has no time limit. Learning algorithms use gamma=0.99.
    """

    metadata = {"render_modes": ["ansi", "human"], "render_fps": 4}
    obstacles = frozenset(
        {(r, 5) for r in range(16) if r not in {4, 12}}
        | {(r, 10) for r in range(4, 20) if r not in {8, 16}}
        | {(r, 15) for r in range(16) if r not in {6, 13}}
        | {(9, c) for c in range(6, 10) if c != 8}
        | {(14, c) for c in range(11, 15) if c != 13}
    )
    start = (19, 0)
    goal = (0, 19)
    _deltas = ((-1, 0), (0, 1), (1, 0), (0, -1))

    def __init__(self, render_mode=None):
        if render_mode not in (None, "ansi", "human"):
            raise ValueError("render_mode must be None, 'ansi', or 'human'")
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(400)
        self._position = None
        self._steps = 0
        self._terminated = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._position = self.start
        self._steps = 0
        self._terminated = False
        if self.render_mode == "human":
            self.render()
        return 380, {"position": self.start, "steps": 0}

    def _validate_action(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action!r}; expected an integer in 0..3")
        return int(action)

    def _move(self, position, direction):
        dr, dc = self._deltas[direction]
        r, c = position[0] + dr, position[1] + dc
        blocked = not (0 <= r < 20 and 0 <= c < 20) or (r, c) in self.obstacles
        return (position if blocked else (r, c)), blocked

    @staticmethod
    def _outcomes(action):
        return ((action, 0.80), ((action - 1) % 4, 0.10), ((action + 1) % 4, 0.10))

    def step(self, action):
        action = self._validate_action(action)
        if self._position is None or self._terminated:
            raise ResetNeeded("Call reset() before stepping or after episode termination")
        outcomes = self._outcomes(action)
        index = int(self.np_random.choice(3, p=[p for _, p in outcomes]))
        executed = outcomes[index][0]
        self._position, blocked = self._move(self._position, executed)
        self._steps += 1
        self._terminated = self._position == self.goal
        reward = 100.0 if self._terminated else -1.0
        info = {
            "position": self._position,
            "steps": self._steps,
            "executed_action": executed,
            "slipped": executed != action,
            "blocked": blocked,
        }
        if self.render_mode == "human":
            self.render()
        r, c = self._position
        return 20 * r + c, reward, self._terminated, False, info

    def transition_probabilities(self, state, action):
        """Return {next_observation: probability}, with duplicate outcomes merged.

        This pure query does not change state or consume any random numbers.
        The mathematical goal state is absorbing, even though step requires reset
        after termination.
        """
        action = self._validate_action(action)
        if not self.observation_space.contains(state):
            raise ValueError(f"Invalid state {state!r}; expected an integer in 0..399")
        position = divmod(int(state), 20)
        if position in self.obstacles:
            raise ValueError(f"State {state!r} is an obstacle")
        if position == self.goal:
            return {19: 1.0}
        result = {}
        for direction, probability in self._outcomes(action):
            (r, c), _ = self._move(position, direction)
            observation = 20 * r + c
            result[observation] = result.get(observation, 0.0) + probability
        return result

    def render(self):
        if self.render_mode is None:
            return None
        if self._position is None:
            raise ResetNeeded("Call reset() before rendering")
        rows = []
        for r in range(20):
            row = []
            for c in range(20):
                position = (r, c)
                if position == self._position:
                    cell = "*" if position == self.goal else "A"
                elif position == self.goal:
                    cell = "G"
                else:
                    cell = "#" if position in self.obstacles else "."
                row.append(cell)
            rows.append("".join(row))
        frame = "\n".join(rows)
        if self.render_mode == "human":
            print(frame)
            return None
        return frame
