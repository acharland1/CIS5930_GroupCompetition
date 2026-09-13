"""Independent specification oracle, deterministic contracts, and sampling checks."""

from collections import Counter
from copy import deepcopy

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.error import ResetNeeded
from gymnasium.utils.env_checker import check_env

from icy_gridworld import IcyGridWorldEnv


# Transcribed from the handout picture, independently of the implementation sets.
MAP = (
    ".....#.........#...G",
    ".....#.........#....",
    ".....#.........#....",
    ".....#.........#....",
    "..........#....#....",
    ".....#....#....#....",
    ".....#....#.........",
    ".....#....#....#....",
    ".....#.........#....",
    ".....###.##....#....",
    ".....#....#....#....",
    ".....#....#....#....",
    "..........#....#....",
    ".....#....#.........",
    ".....#....###.##....",
    ".....#....#....#....",
    "....................",
    "..........#.........",
    "..........#.........",
    "S.........#.........",
)
ROCKS = {(r, c) for r, row in enumerate(MAP) for c, cell in enumerate(row) if cell == "#"}
STATES = [s for s in range(400) if divmod(s, 20) not in ROCKS]
DELTAS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def oracle_move(state, direction):
    r, c = divmod(state, 20)
    dr, dc = DELTAS[direction]
    rr, cc = r + dr, c + dc
    blocked = not (0 <= rr < 20 and 0 <= cc < 20) or (rr, cc) in ROCKS
    return (state if blocked else 20 * rr + cc), blocked


class FixedOutcome:
    def __init__(self, index):
        self.index = index

    def choice(self, count, p):
        assert count == 3
        assert p == [0.8, 0.1, 0.1]
        return self.index


def test_map_reset_spaces_and_checker():
    env = IcyGridWorldEnv()
    assert isinstance(env, gym.Env)
    assert isinstance(env.obstacles, frozenset)
    assert env.obstacles == ROCKS
    assert len(ROCKS) == 48 and len(STATES) == 352
    assert env.start == (19, 0) and env.goal == (0, 19)
    assert env.action_space == gym.spaces.Discrete(4)
    assert env.observation_space == gym.spaces.Discrete(400)
    assert env.reset(seed=10, options={"position": (0, 19)}) == (380, {"position": (19, 0), "steps": 0})
    check_env(env, skip_render_check=True)


@pytest.mark.parametrize("action", range(4))
def test_exhaustive_kernel_and_every_executed_outcome(action):
    env = IcyGridWorldEnv()
    for state in STATES:
        expected = Counter()
        if state == 19:
            assert env.transition_probabilities(state, action) == {19: 1.0}
            continue
        for index, (direction, probability) in enumerate(
            ((action, 0.8), ((action - 1) % 4, 0.1), ((action + 1) % 4, 0.1))
        ):
            next_state, blocked = oracle_move(state, direction)
            expected[next_state] += probability
            env.reset(seed=1)
            env._position = divmod(state, 20)
            env._np_random = FixedOutcome(index)
            observation, reward, terminated, truncated, info = env.step(action)
            assert observation == next_state
            assert type(reward) is float
            assert reward == (100.0 if next_state == 19 else -1.0)
            assert terminated is (next_state == 19)
            assert truncated is False
            assert info == {"position": divmod(next_state, 20), "steps": 1,
                            "executed_action": direction, "slipped": direction != action,
                            "blocked": blocked}
        actual = env.transition_probabilities(state, action)
        assert actual == pytest.approx(dict(expected))
        assert sum(actual.values()) == pytest.approx(1)
        assert all(p > 0 and s in STATES for s, p in actual.items())


def test_handout_examples_and_expected_reward():
    env = IcyGridWorldEnv()
    assert env.transition_probabilities(380, 1) == {381: 0.8, 360: 0.1, 380: 0.1}
    assert env.transition_probabilities(380, 2) == pytest.approx({380: 0.9, 381: 0.1})
    kernel = env.transition_probabilities(18, 1)
    assert kernel == {19: 0.8, 18: 0.1, 38: 0.1}
    assert sum(p * (100 if s == 19 else -1) for s, p in kernel.items()) == pytest.approx(79.8)


@pytest.mark.parametrize("action", [-1, 4, 1.5, "1", None, [1]])
def test_invalid_actions(action):
    env = IcyGridWorldEnv()
    env.reset(seed=5)
    before = deepcopy(env.np_random.bit_generator.state)
    with pytest.raises(ValueError, match="action"):
        env.step(action)
    with pytest.raises(ValueError, match="action"):
        env.transition_probabilities(19, action)
    assert env._steps == 0 and env.np_random.bit_generator.state == before


@pytest.mark.parametrize("state", [-1, 400, 1.5, "1", None] + [20*r+c for r, c in sorted(ROCKS)])
def test_invalid_states(state):
    with pytest.raises(ValueError):
        IcyGridWorldEnv().transition_probabilities(state, 0)


def test_reproducibility_and_query_purity():
    first, second = IcyGridWorldEnv(), IcyGridWorldEnv()
    first.reset(seed=123)
    second.reset(seed=123)
    for i in range(800):
        before = deepcopy(first.np_random.bit_generator.state)
        first.transition_probabilities(STATES[i % len(STATES)], i % 4)
        assert first.np_random.bit_generator.state == before
        assert first.step(i % 4) == second.step(i % 4)
        if i == 399:
            rng = first.np_random
            before = deepcopy(rng.bit_generator.state)
            assert first.reset(seed=None) == second.reset(seed=None)
            assert first.np_random is rng
            assert first.np_random.bit_generator.state == before


@pytest.mark.parametrize("action", range(4))
@pytest.mark.parametrize("state", [202, 380, 18])
def test_empirical_slips_and_next_state_frequencies(action, state):
    # 12 fixed-seed experiments; 20,000 independent one-step trials each.
    env = IcyGridWorldEnv()
    env.reset(seed=20260913 + state + action)
    directions, destinations = Counter(), Counter()
    for _ in range(20_000):
        env._position = divmod(state, 20)
        env._terminated = False
        observation, _, _, _, info = env.step(action)
        directions[info["executed_action"]] += 1
        destinations[observation] += 1
    assert directions[(action + 2) % 4] == 0
    for direction, p in ((action, .8), ((action-1) % 4, .1), ((action+1) % 4, .1)):
        assert abs(directions[direction] / 20_000 - p) < .015
    expected = env.transition_probabilities(state, action)
    assert set(destinations) == set(expected)
    for destination, p in expected.items():
        assert abs(destinations[destination] / 20_000 - p) < .015


def test_time_limit_and_simultaneous_flags():
    env = gym.make("IcyGridWorld-v0")
    assert env.spec.max_episode_steps == 500
    env.reset(seed=3)
    env.unwrapped._np_random = FixedOutcome(0)
    for step in range(1, 501):
        obs, reward, terminated, truncated, info = env.step(2)
        assert obs == 380 and reward == -1.0 and not terminated
        assert truncated is (step == 500)
        assert info["steps"] == step
    env.reset(seed=3)
    env.unwrapped._np_random = FixedOutcome(0)
    for _ in range(499):
        env.step(2)
    env.unwrapped._position = (0, 18)
    assert env.step(1)[1:4] == (100.0, True, True)
    env.close()


def test_base_has_no_time_limit_and_reset_required():
    env = IcyGridWorldEnv()
    with pytest.raises(ResetNeeded):
        env.step(0)
    env.reset()
    env._np_random = FixedOutcome(0)
    for _ in range(501):
        assert env.step(2)[2:4] == (False, False)
    env._position = (0, 18)
    assert env.step(1)[1:4] == (100.0, True, False)
    with pytest.raises(ResetNeeded):
        env.step(1)
    assert env.reset()[1]["steps"] == 0


def test_rendering(capsys):
    env = IcyGridWorldEnv(render_mode="ansi")
    env.reset()
    assert env.render() == "\n".join(MAP).replace("S", "A")
    env._position = env.goal
    assert env.render() == "\n".join(MAP).replace("S", ".").replace("G", "*")
    human = IcyGridWorldEnv(render_mode="human")
    human.reset()
    assert capsys.readouterr().out.rstrip("\n") == "\n".join(MAP).replace("S", "A")
    assert human.render() is None
    assert capsys.readouterr().out.rstrip("\n") == "\n".join(MAP).replace("S", "A")
    assert IcyGridWorldEnv().render() is None
    with pytest.raises(ValueError):
        IcyGridWorldEnv(render_mode="rgb_array")


def test_numpy_integer_inputs():
    env = IcyGridWorldEnv()
    env.reset(seed=7)
    assert env.transition_probabilities(np.int64(380), np.int64(2)) == pytest.approx({380: .9, 381: .1})
    assert type(env.step(np.int64(0))[0]) is int
