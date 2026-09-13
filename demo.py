"""Run reproducible episodes using random actions or an exact-model policy."""

import argparse

import gymnasium as gym
import numpy as np

import icy_gridworld  # Registers the environment.


def value_iteration(env, gamma=0.99, tolerance=1e-10):
    """Solve the infinite-horizon discounted MDP, outside the 500-step wrapper."""
    states = [s for s in range(400) if divmod(s, 20) not in env.obstacles]
    successors = np.zeros((400, 4, 3), dtype=int)
    probabilities = np.zeros((400, 4, 3))
    rewards = np.zeros((400, 4, 3))
    for state in states:
        for action in range(4):
            for index, (next_state, p) in enumerate(env.transition_probabilities(state, action).items()):
                successors[state, action, index] = next_state
                probabilities[state, action, index] = p
                rewards[state, action, index] = 0 if state == 19 else (100 if next_state == 19 else -1)
    values = np.zeros(400)
    for _ in range(100_000):
        q_values = (probabilities * (rewards + gamma * values[successors])).sum(axis=2)
        updated = q_values.max(axis=1)
        if np.max(np.abs(updated - values)) < tolerance:
            return q_values.argmax(axis=1)
        values = updated
    raise RuntimeError("Value iteration failed to converge")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--policy", choices=("random", "optimal"), default="optimal")
    parser.add_argument("--render", choices=("none", "ansi", "human"), default="ansi")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    mode = None if args.render == "none" else args.render
    env = gym.make("IcyGridWorld-v0", render_mode=mode)
    try:
        policy = value_iteration(env.unwrapped) if args.policy == "optimal" else None
        env.action_space.seed(args.seed)
        for episode in range(args.episodes):
            observation, _ = env.reset(seed=args.seed if episode == 0 else None)
            total_reward = 0.0
            while True:
                action = env.action_space.sample() if policy is None else int(policy[observation])
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break
            if mode == "ansi":
                print(env.render())
            print(f"Episode {episode + 1}: steps={info['steps']}, reward={total_reward:.1f}, "
                  f"terminated={terminated}, truncated={truncated}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
