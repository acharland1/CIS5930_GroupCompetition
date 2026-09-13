"""Fixed Icy Grid World; importing this package registers IcyGridWorld-v0."""

from gymnasium.envs.registration import register

from icy_gridworld.env import IcyGridWorldEnv

register(
    id="IcyGridWorld-v0",
    entry_point="icy_gridworld.env:IcyGridWorldEnv",
    max_episode_steps=500,
)

__all__ = ["IcyGridWorldEnv"]
