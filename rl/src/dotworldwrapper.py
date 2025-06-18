import json
import os
from stable_baselines3 import PPO
import requests
from dotenv import load_dotenv
from til_environment import gridworld
from til_environment.types import Action, Direction, Player, RewardNames, Tile, Wall
import gymnasium as gym
import numpy as np
import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
load_dotenv()
class ShiftWrapper(gym.Wrapper):
    """Allow to use Discrete() action spaces with start!=0"""
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.Discrete)
        self.action_space = gym.spaces.Discrete(env.action_space.n, start=0)

    def step(self, action: int):
        return self.env.step(action + self.env.action_space.start)
class SCGridWorld(gym.Env):

    def __init__(self):
        # The size of the square grid
        self.grid_env = gridworld.env(env_wrappers=[], render_mode=None, novice=False)
        self.observation_space = gym.spaces.Dict(
            {
                "viewcone": gym.spaces.Box(
                    0,
                    2**8 - 1,
                    shape=(
                        self.grid_env.viewcone_length,
                        self.grid_env.viewcone_width,
                    ),
                    dtype=np.uint8,
                ),
                "direction": gym.spaces.Discrete(len(Direction)),
                "scout": gym.spaces.Discrete(2),
                "location": gym.spaces.Box(0, self.grid_env.size, shape=(2,), dtype=np.uint8),
                "step": gym.spaces.Discrete(405),
            }
        )
        self.action_space = self.grid_env.action_space(0)
    def _get_obs(self):
        observation, reward, termination, truncation, info = self.grid_env.last()
        return observation
    def reset(self, seed = None, options = None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self.grid_env.reset()

        observation = self._get_obs()
        info = 0#self._get_info()
        
        observation["direction"] = np.uint8(observation["direction"])
        observation["location"] = np.uint8(observation["location"])#np.asarray(observation["location"],dtype=np.uint8)
        # print("bibbles",observation)
        return observation, self.grid_env.get_info(self.grid_env.agent_selection)
    def step(self, action:int):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        self.grid_env.step(action)
        observation, reward, termination, truncation, info = self.grid_env.last()
        observation["direction"] = np.uint8(observation["direction"])
        observation["location"] = np.uint8(observation["location"])
        return observation, reward, termination, truncation, info
def main():
    env_array = []
    # print(check_env(SCGridWorld()))
    for i in range(10):
        env_array.append(ShiftWrapper(SCGridWorld()))
        env_array[i].reset()
    # print(env_array[0]._get_obs())
    # print(env_array[1]._get_obs())
    # print(env_array[02]._get_obs())
    # be the agent at index 0
    model1 = PPO(env = ShiftWrapper(SCGridWorld()),policy = "MultiInputPolicy")
    # for step_number in range(50):
    #     for env_number in range(10):
            
            
    #         for agent in env_array[env_number].agent_iter():
    #             # print(agent)
    #             observation, reward, termination, truncation, info = env_array[env_number].last()
    #             if termination or truncation:
    #                 action = None
    #                 env_array[env_number].step(action)
    #                 continue
    #             action = env_array[env_number].action_space(agent).sample()
    #             # print(action)
    #             env_array[env_number].step(action)
    #         # k+=1



if __name__ == "__main__":
    main()
