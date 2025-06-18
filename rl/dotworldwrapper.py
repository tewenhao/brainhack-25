import json
import os
from stable_baselines3 import PPO
import requests
from dotenv import load_dotenv
from til_environment import gridworld
from til_environment.types import Action, Direction, Player, RewardNames, Tile, Wall
from scutils import adapt_obs,split_obs
import gymnasium as gym
import numpy as np
import gymnasium as gym
import copy
from pettingzoo.utils.conversions import parallel_wrapper_fn
from stable_baselines3.common.env_checker import check_env
from arguments import get_common_args
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

    def __init__(self,args=get_common_args()):
        # The size of the square grid
        self.grid_env = gridworld.env(env_wrappers=[], render_mode=None, novice=False)
        self.adapted_obs = args.adapted_obs
        self.guard_distance_reward = args.guard_distance_reward
        self.args = args
        if self.adapted_obs:
            self.observation_space = gym.spaces.Dict(
                {
                    "wall": gym.spaces.Box(
                        0,
                        2**8 - 1,
                        shape=(
                            self.grid_env.viewcone_length,
                            self.grid_env.viewcone_width,
                        ),
                        dtype=np.uint8,
                    ),
                    "guard_pos": gym.spaces.Box(
                        0,
                        2**8 - 1,
                        shape=(
                            self.grid_env.viewcone_length,
                            self.grid_env.viewcone_width,
                        ),
                        dtype=np.uint8,
                    ),
                    "scout_pos": gym.spaces.Box(
                        0,
                        2**8 - 1,
                        shape=(
                            self.grid_env.viewcone_length,
                            self.grid_env.viewcone_width,
                        ),
                        dtype=np.uint8,
                    ),
                    "points": gym.spaces.Box(
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
        else:
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
        self.prev_location = [None]*4
        self.agent_index = 0
        self.current_location = []
    def change_reward(self,observation,reward,info):
        if not self.is_scout(observation):
            # print(self.prev_location,self.agent_index)
            if self.prev_location[self.agent_index] == None:
                self.agent_index = self.agent_index
            else:
                old_distance = self.get_distance(self.prev_location[0],self.prev_location[self.agent_index])
                new_distance = self.get_distance(self.current_location[0],self.current_location[self.agent_index])
                reward+= (old_distance-new_distance)
        return reward
    def get_distance(self,location1,location2):
        return abs(location1[0]-location2[0])+abs(location1[1]-location2[1])
    def _get_obs(self):
        observation, reward, termination, truncation, info = self.grid_env.last()
        # print(type(observation["viewcone"]))
        
        if self.adapted_obs:
            observation = self.adapt_observation(observation)
        return observation
    def reset(self, seed = None, options = None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self.grid_env.reset()
        # print("reset",observation)
        observation = self._get_obs()
        info = 0#self._get_info()
        self.prev_location = [None]*4
        self.agent_index = 1
        self.current_location = [self.get_location(observation)]
        # print(self.grid_env.get_info(self.grid_env.agent_selection))
        # print(observation)
        # if self.adapted_obs:
        #     observation = self.adapt_observation(observation)
        return observation, self.grid_env.get_info(self.grid_env.agent_selection)
    def adapt_observation(self,observation):
        channels = split_obs(observation["viewcone"])
        observation["wall"] = channels[0]
        observation["guard_pos"] =channels[1]
        observation["scout_pos"] = channels[2]
        observation["points"] = channels[3]
        del observation["viewcone"]
        return observation
    def step(self, action:int):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        self.grid_env.step(action)
        observation, reward, termination, truncation, info = self.grid_env.last()
        self.current_location.append(self.get_location(observation))
        if self.adapted_obs:
            observation = self.adapt_observation(observation)
        reward = self.change_reward(observation,reward,info)


        self.agent_index+=1
        self.agent_index = self.agent_index%4
        if self.agent_index == 0:
            self.prev_location = copy.copy(self.current_location)
            self.current_location = []
        
        return observation, reward, termination, truncation, info
    def is_scout(self,observation):
        return observation["scout"] == 1
    def get_location(self,observation):
        return np.ndarray.tolist(observation["location"])
def main():
    temp = SCGridWorld()
    temp.reset()
    print(temp.grid_env.agent_locations)
    parallel = parallel_wrapper_fn(SCGridWorld)()



if __name__ == "__main__":
    main()
