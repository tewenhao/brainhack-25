"""Manages the RL model."""
import numpy as np
import time
from dotworldwrappereval import SCGridWorldEval
# from dotworldwrapper import SCGridWorld
# from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
# from scutils import extract_obs , env_that_ended
# from marlppo import PPO
from stable_baselines3 import PPO as PPOoriginal
from ppodupe import PPO
from scutils import split_obs
from gymnasium.spaces.utils import flatten, flatten_space
import supersuit as ss
from gymnasium.spaces import Box, Dict, Discrete
from Otherpeoplescode import IndependentPPO
import gridworldresponsive
import torch
from hardcode import Scout

class RLManager:

    def __init__(self,agent_index=0):
        stack_size = 16
        self.env = gridworldresponsive.env(env_wrappers=[],render_mode="human",novice= False)
        
    def rl(self, observation: dict[str, int | list[int]]) -> int:
        """Gets the next action for the agent, based on the observation.

        Args:
            observation: The observation from the environment. See
                `rl/README.md` for the format.

        Returns:
            An integer representing the action to take. See `rl/README.md` for
            the options.
        """
        if observation["scout"]:
            agent_index=0
        else:
            agent_index = 1
        agent_index = 0
        observation = torch.tensor(flatten(self.observation_space,observation))
        # observation = torch.unsqueeze(observation,0)
        print(observation)
        self.stack = torch.concat([self.stack,observation],dim=0)
        self.stack = self.stack[-2304:]
        (
            all_actions,
            all_values,
            all_log_probs,
        ) = self.agent.policies[agent_index].policy.forward(torch.unsqueeze(self.stack,0))
        # print(all_actions,all_values)
        all_actions = all_actions.numpy()
        clipped_actions = np.array(
            [action.item() for action in all_actions]
        )
        # print(clipped_actions)
        return int(clipped_actions[0])

    def evaluate_policy_total_reward(self, n_episodes=10):
        """
        Runs the PPO policy without training or collecting rollouts.
        Returns a list of total rewards per episode.
        """
        all_episode_rewards = []
        for i in range(4):
            all_episode_rewards.append([])
        self.env.render_mode = "human"
        scout = Scout()
        guard = Scout(is_scout=False)
        for _ in range(n_episodes):
            terminated = False
            truncated=  False
            self.env.reset()
            self.agents = []
            self.agents.append(Scout())
            for i in range(3):
                self.agents.append(Scout(is_scout=False))
            while not terminated:
                # print(obs)
                
                if terminated or truncated:
                    self.env.reset()
                    continue
                guard_index = 1
                for i in range(4):
                    obs,reward,terminated,truncated,info  = self.env.last()
                    
                    if obs["scout"]:
                        self.env.step(self.agents[0].get_action_scout(obs))
                    else:
                        # print(f"Guard{guard_index}")
                        self.env.step(self.agents[guard_index].get_action_guard(obs))
                        guard_index+=1
                    obs,reward,terminated,truncated,info  = self.env.previous_last()
                    print(self.env.agent_selection,reward,end="\t")
                print()
                
                time.sleep(1)
            
        return all_episode_rewards
if __name__ == "__main__":
    temp = RLManager()
    print(temp.evaluate_policy_total_reward(1))