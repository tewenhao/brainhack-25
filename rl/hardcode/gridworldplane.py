
# import ..gridworldaec,.flatten_dict
from gridworldaec import env as environmentCreator
from flatten_dict import FlattenDictWrapper
from pettingzoo.utils.conversions import aec_to_parallel
from supersuit import frame_stack_v1, flatten_v0, pad_observations_v0
import gymnasium
import numpy as np
import time
import copy
ONE_FRAME = [  1,   1,   1,   1,  16,  16,   1,   1, 100,   1,   1,   1,   1,   1,
   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
   1,   1,   1,   1,   1,   1,   1,   1,   1,]

class gridworldplaneEnv(gymnasium.Env):

    def __init__(self, verbose = False, manual = False,frame_stack_num = 8,flatten = True,render_mode = "human",novice= False):
        super(gridworldplaneEnv).__init__()
        self.possible_agents = ["player_0","player_1","player_2","player_3"]
        self.flatten = flatten
        self.frame_stack_num = frame_stack_num
        if flatten:
            aec_env = (environmentCreator(novice = novice,env_wrappers = [FlattenDictWrapper,],debug=True,render_mode=render_mode))
            aec_env = frame_stack_v1(aec_env,frame_stack_num)
            aec_env = pad_observations_v0(aec_env)
            # print("new obs automatic",aec_env.observation_space("player_1"))
            self.env = aec_env
            self.env.reset()
            ind_obs_size =self.env.last()[0].shape
            print(ind_obs_size)
            # self.observation_space = gymnasium.spaces.Box(1,255,shape=ind_obs_size,dtype=np.uint8)
            # self.observation_space = aec_env.observation_space("player_1")
            just_obs = np.tile(np.array(ONE_FRAME),8).flatten()
            action_space = np.ones((5,))
            total_obs_space = np.concatenate((just_obs,action_space))
            self.observation_space =gymnasium.spaces.Box(low=0,high=total_obs_space,dtype=np.uint8)
            self.action_space = gymnasium.spaces.Discrete(5)
            
        else:
            aec_env = (environmentCreator(novice = False,env_wrappers = [],debug=True,render_mode=render_mode))#flatten_dict.FlattenDictWrapper
            self.env = aec_env
        self.n_players = 4
        
        self.verbose = verbose
        self.manual = manual
        self.name="gridworldplane"
        self.current_player_num = 0

    def reset(self, *, seed=None, options=None):
        self.agents = self.possible_agents
        self.current_player_num = 0
        self.stored_obs = [None] * self.n_players
        self.env.reset()
        last = self.env.last()
        self.stored_obs[0] = last[0]
        self.episode_rewards = 0
        # _ = 
        return self.observation,None
    # def observe
    def test(self):
        self.env.reset()
        # self.env._cumulative_rewards[self.env.scout] = 1.5
        for i in range(20):
            agent = self.env.agent_selection
            print(agent)
            print(self.env.observe(agent))
            self.env.step(0)
            print(self.env.observe(agent),"\nReward is",self.env._cumulative_rewards[agent])
            # print("Scout Reward is",self.env._cumulative_rewards[self.env.scout])
            # print(self.env.last_agent(agent))
            self.env._cumulative_rewards[agent] = 0 
            time.sleep(1)
            print("_"*20)
    def agent_to_num(self,agent):
        return int(agent[-1])
    def step(self, action):
        # return observation dict, rewards dict, termination/truncation dicts, and infos dict
        # return {"agent_1": [obs of agent_1]}, {...}, ...
        reward = np.zeros(shape=(self.n_players,))#[0]*self.n_players
        # print('what')
        current_agent = self.env.agent_selection
        _ = self.env.step(action)
        # print(_)
        # if self.env._cumulative_rewards.get(self.agent_to_num(current_agent),0)!= 0 :
        #     print(self.env._cumulative_rewards.get(self.agent_to_num(current_agent),0))
        reward[self.agent_to_num(current_agent)] = self.env._cumulative_rewards.get(self.agent_to_num(current_agent),0)
        # reward[0] = self.env._cumulative_rewards.get(self.agent_to_num(current_agent),0)
        observation,cum_reward,termination,truncation,info = self.env.last_agent(current_agent)
        last = self.env.last()
        next_obs = last[0]
        # print(self.env.last())
        # print("Current agent",observation)
        # next_agent = self.env.agent_selection
        # next_obs = self.env.observations[next_agent]
        # print("Next agent2",next_obs)
        for key, value in self.env._cumulative_rewards.items():
            reward[self.agent_to_num(key)] += value
            # reward[0] += value
        # if sum(reward)>0:
            # print(reward)
        self.env._cumulative_rewards[current_agent] = 0 
        self.current_player_num = (self.current_player_num +1)%4
        self.stored_obs[self.current_player_num] = next_obs
        info = {}
        self.episode_rewards += sum(reward)
        info[0] = {"episode":{"r":self.episode_rewards,"l":self.env.num_moves}}
        if self.flatten:
                return np.concatenate((self.stored_obs[self.current_player_num],self.legal_actions))
        else:
            return self.stored_obs[self.current_player_num]
    
    @property
    def observation(self):
        if self.flatten:
            return np.concatenate((self.stored_obs[self.current_player_num],self.legal_actions))
        else:
            return self.stored_obs[self.current_player_num]
    @property
    def legal_actions(self):
        actions = np.ones(shape=(5,))
        actions[-1] = 0
        viewcone = self.env.observe_bypass(f"player_{self.current_player_num}")["viewcone"]

        if self.flatten:
            # print(viewcone)
            viewcone = viewcone[-35:]
            
            viewcone = np.reshape(viewcone,shape=(7,5))
        else:
            viewcone = viewcone
        # print(viewcone)
        # print("raw",self.env.observe_bypass(f"player_{self.current_player_num}"))
        # print(self.observation,self.env.agent_selection)
        cell = int(viewcone[2][2])
        #remember, actions are 0 move forward 1 back, 2 left, 3 right, 4 stay
        #we dont controll how they turn as long as they dont move into walls
        if (cell>>5)&1:actions[1] = 0
        if (cell>>7)&1:actions[0] = 0
        # print(cell)
        # print(actions)
        return actions
    def render(self):
        pass
if __name__ == "__main__":
    test = GridworldSimpleEnv()
    # print(test.env.reset()[0]["player_0"].shape)
    # print(test.env.reset())
    # _ = test.step(0)#{"player_0":0,"player_1":0,"player_2":0,"player_3":0})
    # print(_)
    time.sleep(3)
    for i in range(20):
        _=test.step(0)
        # print(_)
        for i in _:
            print(i)
        print("_"*20)
        time.sleep(1)
    time.sleep(180)
