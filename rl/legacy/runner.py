import numpy as np
from dotworldwrapper import SCGridWorld,ShiftWrapper
# from stable_baselines3.common.env_util import make_vec_env
from arguments import get_common_args
from scutils import extract_obs , env_that_ended
from marlppo import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
class Runner():
    def __init__(self,args = get_common_args()):
        self.env_num = args.env_num
        self.env_array = []
        self.args = args
        self.total_epoch_num = args.epoch_num
        self.agent = []
        self.env_array = DummyVecEnv([SCGridWorld]*self.env_num)
        self.agent= PPO(env = SCGridWorld(),policy = "MultiInputPolicy",
                        device="cuda",n_steps=50,
                        tensorboard_log="./oldgridworld/")
    def run(self):
        
        checkpoint_callback = CheckpointCallback(save_freq=1e6 , save_path='./model_checkpoints/')
        self.agent = self.agent.learn(total_timesteps= 500000,progress_bar=True,callback=[checkpoint_callback])
        self.agent.save("donotuse.zip")

        # self.agent[0].learn(total_timesteps=10000)
        # environment_indexes = 0
        # print(self.env_array.envs[0].reset())
        # # print(self.env_array.step(np.ones((10),dtype=np.uint8)))
        # for epoch_num in range(self.total_epoch_num):
        #     for step in range(100):
        #         for agent_no in range(4):
        #             action = []
        #             action = self.agent[agent_no].predict(obs)
                    
        #             # for environment_n in range(self.env_num):
        #             #     action.append(self.env_array.envs[environment_n].action_space.sample())    
        #             # print(action[0])
        #             self.env_array.step_async(action[0])
        #             observation, reward, done, information =self.env_array.step_wait()
        #             timelimit = 0
                    # for environment_n in range(self.env_num):
                    #     timelimit +=(information[environment_n][    "TimeLimit.truncated"])
                    # if 1 in done and timelimit <5 :
                    #     print(epoch_num,step)
                    #     print(observation)
                    #     print(reward)
                    #     print(done)
                    #     print(information)
                    # for i in range(self.env_num):
                    #     self.env_agent_responsible[i] = (1+self.env_agent_responsible[i])%4
                    # if env_that_ended(information):
                    #     print(env_that_ended(information),self.env_agent_responsible)
                    #     for i in env_that_ended(information):
                    #         self.env_agent_responsible[i] = 0
                        
                    # for i in env_that_ended(information):
                    #     print(information[i])
                    # k+=1
        # print(obs)
        # print(obs["direction"])
        # print(extract_obs(obs,[2,4])["viewcone"])
        # print(obs["viewcone"])
        # for i in range(self.env_num):
        #     self.env_array[i].append(CGridWorld())
        #     self.env_array[i].reset()
        # obs = vec_env.reset()
        # while True:
        #     action, _states = model.predict(obs)
        #     obs, rewards, dones, info = vec_env.step(action)
        #     vec_env.render("human")

if __name__ == "__main__":
    temp = Runner()
    temp.run()