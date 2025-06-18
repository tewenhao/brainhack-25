import gymnasium as gym
from dotworldwrapper import SCGridWorld,ShiftWrapper
from ppodupe import PPO
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
# Parallel environments
env_num = 5
from stable_baselines3.common.vec_env import DummyVecEnv
vec_env = DummyVecEnv([SCGridWorld]*env_num)

model = PPO("MultiInputPolicy", SCGridWorld(), verbose=1,n_steps=1000)
obs = vec_env.reset()
print(obs)
model.learn(total_timesteps=10000000)
model.save("genericallroundedagent1000000adapated2shortn_step1000")

del model # remove to demonstrate saving and loading

model = PPO.load("ppo_cartpole")

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")  