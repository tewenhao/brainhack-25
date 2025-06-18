import gymnasium as gym

from ppodupe import PPO
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
# Parallel environments
vec_env = make_vec_env("Adventure", n_envs=4)

model = PPO("MlpPolicy", vec_env, verbose=1)
obs = vec_env.reset()
print(obs)
model.learn(total_timesteps=25000)
model.save("ppo_cartpole")

del model # remove to demonstrate saving and loading

model = PPO.load("ppo_cartpole")

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")