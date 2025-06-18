import time
from collections import deque
from typing import Any, Dict, List, Optional, Type, Union
import datetime
import gym
import numpy as np
import torch as th
from gym.spaces import Box, Discrete
# from stable_baselines3 import PPO
from otherpeopleppo import PPO
from otherpeopleonpolicy import OnPolicyAlgorithm
from otherpeopleactorcritic import ActorCriticPolicy
# from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import (GymEnv, MaybeCallback,
                                                   Schedule)
from stable_baselines3.common.utils import (configure_logger, obs_as_tensor,
                                            safe_mean)
from stable_baselines3.common.vec_env import DummyVecEnv


class DummyGymEnv(gym.Env):
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space


class IndependentPPO(OnPolicyAlgorithm):
    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        num_agents: int,
        env: GymEnv,
        test_env:GymEnv = None,
        learning_rate: Union[float, Schedule] = 1e-4,
        n_steps: int = 1000,
        batch_size: int = 6000,
        n_epochs: int = 10,
        gamma: float = 0.96,
        gae_lambda: float = 0.98,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 40,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        device: Union[th.device, str] = "auto",
        file_save_location:str = ""
    ):
        self.env = env
        self.test_env = test_env
        self.num_agents = num_agents
        self.num_envs = env.num_envs // num_agents
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.n_steps = n_steps
        self.tensorboard_log = tensorboard_log
        self.verbose = verbose
        self.cum_reward = [0]*4
        self._logger = None
        self.file_save_location = file_save_location
        self.best_reward = 0
        self.best_capture = 0
        self.captures = [0]*4
        # env_fn = lambda: DummyGymEnv(self.observation_space, self.action_space)
        # dummy_env = DummyVecEnv([env_fn] * self.num_envs)
        self.policies = [
            PPO(
                policy=policy,
                env=env,
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_range=clip_range,
                clip_range_vf=clip_range_vf,
                ent_coef=ent_coef,
                vf_coef=vf_coef,
                max_grad_norm=max_grad_norm,
                target_kl=target_kl,
                use_sde=use_sde,
                sde_sample_freq=sde_sample_freq,
                policy_kwargs=policy_kwargs,
                verbose=verbose,
                device=device,
                stats_window_size=100
            )
            for _ in range(self.num_agents)
        ]
        self.obs_size = int(self.policies[0].observation_space.shape[0])
        print(self.obs_size)
    def learn(
        self,
        total_timesteps: int,
        callbacks: Optional[List[MaybeCallback]] = None,
        log_interval: int = 1,
        tb_log_name: str = "IndependentPPO",
        reset_num_timesteps: bool = True,
    ):

        num_timesteps = 0
        all_total_timesteps = []
        if not callbacks:
            callbacks = [None] * self.num_agents
        self._logger = configure_logger(
            self.verbose,
            self.tensorboard_log,
            tb_log_name,
            reset_num_timesteps,
        )
        logdir = self.logger.dir
        train_checkpoint = 200000
        # Setup for each policy
        for polid, policy in enumerate(self.policies):
            policy.start_time = time.time()
            if policy.ep_info_buffer is None or reset_num_timesteps:
                policy.ep_info_buffer = deque(maxlen=100)
                policy.ep_success_buffer = deque(maxlen=100)

            if policy.action_noise is not None:
                policy.action_noise.reset()

            if reset_num_timesteps:
                policy.num_timesteps = 0
                policy._episode_num = 0
                all_total_timesteps.append(total_timesteps)
                policy._total_timesteps = total_timesteps
            else:
                # make sure training timestamps are ahead of internal counter
                all_total_timesteps.append(total_timesteps + policy.num_timesteps)
                policy._total_timesteps = total_timesteps + policy.num_timesteps

            policy._logger = configure_logger(
                policy.verbose,
                logdir,
                "policy",
                reset_num_timesteps,
            )

            callbacks[polid] = policy._init_callback(callbacks[polid])

        for callback in callbacks:
            callback.on_training_start(locals(), globals())

        last_obs = self.env.reset()
        for policy in self.policies:
            policy._last_episode_starts = np.ones((self.num_envs,), dtype=bool)

        while num_timesteps < total_timesteps:
            if num_timesteps>train_checkpoint:
                self.save(self.file_save_location+"/"+str(train_checkpoint))
                train_checkpoint+=200000
            last_obs = self.collect_rollouts(last_obs, callbacks)
            num_timesteps += self.num_envs * self.n_steps
            for polid, policy in enumerate(self.policies):
                policy._update_current_progress_remaining(
                    policy.num_timesteps, total_timesteps
                )
                if log_interval is not None and num_timesteps % log_interval == 0:
                    fps = int(policy.num_timesteps / (time.time() - policy.start_time))
                    policy.logger.record("policy_id", polid, exclude="tensorboard")
                    policy.logger.record(
                        "time/iterations", num_timesteps, exclude="tensorboard"
                    )
                    if sum(self.cum_reward)>self.best_reward and sum(self.cum_reward)>100:
                        self.best_reward = sum(self.cum_reward)
                        self.save(self.file_save_location+"/bestreward"+str(self.best_reward))
                    if sum(self.captures)>self.best_capture:
                        self.best_capture = sum(self.captures)
                        for i in range(4):
                            self.captures[i] = 0
                        self.save(self.file_save_location+"/bestcapture"+str(self.best_capture))
                    policy.logger.record(f"polid/total_reward",int(self.cum_reward[polid]))
                    self.cum_reward[polid]= 0
                    if (
                        len(policy.ep_info_buffer) > 0
                        and len(policy.ep_info_buffer[0]) > 0
                    ):
                        policy.logger.record(
                            "rollout/ep_rew_mean",
                            safe_mean(
                                [ep_info["r"] for ep_info in policy.ep_info_buffer]
                            ),
                        )
                        policy.logger.record(
                            "rollout/ep_len_mean",
                            safe_mean(
                                [ep_info["l"] for ep_info in policy.ep_info_buffer]
                            ),
                        )
                    policy.logger.record("time/fps", fps)
                    policy.logger.record(
                        "time/time_elapsed",
                        int(time.time() - policy.start_time),
                        exclude="tensorboard",
                    )
                    policy.logger.record(
                        "time/total_timesteps",
                        policy.num_timesteps,
                        exclude="tensorboard",
                    )
                    policy.logger.dump(step=policy.num_timesteps)
                self.policies[0].rollout_buffer,self.policies[polid].rollout_buffer = self.policies[polid].rollout_buffer,self.policies[0].rollout_buffer,
                self.policies[0].train()
            if self.test_env!=None:
                #Start testing evaluation
                all_episode_rewards = [0]*4
                self.stacks = []
                for i in range(4):
                    self.stacks.append(th.zeros((0,)))
                # self.env.render_mode = "human"
                obs = self.test_env.reset()
                for _ in range(32):
                    
                    for i in range(4):
                        self.add_to_stack(i,obs[i])
                    dones = [0,0,0,0]
                    total_reward = [0]*4

                    while not int(sum(dones)):
                        actions = []
                        for i in range(4):
                            (
                                action,
                                all_values,
                                all_log_probs,
                            ) = self.policies[0].policy.forward(th.unsqueeze(self.stacks[i],0))
                            action = action.numpy()
                            clipped_actions = np.array(
                                [action.item() for action in action]
                            )

                            actions.append(int(clipped_actions[0]))
                        actions = np.array(actions,dtype=np.uint8)
                        actions = np.reshape(actions,shape=(4,))
                        # print(actions)
                        obs, rewards, dones, infos = self.test_env.step(actions)
                        # self.env.render()
                        # prit
                        rewards= rewards.tolist()
                        # print(rewards)
                        for i in range(4):
                            self.captures[i]+=(rewards[i]>=49)
                            total_reward[i] += int(rewards[i])
                    # print(total_reward)
                    for i in range(4):
                        all_episode_rewards[i]+=(total_reward[i])
                for i in range(4):
                    self.policies[i].logger.record(f"polid/eval_reward",int(all_episode_rewards[i]))
            #End testing evaluation
        for callback in callbacks:
            callback.on_training_end()
    def add_to_stack(self,stack_index,obs):
        obs = th.tensor(obs)
        obs = th.flatten(obs,start_dim=0,end_dim=-1)
        self.stacks[stack_index] = th.concat((self.stacks[stack_index],obs))
        self.stacks[stack_index] = self.stacks[stack_index][-self.obs_size:]


    def collect_rollouts(self, last_obs, callbacks):

        all_last_episode_starts = [None] * self.num_agents
        all_obs = [None] * self.num_agents
        all_last_obs = [None] * self.num_agents
        all_rewards = [None] * self.num_agents
        all_dones = [None] * self.num_agents
        all_infos = [None] * self.num_agents
        steps = 0

        for polid, policy in enumerate(self.policies):
            for envid in range(self.num_envs):
                assert (
                    last_obs[envid * self.num_agents + polid] is not None
                ), f"No previous observation was provided for env_{envid}_policy_{polid}"
            all_last_obs[polid] = np.array(
                [
                    last_obs[envid * self.num_agents + polid]
                    for envid in range(self.num_envs)
                ]
            )
            policy.policy.set_training_mode(False)
            policy.rollout_buffer.reset()
            callbacks[polid].on_rollout_start()
            all_last_episode_starts[polid] = policy._last_episode_starts

        while steps < self.n_steps:
            all_actions = [None] * self.num_agents
            all_values = [None] * self.num_agents
            all_log_probs = [None] * self.num_agents
            all_clipped_actions = [None] * self.num_agents
            with th.no_grad():
                for polid, policy in enumerate(self.policies):
                    
                    obs_tensor = obs_as_tensor(all_last_obs[polid], policy.device)
                    # print(obs_tensor)
                    # print(obs_tensor.shape)
                    (
                        all_actions[polid],
                        all_values[polid],
                        all_log_probs[polid],
                    ) = self.policies[0].policy.forward(obs_tensor)
                    clipped_actions = all_actions[polid].cpu().numpy()
                    if isinstance(self.action_space, Box):
                        clipped_actions = np.clip(
                            clipped_actions,
                            self.action_space.low,
                            self.action_space.high,
                        )
                    elif isinstance(self.action_space, Discrete):
                        # get integer from numpy array
                        clipped_actions = np.array(
                            [action.item() for action in clipped_actions]
                        )
                    all_clipped_actions[polid] = clipped_actions

            all_clipped_actions = (
                np.vstack(all_clipped_actions).transpose().reshape(-1)
            )  # reshape as (env, action)
            obs, rewards, dones, infos = self.env.step(all_clipped_actions)

            for polid in range(self.num_agents):
                all_obs[polid] = np.array(
                    [
                        obs[envid * self.num_agents + polid]
                        for envid in range(self.num_envs)
                    ]
                )
                all_rewards[polid] = np.array(
                    [
                        rewards[envid * self.num_agents + polid]
                        for envid in range(self.num_envs)
                    ]
                )
                self.cum_reward[polid]+=sum(all_rewards[polid])
                all_dones[polid] = np.array(
                    [
                        dones[envid * self.num_agents + polid]
                        for envid in range(self.num_envs)
                    ]
                )
                all_infos[polid] = np.array(
                    [
                        infos[envid * self.num_agents + polid]
                        for envid in range(self.num_envs)
                    ]
                )
            
            for policy in self.policies:
                policy.num_timesteps += self.num_envs

            for callback in callbacks:
                callback.update_locals(locals())
            if not [callback.on_step() for callback in callbacks]:
                break

            for polid, policy in enumerate(self.policies):
                policy._update_info_buffer(all_infos[polid])

            steps += 1

            # add data to the rollout buffers
            for polid, policy in enumerate(self.policies):
                if isinstance(self.action_space, Discrete):
                    all_actions[polid] = all_actions[polid].reshape(-1, 1)
                all_actions[polid] = all_actions[polid].cpu().numpy()
                policy.rollout_buffer.add(
                    all_last_obs[polid],
                    all_actions[polid],
                    all_rewards[polid],
                    all_last_episode_starts[polid],
                    all_values[polid],
                    all_log_probs[polid],
                )
            all_last_obs = all_obs
            all_last_episode_starts = all_dones

        with th.no_grad():
            for polid, policy in enumerate(self.policies):
                obs_tensor = obs_as_tensor(all_last_obs[polid], policy.device)
                _, value, _ = self.policies[0].policy.forward(obs_tensor)
                policy.rollout_buffer.compute_returns_and_advantage(
                    last_values=value, dones=all_dones[polid]
                )

        for callback in callbacks:
            callback.on_rollout_end()

        for polid, policy in enumerate(self.policies):
            policy._last_episode_starts = all_last_episode_starts[polid]

        return obs

    def load(self,
        path: str,
        env,
        num_agents: int,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        **kwargs,
    ) -> "IndependentPPO":
        # model = cls(
        #     policy=policy,
        #     num_agents=num_agents,
        #     env=env,
        #     n_steps=n_steps,
        #     policy_kwargs=policy_kwargs,
        #     tensorboard_log=tensorboard_log,
        #     verbose=verbose,
        #     **kwargs,
        # )
        # env_fn = lambda: DummyGymEnv(env.observation_space, env.action_space)
        # dummy_env = DummyVecEnv([env_fn] * (env.num_envs))# // num_agents))
        
        for polid in range(num_agents):
            self.policies[polid] = PPO.load(
                path=path + f"/policy_{polid + 1}/model", env=env, **kwargs,device="cpu"
            )
        return self

    def save(self, path: str) -> None:
        for polid in range(self.num_agents):
            self.policies[polid].save(path=path + f"/policy_{polid + 1}/model")