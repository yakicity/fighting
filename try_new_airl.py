# gymのインポート
import gymnasium as gym
# pandasのインポート
import pandas as pd
# matplotlibのインポート
import matplotlib.pyplot  as plt
import pygame
import sys
from collections import deque
import random
import numpy as np
import gymnasium as gym
import joblib
import math
from env_new_4 import MyEnv
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal

import wandb
from stable_baselines3 import PPO

from imitation.algorithms.adversarial.airl import AIRL
from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.util.networks import RunningNorm
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
import pickle
from collections import namedtuple
from imitation.data import rollout
from imitation.algorithms import bc
from imitation.data.types import Trajectory,Transitions
from imitation.data.wrappers import RolloutInfoWrapper

import datetime
import functools
import itertools
import os
import pathlib
import uuid
import warnings
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    overload,
)

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3.common import monitor, policies
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from imitation.util.util import make_seeds
if __name__ == "__main__":

    # # start a new wandb run to track this script
    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="fight-project_airl",
    #     name='ppo_learner_model_2',

    #     # # track hyperparameters and run metadata
    #     # config={
    #     # "epochs": 10,
    #     # }
    # )

    with open("transitions_final_1.pickle", "rb") as f:
        trajectories_1 = pickle.load(f)
    transitions_1 = rollout.flatten_trajectories(trajectories_1)
    with open("transitions_final_2.pickle", "rb") as f:
        trajectories_2 = pickle.load(f)
    transitions_2 = rollout.flatten_trajectories(trajectories_2)
    with open("transitions_final_3.pickle", "rb") as f:
        trajectories_3 = pickle.load(f)
    transitions_3= rollout.flatten_trajectories(trajectories_3)
    with open("transitions_final_4.pickle", "rb") as f:
        trajectories_4 = pickle.load(f)
    transitions_4= rollout.flatten_trajectories(trajectories_4)
    with open("transitions_final_5.pickle", "rb") as f:
        trajectories_5 = pickle.load(f)
    transitions_5= rollout.flatten_trajectories(trajectories_5)
    transitions = transitions_1 + transitions_2 + transitions_3 + transitions_4 + transitions_5

    filtered_transitions_6 = [t for t in transitions if t['acts'] == 6]

    ideal_len_ts = len(filtered_transitions_6)
    print(ideal_len_ts)
    balanced_transitions_list = []
    for i in range(7):
        now_filtered_transitions = [t for t in transitions if t['acts'] == i]
        # if len(now_filtered_transitions) > ideal_len_ts:
        #     sample_num = min(ideal_len_ts * 2, len(now_filtered_transitions))
        #     now_filtered_transitions = random.sample(now_filtered_transitions, sample_num)
        balanced_transitions_list += now_filtered_transitions

    # フィルタリングで得たサンプル
    print(len([t for t in balanced_transitions_list if t['acts'] == 0]))
    print(len([t for t in balanced_transitions_list if t['acts'] == 1]))
    print(len([t for t in balanced_transitions_list if t['acts'] == 2]))
    print(len([t for t in balanced_transitions_list if t['acts'] == 3]))
    print(len([t for t in balanced_transitions_list if t['acts'] == 4]))
    print(len([t for t in balanced_transitions_list if t['acts'] == 5]))
    print(len([t for t in balanced_transitions_list if t['acts'] == 6]))
    balanced_transitions = Transitions(
        obs=np.array([t["obs"] for t in balanced_transitions_list]),
        acts=np.array([t["acts"] for t in balanced_transitions_list]),
        infos=np.array([t["infos"] for t in balanced_transitions_list], dtype=object),
        next_obs=np.array([t["next_obs"] for t in balanced_transitions_list]),
        dones=np.array([t["dones"] for t in balanced_transitions_list])
        # terminal=True
    )
    # balanced_transitions = rollout.flatten_trajectories(balanced_transitions)

    # print(balanced_transitions)
    print(len(balanced_transitions))



    def make_vec_env(
        env_name: str,
        *,
        rng: np.random.Generator,
        n_envs: int = 8,
        parallel: bool = False,
        log_dir: Optional[str] = None,
        max_episode_steps: Optional[int] = None,
        post_wrappers: Optional[Sequence[Callable[[gym.Env, int], gym.Env]]] = None,
        env_make_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> VecEnv:
        """Makes a vectorized environment.

        Args:
            env_name: The Env's string id in Gym.
            rng: The random state to use to seed the environment.
            n_envs: The number of duplicate environments.
            parallel: If True, uses SubprocVecEnv; otherwise, DummyVecEnv.
            log_dir: If specified, saves Monitor output to this directory.
            max_episode_steps: If specified, wraps each env in a TimeLimit wrapper
                with this episode length. If not specified and `max_episode_steps`
                exists for this `env_name` in the Gym registry, uses the registry
                `max_episode_steps` for every TimeLimit wrapper (this automatic
                wrapper is the default behavior when calling `gym.make`). Otherwise
                the environments are passed into the VecEnv unwrapped.
            post_wrappers: If specified, iteratively wraps each environment with each
                of the wrappers specified in the sequence. The argument should be a Callable
                accepting two arguments, the Env to be wrapped and the environment index,
                and returning the wrapped Env.
            env_make_kwargs: The kwargs passed to `spec.make`.

        Returns:
            A VecEnv initialized with `n_envs` environments.
        """
        # Resolve the spec outside of the subprocess first, so that it is available to
        # subprocesses running `make_env` via automatic pickling.
        # Just to ensure packages are imported and spec is properly resolved
        # tmp_env = gym.make(env_name)
        tmp_env = MyEnv(render_mode="human")
        tmp_env.close()
        spec = tmp_env.spec
        env_make_kwargs = env_make_kwargs or {}

        def make_env(i: int, this_seed: int) -> gym.Env:
            # Previously, we directly called `gym.make(env_name)`, but running
            # `imitation.scripts.train_adversarial` within `imitation.scripts.parallel`
            # created a weird interaction between Gym and Ray -- `gym.make` would fail
            # inside this function for any of our custom environment unless those
            # environments were also `gym.register()`ed inside `make_env`. Even
            # registering the custom environment in the scope of `make_vec_env` didn't
            # work. For more discussion and hypotheses on this issue see PR #160:
            # https://github.com/HumanCompatibleAI/imitation/pull/160.
            # assert env_make_kwargs is not None  # Note: to satisfy mypy
            # assert spec is not None  # Note: to satisfy mypy
            # env = gym.make(spec, max_episode_steps=max_episode_steps, **env_make_kwargs)
            env = MyEnv(render_mode="human")

            # Seed each environment with a different, non-sequential seed for diversity
            # (even if caller is passing us sequentially-assigned base seeds). int() is
            # necessary to work around gym bug where it chokes on numpy int64s.
            env.reset(seed=int(this_seed))
            # NOTE: we do it here rather than on the final VecEnv, because
            # that would set the same seed for all the environments.

            # Use Monitor to record statistics needed for Baselines algorithms logging
            # Optionally, save to disk
            log_path = None
            if log_dir is not None:
                log_subdir = os.path.join(log_dir, "monitor")
                os.makedirs(log_subdir, exist_ok=True)
                log_path = os.path.join(log_subdir, f"mon{i:03d}")

            env = monitor.Monitor(env, log_path)

            if post_wrappers:
                for wrapper in post_wrappers:
                    env = wrapper(env, i)

            return env

        env_seeds = make_seeds(rng, n_envs)
        env_fns: List[Callable[[], gym.Env]] = [
            functools.partial(make_env, i, s) for i, s in enumerate(env_seeds)
        ]
        if parallel:
            # See GH hill-a/stable-baselines issue #217
            return SubprocVecEnv(env_fns, start_method="forkserver")
        else:
            return DummyVecEnv(env_fns)

    SEED = 42
    FAST = False

    if FAST:
        N_RL_TRAIN_STEPS = 100_000
    else:
        # N_RL_TRAIN_STEPS = 2_000_000
        N_RL_TRAIN_STEPS = 300_000

    env = make_vec_env(
        "seals:seals/CartPole-v0",
        rng=np.random.default_rng(SEED),
        n_envs=8,
        post_wrappers=[
            lambda env, _: RolloutInfoWrapper(env)
        ],  # needed for computing rollouts later
    )

    # env = MyEnv(render_mode="human")


    learner = PPO(
        env=env,
        policy=MlpPolicy,
        batch_size=64,
        ent_coef=0.0,
        learning_rate=0.0005,
        gamma=0.95,
        clip_range=0.1,
        vf_coef=0.1,
        n_epochs=5,
        seed=SEED,
    )
    reward_net = BasicShapedRewardNet(
        observation_space=env.observation_space,
        action_space=env.action_space,
        normalize_input_layer=RunningNorm,
    )
    airl_trainer = AIRL(
        demonstrations=balanced_transitions,
        demo_batch_size=2048,
        gen_replay_buffer_capacity=512,
        n_disc_updates_per_round=16,
        venv=env,
        gen_algo=learner,
        reward_net=reward_net,
        allow_variable_horizon=True,  # ここでエラーを回避するための設定を追加
    )
    # print(transitions_2[0])
    # print(transitions_2)
    airl_trainer.train(N_RL_TRAIN_STEPS)
    airl_trainer.gen_algo.save("airl_trainer_gen_algo")

    # model_path = "ppo_learner_model_3"
    # enemy_model = PPO.load(model_path, env=env)
    # env = MyEnv(render_mode="human",enemy_model= enemy_model)
    # モデルの準備
    # model = PPO('MlpPolicy', env, verbose=1)
    # model = PPO.load("ppo_learner_model_3", env=env)
    # model.learn(total_timesteps=100000)
    # model.learn(total_timesteps=10000)
    # モデルの保存
    learner.save(f"ppo_learner_model_{N_RL_TRAIN_STEPS}_sonly")
    torch.save(reward_net.state_dict(), f"airl_reward_net_{N_RL_TRAIN_STEPS}_sonly.pth")
    # model.save(f"ppo_learner_model_second")
    # model.save(f"ppo_learner_model_second_1")

    num_episodes = 1

    player_info = ["cpu","random"]

    env = MyEnv(render_mode="human")
    """メインループ"""
    for episode in range(1,num_episodes+1):
        if episode % 2 == 0:
            meind = 1
        else:
            meind = 2
        state,_ = env.reset() # observerを初期化し、前処理済みの初期状態を返す
        done = False # エピソードの終了フラグ
        reward_per_episode = 0 # 1エピソード当たりの報酬の総和
        t = 0
        while (not done): # エピソードが終了しない間はずっと処理を行う

            t += 1
            action, _ = learner.predict(state, deterministic=True)
            if player_info[1] == "random":
                # enemy_action = np.random.uniform(-1,1)
                # enemy_action = np.random.choice([0,1,2,3,4,5,6], 1)[0]
                enemy_action = np.random.choice([2,3,6], 1)[0]
            # elif player_info[1] == "cpu":
                # ここにstateのmeとenamyいれかえたstate[0]をもとめる処理角
                # enemy_state = [state[0][0],state[0][1],state[0][4],state[0][5],state[0][2],state[0][3],state[0][0],state[0][8],state[0][7],state[0][10],state[0][9]]
                # enemy_state = np.array(enemy_state, dtype=np.float32)
                # enemy_action = algo_enemy.exploit(enemy_state)[0]

            state, reward, done, _,_ = env.step_eval([action.item(),enemy_action])

            env.render()
            # if t >= 100:
            #     break

    # ④ wandbのrunを終了
    env.close()
    # wandb.finish()

