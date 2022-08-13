import sys

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common import make_vec_env, set_global_seeds
from stable_baselines.bench import Monitor

from stable_baselines import SAC, PPO2, DQN, A2C

from utils.callbacks import getBestRewardCallback, logDir
from utils.arg_parser import common_arg_parser, parseLayersFromArgs

from env import Manipulator2D

import os
import tensorflow as tf

def main(args):
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)

    env = Manipulator2D()

    log_dir = logDir()
    if not os.path.isdir(log_dir + args.prefix):
        os.makedirs(log_dir + args.prefix, exist_ok=True)

    env = Monitor(env, logDir()+args.prefix+"/log", allow_early_resets=True)
    layers = parseLayersFromArgs(args=args) # default [32, 32]

    bestRewardCallback = getBestRewardCallback(args)

    policy_kwargs = dict(layers=layers)

    from stable_baselines.sac.policies import MlpPolicy
    model = SAC(MlpPolicy, env, verbose=1, policy_kwargs=policy_kwargs)   

    model.learn(total_timesteps=100000, log_interval=1, callback=bestRewardCallback)

if __name__ == '__main__':
    main(sys.argv)

