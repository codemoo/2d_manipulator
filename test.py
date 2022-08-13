import sys

# from baselines.common.cmd_util import common_arg_parser, parse_unknown_args

from stable_baselines.common.policies import MlpLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.bench import Monitor

from stable_baselines import SAC, PPO2

from utils.callbacks import getBestRewardCallback, logDir
from utils.arg_parser import common_arg_parser, parseLayersFromArgs

from env import Manipulator2D

import numpy as np
import os
from glob import glob
import tensorflow as tf

import csv
import re

import json

from PyInquirer import prompt, print_json
from argparse import Namespace

def main(args):
    env = Manipulator2D()

    layers = parseLayersFromArgs(args=args) # default [32, 32]
    policy_kwargs = dict(layers=layers)

    if args.alg == "sac":
        model = SAC.load(args.model_path)

    test_runs = 1

    for i in range(test_runs):
        obs = env.reset()

        print(i+1,"/",test_runs)
        while True:
            
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)

            if dones == True:
                break


    env.render()

     
if __name__ == '__main__':
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(sys.argv)

    log_files = sorted(glob(logDir()+"/*"))

    questions = [
        {
            'type': 'list',
            'name': 'target_model',
            'message': 'Which run run run?',
            'choices':log_files
        }
    ]

    answers = prompt(questions)

    f = open(answers['target_model']+'/log.monitor.csv', 'r')
    # _args = json.loads(f.readline().replace('#',''))['args']
    _args = {}
    _args['play'] = True
    _args['layer_size'] = 2
    _args['network_size'] = 64
    _args['alg'] = 'sac'

    model_files = sorted(glob(answers['target_model'].replace('.monitor.csv','')+'/*_model.pkl'))
    model_files.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
    
    _args['model_path'] = model_files[-1]

    if args.prefix != "":
        _args['prefix'] = args.prefix

    _args['multi'] = True

    args = Namespace(**_args)
    print("Load saved args", args)
    f.close()

    main(args=args)

