
def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    import argparse
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def common_arg_parser():
    """
    Create an argparse.ArgumentParser for run_mujoco.py.
    """
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', type=str, default='tactime_simple')
    parser.add_argument('--seed', help='RNG seed', type=int, default=None)
    parser.add_argument('--alg', help='Algorithm', type=str, default='sac')
    
    parser.add_argument('--num_timesteps', type=float, default=2e10),
    parser.add_argument('--network', help='network type (mlp, cnn, lstm, cnn_lstm, conv_only)', default='mlp_big')
    parser.add_argument('--num_env', help='Number of environment copies being run in parallel. When not specified, set to number of cpus for Atari, and to 1 for Mujoco', default=None, type=int)
    parser.add_argument('--save_path', help='Path to save trained model to', default=None, type=str)
    parser.add_argument('--play', default=False, action='store_true')
    parser.add_argument('--render', default=False, type=bool)

    parser.add_argument('--prefix', default='', type=str)
    parser.add_argument('--model_path', help='Path to load trained model to', default=None, type=str)
    parser.add_argument('--reward', help='', default='com_roll_pitch', type=str)

    parser.add_argument('--network_size', help='plot', default=64, type=int)
    parser.add_argument('--layer_size', help='plot', default=2, type=int)

    return parser

def parseLayersFromArgs(args):
    layers = []
    for l in range(args.layer_size):
        layers.append(args.network_size)

    return layers