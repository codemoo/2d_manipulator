import numpy as np
import os
from stable_baselines.results_plotter import load_results, ts2xy
import tensorflow as tf

best_mean_reward, n_steps = -np.inf, 0

# Create log dir
log_dir = "tmp/"
results_dir = "logs/"
benchmark_dir = "benchmarks/"

os.makedirs(log_dir, exist_ok=True)

def getBestRewardCallback(args):
    if not os.path.isdir(log_dir + args.prefix):
        os.makedirs(log_dir + args.prefix, exist_ok=True)

    def bestRewardCallback(_locals, _globals, model=None):
        """
        Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
        :param _locals: (dict)
        :param _globals: (dict)
        """
        global n_steps, best_mean_reward
        # Print stats every 1000 calls
        divider = 2
        print(best_mean_reward)
        if (n_steps + 1) % divider == 0 and (n_steps + 1) / divider > 1:
            # if _locals['self'].env.get_attr('proc_id') == 0:
        # if True:
            # Evaluate policy training performance
            x, y = ts2xy(load_results(log_dir+args.prefix), 'timesteps')
            if len(x) > 0:
                mean_reward = np.nanmean(y[-100:])

                # New best model, you could save the agent here
                if mean_reward > best_mean_reward:
                    best_mean_reward = mean_reward
                    # Example for saving best model
                    print("Saving new best model",best_mean_reward)
                    if model is not None:
                        model.save(log_dir + args.prefix + '/' + str(n_steps) +'_best_model.pkl')
                    else:
                        _locals['self'].save(log_dir + args.prefix + '/' + str(n_steps) +'_best_model.pkl')

                elif (n_steps + 1) % (divider*200) == 0:
                    if not os.path.exists(log_dir + args.prefix + '/checkpoints'):
                        os.makedirs(log_dir + args.prefix + '/checkpoints')
                    print("Saving checkpoint",best_mean_reward)
                    if model is not None:
                        model.save(log_dir + args.prefix + '/checkpoints/' + str(n_steps) +'_Check_model.pkl')
                    else:
                        _locals['self'].save(log_dir + args.prefix + '/checkpoints/' + str(n_steps) +'_Check_model.pkl')
                    
                if model is not None:
                    model.save(log_dir + args.prefix + '/model.pkl')
                else:
                    _locals['self'].save(log_dir + args.prefix + '/model.pkl')
    
        n_steps += 1

        return True

    return bestRewardCallback

def logDir():
    return log_dir

def resultsDir():
    return results_dir

def benchmarkDir():
    return benchmark_dir