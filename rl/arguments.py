
# import re
import math

import argparse
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def get_common_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_num', type=int, default=16, help='Number of environments to simultaneously run')
    parser.add_argument('--ent_coeff', type=float, default=0.00, help='Entropy bonus')    
    parser.add_argument('--n_cpus', type=int, default=8, help='Number of Cpus to simultaneously run')
    parser.add_argument('--gae_lambda', type=float, default=.95, help='GAE lambda')   
    parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs to run on each rollout buffer')      
    parser.add_argument('--n_steps', type=int, default=2048, help='Number of steps before training')
    parser.add_argument('--total_timestep', type=int, default=10000, help='Total number of time steps throughout the training')
    parser.add_argument('--save_name', type=str, default="None", help='Total number of time steps throughout the training')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument("--adapted_obs",type=str2bool,default= False,help = "Whether to reward for getting closer to the model")
    parser.add_argument("--port",type=int,default= 5004,help = "Whether to reward for getting closer to the model")
    parser.add_argument("--host",type=str,default="0.0.0.0",help = "Whether to reward for getting closer to the model")
    parser.add_argument("--load",type=str2bool,default=False,help = "Whether to load a pretrained model")
    parser.add_argument("--env_size",type=int,default= 16,help = "Environment size")
    parser.add_argument("--load_file_path",type=str,default=False,help = "Model file path in linux format")
    parser.add_argument("--kldiv",type=float,default=None,help = "kl limit")
    parser.add_argument("--guard_distance_reward",type=str2bool,default= True,help = "Whether to reward for getting closer to the model")
    args = parser.parse_args()
    return args