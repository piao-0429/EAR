import sys
import csv
from threading import Thread
import copy

sys.path.append(".") 

import torch
import os
import time
import os.path as osp

import numpy as np
import torch.nn.functional as F
from torchrl.utils import get_args
from torchrl.utils import get_params
from torchrl.env import get_env
from torchrl.env.mujoco_envs import ENVS
from torchrl.utils import Logger_Adaptation


args = get_args()
params = get_params(args.config)

import torchrl.policies as policies
import torchrl.networks as networks
from torchrl.algo import EARCEM
import gym

def adapt(experiment_name, env, pf_state, pf_action, params, task_idx, task):
    
    logger = Logger_Adaptation(experiment_name, params['env_name'], args.seed, params, task, args.log_dir)
    
    
    agent = EARCEM(
        env = env,
        task = task,
        task_idx = task_idx,
        lse_shape = params['lse_shape'],
        lte_shape = params['lte_shape'],
        pf_state = pf_state,
        pf_action = pf_action,
        logger = logger,
        **params['general_setting']
    )
    agent.adapt()

def experiment(args):

    device = torch.device("cuda:{}".format(args.device) if args.cuda else "cpu")
    num_tasks = params['num_tasks']
    env = ENVS[params['env_name']](num_tasks=num_tasks, mode='adapt')
    task_list = env.get_task_list()
    
    lse_shape = params['lse_shape']
    lte_shape = params['lte_shape']

    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.backends.cudnn.deterministic=True
        
    experiment_name = os.path.split(os.path.splitext( args.config )[0] )[-1] if args.id is None \
        else args.id

    pf_state = networks.NormNet(
        input_shape=env.observation_space.shape[0], 
        output_shape=lse_shape,
        base_type=networks.MLPBase,
        **params['state_net']
    )
    
    pf_action=policies.EmergentActionRepresentationGuassianContPolicy(
        input_shape = lse_shape + lte_shape,
        output_shape = 2 * env.action_space.shape[0],
        base_type=networks.MLPBase,
        **params['action_net'] 
    )
    
    model_dir = "log/"+experiment_name+"/"+params['env_name']+"/"+str(args.seed)+"/model/"
    pf_state.load_state_dict(torch.load(model_dir+"model_pf_state_best.pth", map_location='cpu'))
    pf_action.load_state_dict(torch.load(model_dir+"model_pf_action_best.pth", map_location='cpu'))
    
    experiment_name = experiment_name + '_adapt'
    
    for task_idx, task in enumerate(task_list):
        t = Thread(target=adapt, args=(experiment_name, env, pf_state, pf_action, params, task_idx, task))
        t.start()

        
if __name__ == "__main__":
    experiment(args)



    