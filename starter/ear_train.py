import sys
sys.path.append(".") 

import torch
import os
import time
import os.path as osp
import numpy as np
from torchrl.utils import get_args
from torchrl.utils import get_params
from torchrl.env import get_env
from torchrl.env.mujoco_envs import ENVS
from torchrl.utils import Logger

args = get_args()
params = get_params(args.config)

import torchrl.policies as policies
import torchrl.networks as networks
from torchrl.algo import EARSAC
from torchrl.collector.para.async_mt import AsyncMultiTaskParallelCollectorForEAR
from torchrl.replay_buffers.shared import AsyncSharedReplayBuffer
import gym




def experiment(args):

    device = torch.device("cuda:{}".format(args.device) if args.cuda else "cpu")
    
    num_tasks = params['num_tasks']
    if params['env_name'] == 'HalfCheetah-RunJump':
        env = ENVS[params['env_name']](num_v_tasks=params['num_v_tasks'], num_h_tasks=params['num_h_tasks'])
    else:
        env = ENVS[params['env_name']](num_tasks=num_tasks)
    task_list = env.get_task_list()
    
    
    lse_shape = params['lse_shape']
    lte_shape = params['lte_shape']

    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.backends.cudnn.deterministic=True
    
    buffer_param = params['replay_buffer']

    experiment_name = os.path.split( os.path.splitext( args.config )[0] )[-1] if args.id is None \
        else args.id
    logger = Logger(experiment_name , params['env_name'], args.seed, params, args.log_dir)

    params['general_setting']['env'] = env
    params['general_setting']['logger'] = logger
    params['general_setting']['device'] = device

    params['state_net']['base_type']=networks.MLPBase
    params['task_net']['base_type']=networks.MLPBase
    params['action_net']['base_type']=networks.MLPBase
    params['q_net']['base_type']=networks.MLPBase

    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    pf_state = networks.NormNet(
        input_shape=env.observation_space.shape[0], 
        output_shape=lse_shape,
        **params['state_net']
    )

    pf_task=networks.NormNet(
        input_shape=num_tasks, 
        output_shape=lte_shape,
        **params['task_net']
    )

    pf_action=policies.EmergentActionRepresentationGuassianContPolicy(
        input_shape = lse_shape + lte_shape,
        output_shape = 2 * env.action_space.shape[0],
        **params['action_net'] 
    )
    
    qf1 = networks.FlattenNet( 
        input_shape = env.observation_space.shape[0] + env.action_space.shape[0] + num_tasks,
        output_shape = 1,
        **params['q_net'] 
    )
    qf2 = networks.FlattenNet( 
        input_shape = env.observation_space.shape[0] + env.action_space.shape[0] + num_tasks,
        output_shape = 1,
        **params['q_net'] 
    )
    
    example_ob = env.reset()
    example_dict = { 
        "obs": example_ob,
        "next_obs": example_ob,
        "acts": env.action_space.sample(),
        "rewards": [0],
        "terminals": [False],
        "task_idxs": [0],
        "task_inputs": np.zeros(num_tasks),
    }
    
    replay_buffer = AsyncSharedReplayBuffer( int(buffer_param['size']),
            args.worker_nums
    )
    replay_buffer.build_by_example(example_dict)

    params['general_setting']['replay_buffer'] = replay_buffer

    epochs = params['general_setting']['pretrain_epochs'] + \
        params['general_setting']['num_epochs']

    params['general_setting']['collector'] = AsyncMultiTaskParallelCollectorForEAR(
        env=env, pf=[pf_state, pf_task, pf_action], replay_buffer=replay_buffer,
        task_list=task_list,
        device=device,
        reset_idx=True,
        epoch_frames=params['general_setting']['epoch_frames'],
        max_episode_frames=params['general_setting']['max_episode_frames'],
        eval_episodes = params['general_setting']['eval_episodes'],
        worker_nums=args.worker_nums, eval_worker_nums=args.eval_worker_nums,
        train_epochs = epochs, eval_epochs= params['general_setting']['num_epochs']
    )
    params['general_setting']['batch_size'] = int(params['general_setting']['batch_size'])
    params['general_setting']['save_dir'] = osp.join(logger.work_dir,"model")

    agent = EARSAC(
        pf_state = pf_state,
        pf_task = pf_task,
        pf_action = pf_action,
        qf1 = qf1,
        qf2 = qf2,
        **params['sac'],
        **params['general_setting']
    )
    agent.train()
    

if __name__ == "__main__":
    experiment(args)



    