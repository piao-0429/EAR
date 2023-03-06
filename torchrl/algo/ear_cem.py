import time
import numpy as np
import copy

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import os
import os.path as osp

class EARCEM():
    """
    CEM-like Adaptation for Emergent Action Representation
    """

    def __init__(
        self,
        env,
        task,
        task_idx,
        pf_state,pf_action,
        lse_shape, lte_shape,
        device = 'cpu',
        logger = None,
        save_dir = None,
        num_epoch = 10,
        num_sample = 10, 
        num_best = 5,
        n_std = 0.5,
        max_episode_frames = 200
    ):
        self.env = env
        self.task = task
        self.task_idx = task_idx
        self.pf_state=pf_state
        self.pf_action=pf_action
        self.lse_shape = lse_shape
        self.lte_shape = lte_shape
        
        self.num_epoch = num_epoch
        self.num_sample = num_sample
        self.num_best = num_best
        self.n_std = n_std
        self.max_episode_frames = max_episode_frames
        
        self.logger = logger

    def adapt(self):
        lte_shape = self.lte_shape
        num_sample = self.num_sample
        num_best = self.num_best
        n_mean = torch.zeros(((num_sample-1) * num_best, 1, lte_shape))
        n_std = torch.full(((num_sample-1) * num_best, 1, lte_shape), self.n_std)
        best_ltes = F.normalize(torch.randn((num_best, lte_shape))).unsqueeze(1)
        zeros = torch.zeros_like(best_ltes)
        
        
            
        lte = F.normalize(torch.randn((1, lte_shape)))
        with torch.no_grad():
            eval_reward = 0
            ob = self.env.reset()
            for _ in range(self.max_episode_frames):
                lse = self.pf_state.forward(torch.Tensor(ob).unsqueeze(0).to("cpu"))
                out = self.pf_action.explore(lse, lte)
                action = out["action"]
                action = action.detach().numpy()
                next_ob, rewards, done, info = self.env.step(action)
                reward = rewards[self.task_idx]
                eval_reward += reward
                ob = next_ob
                if done:
                    break
        self.logger.add_epoch_info(0, eval_reward)      
        
        for epoch in range(self.num_epoch):
            sample_ltes = best_ltes.repeat(num_sample, 1, 1)
            
            noises = torch.cat((zeros, torch.normal(n_mean, n_std)), dim=0)
            
            sample_ltes = F.normalize(sample_ltes + noises, dim=-1)
            
            lte_info = []
            with torch.no_grad():
                for j in range(num_best * num_sample):
                    eval_reward = 0
                    ob = self.env.reset()
                    for t in range(self.max_episode_frames):
                        lse = self.pf_state.forward(torch.Tensor(ob).unsqueeze(0).to("cpu"))
                        out = self.pf_action.explore(lse, sample_ltes[j])
                        action = out["action"]
                        action = action.detach().numpy()
                        next_ob, rewards, done, info = self.env.step(action)
                        reward = rewards[self.task_idx]
                        eval_reward += reward
                        ob = next_ob
                        if done:
                            break
                    lte_info.append({'id': j, 'eval_reward': eval_reward})
                    
            lte_info = sorted(lte_info, key = lambda x:x['eval_reward'], reverse=True)
            self.logger.add_epoch_info(epoch+1, lte_info[0]['eval_reward'])

            print("----------------------------------------------------------------------------------")
            print(self.task, ":", "Epoch", epoch+1)
            sum_reward = 0
            for j in range(num_best):
                best_ltes[j] = sample_ltes[lte_info[j]['id']]
                sum_reward += lte_info[j]['eval_reward']
                print("Best", j+1, "th reward:", lte_info[j]['eval_reward'])
                print("Best", j+1, "th lte:", best_ltes[j].squeeze(0).detach().cpu().numpy())
            print("----------------------------------------------------------------------------------") 
                