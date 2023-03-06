import time
import numpy as np
import copy

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from .off_rl_algo import OffRLAlgo

class EARSAC(OffRLAlgo):
    """
    SAC for Emergent Action Representation
    """

    def __init__(
        self,
        pf_state,pf_task,pf_action,
        qf1, qf2,
        plr, qlr,
        task_nums = 1,
        optimizer_class=optim.Adam,

        policy_std_reg_weight=1e-3,
        policy_mean_reg_weight=1e-3,

        reparameterization=True,
        automatic_entropy_tuning=True,
        target_entropy=None,
        
        n_std = 0.02,
        
        **kwargs
    ):
        super(EARSAC,self).__init__(**kwargs)
        self.pf_state=pf_state
        self.pf_task=pf_task
        self.pf_action=pf_action
        self.qf1=qf1
        self.qf2=qf2

        self.target_qf1 = copy.deepcopy(qf1)
        self.target_qf2 = copy.deepcopy(qf2)

        self.to(self.device)

        self.plr = plr
        self.qlr = qlr

        self.optimizer_class = optimizer_class
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=self.qlr,
        )

        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=self.qlr,
        )

        self.pf_optimizer = optimizer_class(
            (para for para in list(self.pf_state.parameters()) + list(self.pf_task.parameters()) + list(self.pf_action.parameters())),
            lr=self.plr
        )

        self.automatic_entropy_tuning = automatic_entropy_tuning
        if self.automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(self.env.action_space.shape).item()  # from rlkit
            self.log_alpha = torch.zeros(1).to(self.device)
            self.log_alpha.requires_grad_()
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=self.plr,
            )
        self.sample_key = ["obs", "next_obs", "acts", "rewards", "terminals",  "task_idxs", "task_inputs"]
        self.qf_criterion = nn.MSELoss()

        self.policy_std_reg_weight = policy_std_reg_weight
        self.policy_mean_reg_weight = policy_mean_reg_weight

        self.reparameterization = reparameterization
        
        self.n_std = n_std

    def update(self, batch):
            self.training_update_num += 1
            obs       = batch['obs']
            actions   = batch['acts']
            next_obs  = batch['next_obs']
            rewards   = batch['rewards']
            terminals = batch['terminals']
            task_inputs = batch["task_inputs"]
            task_idx    = batch['task_idxs']

            rewards   = torch.Tensor(rewards).to( self.device )
            rewards_scaled = rewards * self.reward_scale
            terminals = torch.Tensor(terminals).to( self.device )
            obs       = torch.Tensor(obs).to( self.device )
            actions   = torch.Tensor(actions).to( self.device )
            next_obs  = torch.Tensor(next_obs).to( self.device )
            task_inputs = torch.Tensor(task_inputs).to(self.device)
            task_idx    = torch.Tensor(task_idx).to( self.device ).long()

            self.pf_state.train()
            self.pf_task.train()
            self.pf_action.train()
            self.qf1.train()
            self.qf2.train()

            """
            Policy operations.
            """
            lses = self.pf_state.forward(obs)
            ltes = self.pf_task.forward(task_inputs)
            n_mean=torch.zeros_like(ltes)
            n_std=torch.full_like(ltes, self.n_std)
            noise=torch.normal(n_mean,n_std)
            ltes_withnoise=ltes+noise
           
            ltes_withnoise=F.normalize(ltes_withnoise)
            sample_info = self.pf_action.explore(lses, ltes, return_log_probs=True )

            mean        = sample_info["mean"]
            log_std     = sample_info["log_std"]
            new_actions = sample_info["action"]
            log_probs   = sample_info["log_prob"]

            q1_pred = self.qf1([obs, actions, task_inputs])
            q2_pred = self.qf2([obs, actions, task_inputs])

            if self.automatic_entropy_tuning:
                """
                Alpha Loss
                """
                alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                alpha = self.log_alpha.exp().detach()
            else:
                alpha = 1
                alpha_loss = 0

            with torch.no_grad():
                lses = self.pf_state.forward(next_obs)

                target_sample_info = self.pf_action.explore(lses, ltes, return_log_probs=True )

                target_actions   = target_sample_info["action"]
                target_log_probs = target_sample_info["log_prob"]

                target_q1_pred = self.target_qf1([next_obs, target_actions,task_inputs])
                target_q2_pred = self.target_qf2([next_obs, target_actions,task_inputs])
                min_target_q = torch.min(target_q1_pred, target_q2_pred)
                target_v_values = min_target_q - alpha * target_log_probs
            """
            QF Loss
            """
            q_target = rewards_scaled + (1. - terminals) * self.discount * target_v_values
            qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
            qf2_loss = self.qf_criterion(q2_pred, q_target.detach())
            assert q1_pred.shape == q_target.shape
            assert q2_pred.shape == q_target.shape

            q_new_actions = torch.min(
                self.qf1([obs, new_actions,task_inputs]),
                self.qf2([obs, new_actions,task_inputs]))
            """
            Policy Loss
            """
            if not self.reparameterization:
                raise NotImplementedError
            else:
                assert log_probs.shape == q_new_actions.shape
                policy_loss = ( alpha * log_probs - q_new_actions).mean()

            std_reg_loss = self.policy_std_reg_weight * (log_std**2).mean()
            mean_reg_loss = self.policy_mean_reg_weight * (mean**2).mean()
            policy_loss += std_reg_loss + mean_reg_loss
            
            """
            Update Networks
            """
            self.pf_optimizer.zero_grad()
            policy_loss.backward()
            pf_state_norm = torch.nn.utils.clip_grad_norm_(self.pf_state.parameters(), 10)
            pf_task_norm = torch.nn.utils.clip_grad_norm_(self.pf_task.parameters(), 10)
            pf_action_norm = torch.nn.utils.clip_grad_norm_(self.pf_action.parameters(), 10)
            self.pf_optimizer.step()

            self.qf1_optimizer.zero_grad()
            qf1_loss.backward()
            qf1_norm = torch.nn.utils.clip_grad_norm_(self.qf1.parameters(), 10)
            self.qf1_optimizer.step()

            self.qf2_optimizer.zero_grad()
            qf2_loss.backward()
            qf2_norm = torch.nn.utils.clip_grad_norm_(self.qf2.parameters(), 10)
            self.qf2_optimizer.step()

            self._update_target_networks()

            # Information For Logger
            info = {}
            info['Reward_Mean'] = rewards.mean().item()

            if self.automatic_entropy_tuning:
                info["Alpha"] = alpha.item()
                info["Alpha_loss"] = alpha_loss.item()
            info['Training/policy_loss'] = policy_loss.item()
            info['Training/qf1_loss'] = qf1_loss.item()
            info['Training/qf2_loss'] = qf2_loss.item()

            info['Training/pf_state_norm'] = pf_state_norm.item()
            info['Training/pf_task_norm'] = pf_task_norm.item()
            info['Training/pf_action_norm'] = pf_action_norm.item()
            info['Training/qf1_norm'] = qf1_norm.item()
            info['Training/qf2_norm'] = qf2_norm.item()

            info['log_std/mean'] = log_std.mean().item()
            info['log_std/std'] = log_std.std().item()
            info['log_std/max'] = log_std.max().item()
            info['log_std/min'] = log_std.min().item()

            info['log_probs/mean'] = log_probs.mean().item()
            info['log_probs/std'] = log_probs.std().item()
            info['log_probs/max'] = log_probs.max().item()
            info['log_probs/min'] = log_probs.min().item()

            info['mean/mean'] = mean.mean().item()
            info['mean/std'] = mean.std().item()
            info['mean/max'] = mean.max().item()
            info['mean/min'] = mean.min().item()

            return info

    @property
    def networks(self):
        return [
            self.pf_state,
            self.pf_task,
            self.pf_action,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2
        ]
        
    @property
    def snapshot_networks(self):
        return [
            ["pf_state", self.pf_state],
            ["pf_task", self.pf_task],
            ["pf_action", self.pf_action],
            ["qf1", self.qf1],
            ["qf2", self.qf2],
        ]

    @property
    def target_networks(self):
        return [
            ( self.qf1, self.target_qf1 ),
            ( self.qf2, self.target_qf2 )
        ]
