import numpy as np

from gym.envs.mujoco import AntEnv
from . import register_env


@register_env('Ant-Dir')
class AntMultitaskEnv(AntEnv):

    def __init__(self, num_tasks=24, mode="train"):
        self.tasks = self.sample_tasks(num_tasks, mode)
        super(AntMultitaskEnv, self).__init__()

    def step(self, action):
        torso_xyz_before = np.array(self.get_body_com("torso"))

        

        self.do_simulation(action, self.frame_skip)
        torso_xyz_after = np.array(self.get_body_com("torso"))
        torso_velocity = (torso_xyz_after - torso_xyz_before)/self.dt
        ctrl_cost = 0.5 * np.square(action).sum()
        
        contact_cost = 0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        rewards = []
        
        for task in self.tasks:
            direction = task['direction']/ 180 * np.pi
            dir_reward = (np.cos(direction), np.sin(direction))
            dir_penalty = (-np.sin(direction), np.cos(direction))
        
            forward_reward = np.dot(torso_velocity[:2], dir_reward)
            # forward_penalty = abs(np.dot(torso_velocity[:2], dir_penalty))
            forward_penalty = 0
            
            rewards.append(forward_reward - forward_penalty - ctrl_cost - contact_cost + survive_reward)
        
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
                  and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return ob, rewards, done, dict(
            rewards = rewards, 
            x_velocity = torso_velocity[0],
            y_velocity = torso_velocity[1],
            x_position = torso_xyz_after[0],
            y_position = torso_xyz_after[1]
        )

    def sample_tasks(self, num_tasks, mode):
        if mode == "train":
            directions = np.linspace(0, 360, endpoint=False, num=num_tasks)
        elif mode == "adapt":
            directions = np.random.randint(0, 360, size=num_tasks)
        elif mode == 'interpolate':
            directions = [0]
            
        tasks = [{'direction': direction} for direction in directions]
        return tasks
    
    def get_all_task_idx(self):
        return range(len(self.tasks))
    
    def get_task_list(self):
        task_list = ['direction-' + str(task['direction']) for task in self.tasks]
        return task_list