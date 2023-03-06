import numpy as np

from . import register_env
from gym.envs.mujoco import HalfCheetahEnv


@register_env('HalfCheetah-RunJump')
class HalfCheetahMixtaskEnv(HalfCheetahEnv):

    def __init__(self, num_v_tasks=7, num_h_tasks=3, mode='train'):
        self.num_v_tasks = num_v_tasks
        self.num_h_tasks = num_h_tasks
        self.tasks = self.sample_tasks(num_v_tasks, num_h_tasks, mode)
        super(HalfCheetahMixtaskEnv, self).__init__()

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]

        forward_vel = (xposafter - xposbefore) / self.dt
        ctrl_cost = 0.1 * np.sum(np.square(action))
        observation = self._get_obs()
        z_position = observation[0]

        rewards = []
        for task in self.tasks:
            if 'velocity' in task:
                rewards.append(-abs(task['velocity'] - forward_vel) - ctrl_cost)
            elif 'height_weight' in task:
                rewards.append(task['height_weight'] * z_position - ctrl_cost)
        done = False
        infos = dict(
            rewards=rewards,
            x_velocity = forward_vel,
            height = z_position
            )
        return (observation, rewards, done, infos)

    def sample_tasks(self, num_v_tasks, num_h_tasks, mode):
        if mode == 'train':
            velocities = np.linspace(1, 7, num=num_v_tasks)
            height_weights = np.array([1, 5, 10])
        elif mode == 'interpolate':
            velocities = [0]
            height_weights = [0]
        
        tasks = [{'velocity': velocity} for velocity in velocities] + [{'height_weight': height_weight} for height_weight in height_weights]
    
        return tasks 

    def get_all_task_idx(self):
        return range(len(self.tasks))
    
    def get_task_list(self):
        task_list = []
        for task in self.tasks:
            if 'velocity' in task:
                task_list.append('velocity-' + str(task['velocity']))
            elif 'height_weight' in task:
                task_list.append('height_weight-' + str(task['height_weight']))
        
        return task_list