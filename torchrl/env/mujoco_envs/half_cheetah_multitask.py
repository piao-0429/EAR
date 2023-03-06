import numpy as np

from . import register_env
from gym.envs.mujoco import HalfCheetahEnv


@register_env('HalfCheetah-Vel')
class HalfCheetahMultitaskEnv(HalfCheetahEnv):

    def __init__(self, num_tasks=10, mode='train'):
        self.tasks = self.sample_tasks(num_tasks, mode)
        super(HalfCheetahMultitaskEnv, self).__init__()

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]

        forward_vel = (xposafter - xposbefore) / self.dt
        ctrl_cost = 0.1 * np.sum(np.square(action))
        observation = self._get_obs()
        
        rewards = [-abs(task['velocity'] - forward_vel) - ctrl_cost for task in self.tasks]

        done = False
        infos = dict(
            rewards=rewards,
            x_velocity = forward_vel
            )
        return (observation, rewards, done, infos)

    def sample_tasks(self, num_tasks, mode):
        if mode == 'train':
            velocities = np.linspace(1, 10, num=num_tasks)
        elif mode == 'adapt':
            velocities = np.random.randint(0, 100, size=num_tasks)/10
        elif mode == 'interpolate':
            velocities = [0]
        
        tasks = [{'velocity': velocity} for velocity in velocities]
        return tasks 

    def get_all_task_idx(self):
        return range(len(self.tasks))
    
    def get_task_list(self):
        task_list = ['velocity-' + str(task['velocity']) for task in self.tasks]
        return task_list
        