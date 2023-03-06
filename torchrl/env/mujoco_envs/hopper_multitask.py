import numpy as np

from . import register_env
from gym.envs.mujoco import HopperEnv

@register_env('Hopper-Vel')
class HopperMultitaskEnv(HopperEnv):

    def __init__(self,  num_tasks=10, mode='train'):
        self.tasks = self.sample_tasks(num_tasks, mode)
        super(HopperMultitaskEnv, self).__init__()

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter, height, ang = self.sim.data.qpos[0:3]

        forward_vel = (xposafter - xposbefore) / self.dt
        ctrl_cost = 1e-3 * np.sum(np.square(action))
        healthy_reward = 1.05
        s=self.state_vector()
        observation = self._get_obs()
        rewards = [-abs(forward_vel - task['velocity']) -ctrl_cost + healthy_reward for task in self.tasks]
        done = not (
            np.isfinite(s).all()
            and (np.abs(s[2:]) < 100).all()
            and (height > 0.7)
            and (abs(ang) < 0.2)
        )
        infos = dict(
            rewards=rewards,
            x_velocity = forward_vel,
            )
        return (observation, rewards, done, infos)

    def sample_tasks(self, num_tasks, mode):
        if mode == 'train':
            velocities = np.linspace(0.2, 2, num=num_tasks)
        elif mode == 'adapt':
            velocities = np.random.randint(0, 20, size=num_tasks)/10
        elif mode == 'interpolate':
            velocities = [0]
    
        tasks = [{'velocity': velocity} for velocity in velocities]
        return tasks

    def get_all_task_idx(self):
        return range(len(self.tasks))
    
    def get_task_list(self):
        task_list = ['velocity-' + str(np.around(task['velocity'],1)) for task in self.tasks]
        return task_list



