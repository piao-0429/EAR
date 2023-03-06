import numpy as np

class BaseReplayBuffer():
    """
    Basic Replay Buffer
    """
    def __init__(
        self, max_replay_buffer_size, time_limit_filter=False,
    ):
        self.worker_nums = 1
        self._max_replay_buffer_size = max_replay_buffer_size
        self._top = 0
        self._size = 0
        self.time_limit_filter = time_limit_filter

    def add_sample(self, sample_dict, task_rank=0, **kwargs):
        for key in sample_dict:
            if not hasattr(self, "_" + key):
                self.__setattr__(
                    "_" + key,
                    np.zeros((self._max_replay_buffer_size, 1) + \
                            np.shape(sample_dict[key])))
            self.__getattribute__("_" + key)[self._top, 0] = sample_dict[key] 
        self._advance()

    def terminate_episode(self):
        pass

    def _advance(self):
        self._top = (self._top + 1) % self._max_replay_buffer_size
        if self._size < self._max_replay_buffer_size:
            self._size += 1

    def random_batch(self, batch_size, sample_key):
        indices = np.random.randint(0, self._size, batch_size)
        return_dict = {}
        for key in sample_key:
            return_dict[key] = np.squeeze(self.__getattribute__("_"+key)[indices], axis=1)
        return return_dict

    def num_steps_can_sample(self):
        return self._size

class BaseMTReplayBuffer(BaseReplayBuffer):
    """
    Just for imitation Learning
    """
    def __init__(self, 
        max_replay_buffer_size,
        task_nums,
    ):
        super(BaseMTReplayBuffer, self).__init__(max_replay_buffer_size)
        self.task_nums = task_nums

    # Not USED
    def add_sample(self, sample_dict, task_rank = 0, **kwargs):
        pass

    def terminate_episode(self):
        pass

    def _advance(self):
        pass

    def random_batch(self, batch_size, sample_key, reshape = True):
        assert batch_size % self.task_nums== 0, \
            "batch size should be dividable by worker_nums"
        batch_size //= self.task_nums
        size = self.num_steps_can_sample()
        indices = np.random.randint(0, size, batch_size)
        return_dict = {}
        for key in sample_key:
            return_dict[key] = self.__getattribute__("_"+key)[indices]
            if reshape:
                return_dict[key] = return_dict[key].reshape(
                    (batch_size * self.worker_nums, -1))
        return return_dict

    def num_steps_can_sample(self):
        return self._size