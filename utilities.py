from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import warnings
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
import os

class AbstractNetwork(ABC, nn.Module):
    def __init__(self, outputs):
        super().__init__()
        self.output_size = outputs
        self._task = 0
        self.used_tasks = set()

    @abstractmethod
    def build_net(self):
        pass

    @abstractmethod
    def eval_forward(self, x):
        pass

    @abstractmethod
    def embedding(self, x):
        pass

    @property
    def task(self):
        return self._task

    @task.setter
    def task(self, value):
        # if value > self.output_size:
        #     value = self.output_size
        # self._used_tasks.update(value)

        self._task = value

    @task.getter
    def task(self):
        return self._task

class GeneralDatasetLoader(ABC, Dataset):
    def __init__(self, folder: str, transform=None, target_transform=None, *args, **kwargs):
        super(Dataset).__init__()
        self.folder = folder

        self.transform = transform
        self.target_transform = target_transform

        # download fields
        self.url = None
        self.filename = None
        self.unzipped_folder = None
        self.download_path = os.path.join(folder, 'download')

        self.transform = transform
        self.target_transform = target_transform

        self._phase = 'train'
        self._current_task = 0

        self.download = False
        self.force_download = False

        self.train_split = 1.0

        self.task_manager = None

        self.X, self.Y, self.class_to_idx, self.idx_to_class = None, None, None, None
        self.task2idx = None
        self._n_tasks = None
        self.class_to_idx = None
        self.idx_to_class = None

    def train_phase(self):
        self._phase = 'train'

    def test_phase(self):
        self._phase = 'test'

    def next_task(self, round_robin=False):
        self._current_task = self._current_task + 1

        if round_robin:
            self._current_task = self._current_task % self._n_tasks
        else:
            if self._current_task > self._n_tasks - 1:
                warnings.warn("No more tasks...")
                self._current_task = self._n_tasks - 1
                return False
        return True

    def reset(self):
        self._phase = 'train'
        self._current_task = 0

    @property
    def tasks_number(self):
        return self._n_tasks

    @property
    def phase(self):
        return self._phase

    @property
    def task(self):
        return self._current_task

    @task.setter
    def task(self, value):
        # if value >= self._n_tasks:
        #     value = self._n_tasks - 1
        self._current_task = value

    @task.getter
    def task(self):
        return self._current_task

    def task_mask(self, task=None):
        if task is None:
            task = self._current_task
        return list(self.task2idx.keys())[task]

    @abstractmethod
    def getIterator(self, batch_size, task=None):
        raise NotImplementedError

    @abstractmethod
    def load_dataset(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def download_dataset(self):
        raise NotImplementedError

    def already_downloaded(self):
        print(self.download_path)
        if not os.path.exists(self.download_path):
            os.makedirs(self.download_path)
            return False
        else:
            if len(os.listdir(self.download_path)) == 0:
                return False
            return True