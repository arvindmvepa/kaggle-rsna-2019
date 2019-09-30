from __future__ import absolute_import, print_function

from joblib import Parallel, delayed
import torch
import random

class CustomDataLoader(object):

    def __init__(self, dataset, batch_size = 1, shuffle = False, num_workers = None, backend='multiprocessing', *args,
                 **kwargs):
        super(CustomDataLoader, self).__init__()
        self.dataset = dataset
        self.shuffle = shuffle
        self.nimages = len(self.dataset)
        self.batcher = list(range(self.nimages))
        self.batch_size = batch_size
        self.reset()

        self.num_workers = num_workers
        self.backend = backend


    def __iter__(self):
        return self

    def __next__(self):
        """
        """
        if self.curr_i >= len(self.dataset):
            self.reset()
            raise StopIteration
        else:
            batch = self.batcher[self.curr_i * self.batch_size:(self.curr_i + 1) * self.batch_size]
            batch_data = Parallel(n_jobs=self.num_workers, backend=self.backend)(delayed(self.dataset.__getitem__)(x)
                                                                                 for x in batch)
            self.curr_i = self.curr_i + self.batch_size
            return  torch.stack(batch_data)

    def reset(self):
        self.curr_i = 0
        if self.shuffle:
            random.shuffle(self.batcher)

    def __len__(self):
        return len(self.dataset) // self.batch_size
