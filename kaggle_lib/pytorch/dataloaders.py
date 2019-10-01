from __future__ import absolute_import, print_function

from joblib import Parallel, delayed
#from torch.utils.data._utils.collate import default_collate
import time
import torch
import random


def get_ds_data(chunk, dataset):
    output = []
    for x in chunk:
        print('dataset index: {}'.format(x))
        output.append(dataset[x])
    return output


class CustomDataLoader(object):

    def __init__(self, dataset, batch_size = 1, shuffle = False, num_workers = None, backend='multiprocessing', *args,
                 **kwargs):
        super(CustomDataLoader, self).__init__()
        self.dataset = dataset
        self.shuffle = shuffle
        self.batcher = list(self.dataset.ids.keys())
        print("dataset ids first {}, last {}".format(self.batcher[0], self.batcher[len(self.batcher)-1]))
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.chunk_size = self.batch_size // self.num_workers
        self.backend = backend
        self.reset()

    def __iter__(self):
        return self

    def __next__(self):
        """
        """
        dl_next0 = time.time()

        if self.curr_i >= (len(self.dataset)-1):
            self.reset()
            raise StopIteration
        else:
            batch_indices = self.batcher[self.curr_i * self.batch_size:(self.curr_i + 1) * self.batch_size]
            batch_chunks = [batch_indices[j*self.chunk_size:(j+1)*self.chunk_size] for j in range(self.num_workers)]
            dl_get_batch0 = time.time()
            print("batch_chunks: {}".format(batch_chunks))
            batch_data = Parallel(n_jobs=self.num_workers, backend=self.backend)(delayed(get_ds_data)(chunk, self.dataset)
                                                                                 for chunk in batch_chunks)
            dl_get_batch1 = time.time()
            print("time for get batch: {}".format(dl_get_batch1 - dl_get_batch0))
            self.curr_i = self.curr_i + self.batch_size
            dl_collate0 = time.time()
            batch_data = [item for chunk in batch_data for item in chunk]
            batch_data_dict = {"image": torch.stack([data['image'] for data in batch_data], 0),
                               "target": torch.stack([data['target'] for data in batch_data], 0)}
            dl_collate1 = time.time()
            print("time for collate: {}".format(dl_collate1 - dl_collate0))
            dl_next1 = time.time()
            print("time for next: {}".format(dl_next1 - dl_next0))
            return batch_data_dict

    def reset(self):
        self.curr_i = 0
        if self.shuffle:
            random.shuffle(self.batcher)

    def __len__(self):
        return len(self.dataset) // self.batch_size