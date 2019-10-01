from __future__ import absolute_import, print_function

from joblib import Parallel, delayed
from torch.utils.data._utils.collate import default_collate
import time
import random

dataset_ = None
def get_ds_data(chunk):
    global dataset_
    output = []
    chunk_fetch_beg = time.time()
    for x in chunk:
        output.append(dataset_[x])
    chunk_fetch_end = time.time()
    print("time fetch chunk: {}".format(chunk_fetch_end - chunk_fetch_beg))
    return output


class CustomDataLoader(object):

    def __init__(self, dataset, batch_size = 1, shuffle = False, num_workers = None, backend='loky', *args,
                 **kwargs):
        super(CustomDataLoader, self).__init__()
        global dataset_
        dataset_ = dataset
        self.dataset = dataset
        self.shuffle = shuffle
        self.batcher = list(self.dataset.ids.keys())
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

        if (self.curr_i * self.batch_size) >= len(self.dataset):
            self.reset()
            raise StopIteration
        else:
            batch_indices = self.batcher[self.curr_i * self.batch_size:(self.curr_i + 1) * self.batch_size]
            batch_chunks = [batch_indices[j*self.chunk_size:(j+1)*self.chunk_size] for j in range(self.num_workers)]
            dl_get_batch0 = time.time()
            batch_data = Parallel(n_jobs=self.num_workers, backend=self.backend)(delayed(get_ds_data)(chunk)
                                                                                 for chunk in batch_chunks)
            dl_get_batch1 = time.time()
            print("time for get batch: {}".format(dl_get_batch1 - dl_get_batch0))
            self.curr_i = self.curr_i + 1
            dl_collate0 = time.time()
            batch_data = [item for chunk in batch_data for item in chunk]
            batch_data = default_collate(batch_data)
            dl_collate1 = time.time()
            print("time for collate: {}".format(dl_collate1 - dl_collate0))
            dl_next1 = time.time()
            print("time for next: {}".format(dl_next1 - dl_next0))
            return batch_data

    def reset(self):
        self.curr_i = 0
        if self.shuffle:
            random.shuffle(self.batcher)

    def __len__(self):
        return len(self.dataset) // self.batch_size