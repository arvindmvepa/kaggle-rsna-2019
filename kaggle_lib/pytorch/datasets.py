from __future__ import absolute_import, print_function

import os
from collections import defaultdict
import h5py
import numpy as np
import pandas as pd
import torch
from albumentations.core.composition import BaseCompose
from torchvision.datasets.vision import VisionDataset

import random

from ..dicom import sitk_read_image
from .utils import Timer

def h5_read_image(fn):
    with h5py.File(fn, 'r') as f:
        array = np.array(f['data']).astype('float32')
    return array


readers = {
    'dcm': sitk_read_image,
    'h5': h5_read_image,
}


class RSNA2019Dataset(VisionDataset):

    label_map = {
        'any': 'any',
        'edh': 'epidural',
        'iph': 'intraparenchymal',
        'ivh': 'intraventricular',
        'sah': 'subarachnoid',
        'sdh': 'subdural',
    }

    def __init__(self, root, csv_file, transform=None, target_transform=None, transforms=None,
                 convert_rgb=True, preprocessing=None, img_ids=None,
                 reader='h5',
                 class_order=('sdh', 'sah', 'ivh', 'iph', 'edh', 'any'), debug=False,
                 limit=None,
                 **filter_params):

        self.timers = defaultdict(Timer)
        self.debug = debug

        self.timers['init/super'].tic()
        super(RSNA2019Dataset, self).__init__(root, transforms, transform, target_transform)
        self.timers['init/super'].toc()

        self.timers['init/read_csv'].tic()
        data = pd.read_csv(csv_file).set_index('ImageId')

        data['fullpath'] = self.root + "/" + data['filepath']

        self.timers['int/read_csv'].toc()

        # assert all(c in self.label_map for c in class_order), "bad class order"
        self.class_order = class_order

        self.timers['init/setup_img_ids'].tic()
        img_ids = img_ids or data.index.tolist()
        if limit:
            random.shuffle(img_ids)
            img_ids = img_ids[:limit]
        img_ids = self.apply_filter(img_ids, **filter_params)
        self.ids = {i: imgid for i, imgid in enumerate(img_ids)}

        self._num_images = len(self.ids)

        # id = list(self.ids.values())
        # random.shuffle(id)
        # self.id = id[0]
        # self.row = data.loc[self.id].to_dict()
        # self.path = self.row['fullpath']

        self.data = data.loc[list(self.ids.values())].T.to_dict()

        self.rev_ids = {v: k for k, v in self.ids.items()}
        self.timers['init/setup_img_ids'].toc()

        self.timers['init/finish'].tic()
        self.transforms_are_albumentation = isinstance(self.transforms, BaseCompose)
        self.convert_rgb = convert_rgb
        self.preprocessing = preprocessing

        assert reader in readers, 'bad reader type'

        self.image_ext = reader
        self.read_image = readers[reader]
        self.timers['init/finish'].toc()

        if self.debug:
            for name, t in self.timers.items():
                print(name, t.average_time_str)

        self.timers = defaultdict(Timer)

    def apply_filter(self, img_ids, **filter_params):
        # place holder for now
        return img_ids

    def h5_read_image(self, fn):
        self.timers['actual_read'].tic()
        with h5py.File(fn, 'r') as f:
            array = np.array(f['data']).astype('float32')
        self.timers['actual_read'].toc()
        return array

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is a list of captions for the image.
        """
        self.timers['get/get_id'].tic()
        img_id = self.ids[index]
        # img_id = self.id
        self.timers['get/get_id'].toc()

        self.timers['get/get_row'].tic()
        image_row = self.data[img_id]
        self.timers['get/get_row'].toc()

        self.timers['get/get_path'].tic()
        path = image_row['fullpath']
        # path = self.path
        path = os.path.splitext(path)[0] + '.' + self.image_ext
        self.timers['get/get_path'].toc()
        self.timers['get/read_image'].tic()
        img = self.read_image(path)
        # img = self.h5_read_image(path)
        self.timers['get/read_image'].toc()

        self.timers['get/get_target'].tic()
        # try:
        #     target = [(image_row['label__' + self.label_map[c]]) for c in self.class_order]
        # except KeyError:
        target = None
        self.timers['get/get_target'].toc()

        self.timers['get/augmentation'].tic()

        output = dict(image=img)
        if self.transforms is not None:
            if self.transforms_are_albumentation:
                output = self.transforms(**output)
            else:
                raise NotImplementedError('Not implemented yet, must be albumentation based transform')
        self.timers['get/augmentation'].toc()

        self.timers['get/preprocessing'].tic()
        if self.preprocessing:
            if target is not None:
                target = torch.tensor(target).float()
            output = self.preprocessing(**output)
        self.timers['get/preprocessing'].toc()

        self.timers['get/finish'].tic()
        output['index'] = index
        output['image_id'] = img_id
        if target is not None:
            output['target'] = target
        self.timers['get/finish'].toc()

        if self.debug:
            for name, t in self.timers.items():
                print(name, t.average_time_str, t.total_time_str)

        return output

    def __len__(self):
        return self._num_images
