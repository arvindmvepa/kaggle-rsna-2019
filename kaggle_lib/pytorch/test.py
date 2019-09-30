import sys
sys.path.append("/opt/Scattershot/libs/kaggle-rsna-2019")
from kaggle_lib.pytorch.datacatalog import get_dataset, dataset_map, datacatalog, get_csv_file
from torch.utils.data import DataLoader
from kaggle_lib.pytorch.get_model import get_model
from kaggle_lib.pytorch.augmentation import get_preprocessing
from joblib import Parallel, delayed

import albumentations as A
import json
from kaggle_lib.pytorch.utils import Timer

def run_batch(i, batch_size, train_dataset):
    nimages = len(train_dataset)
    batcher = list(range(nimages))
    batch = batcher[i * batch_size:(i + 1) * batch_size]
    for x in batch:
        train_dataset[x]


def test(h='lambda2', ds='rsna2019-stage1',
         batch_size=32, shuffle=True, pin_memory=False, small=True, N = 15, use_dataloader=True, use_joblib=False,
         joblib_backend='loky',
         num_workers=0,
         use_transforms=True,
         limit=None):
    # h = 'lambda2'
    # ds = 'rsna2019-stage1'
    # pin_memory = True
    # batch_size = 32
    # use_dataloader = False

    img_ids = None
    if small and limit is None:
        img_ids = ["ID_0012b1611", "ID_0029cbda4", "ID_002e47842", "ID_0033c33fb", "ID_0040ec4c2", "ID_004260f12", "ID_00530a1c6", "ID_005412cf3", "ID_00566362f", "ID_0057098ae", "ID_0066fb0a6", "ID_006c4e834", "ID_006e4394d", "ID_0079dd17a", "ID_008afa33d", "ID_009b9e8d8", "ID_009c1392d", "ID_00b6a477a", "ID_00cdcb263", "ID_00d29eb72", "ID_00dd97d43", "ID_00e891aef", "ID_00f38e20d", "ID_010111261", "ID_010d6c71a", "ID_0110a750e", "ID_0115811fa", "ID_0119318cc", "ID_012a710a1", "ID_012eed395", "ID_012fde3f8", "ID_01302c481", "ID_013e3784a", "ID_0144ddd8c", "ID_014c0038d", "ID_0152232a7", "ID_0184b24fa", "ID_018ea7a03", "ID_01922dde5", "ID_0198309a9", "ID_0199b7da5", "ID_01b8422c0", "ID_01b88a072", "ID_01bafe2ec", "ID_01ce31b90", "ID_01dba213d", "ID_01e0ee640", "ID_01f00619b", "ID_01f9aef6f", "ID_01f9de7d7", "ID_020c3278f", "ID_021b4a7ce", "ID_021e705f6", "ID_022c8db40", "ID_024249eb6", "ID_0242fe7fb", "ID_0244d72b7", "ID_024d8dae0", "ID_024f430e6", "ID_0256b834d", "ID_0266c755e", "ID_0266e96e7", "ID_02705cf7e", "ID_02753d81e", "ID_0280ebdd7", "ID_028c75b1e", "ID_02998e7ff", "ID_029f70def", "ID_02a41541e", "ID_02a5fe53b", "ID_02ae19dfc", "ID_02b6ff5b8", "ID_02c06dd86", "ID_02c443bda", "ID_02cb68e3d", "ID_02cc03f6c", "ID_02e1552ef", "ID_02e2b61b3", "ID_02ececb07", "ID_02f307e9e", "ID_0306e5de5", "ID_0306fde13", "ID_03079fbc5", "ID_030c1a93f", "ID_032168597", "ID_032871209", "ID_0364d50c5", "ID_036daa1a7", "ID_037506971", "ID_037e7c6a5", "ID_037ecd3d0", "ID_0388171b3", "ID_038a0cd84", "ID_0392eb1c1", "ID_0397a14f4", "ID_039bfae27", "ID_03a73163c", "ID_03a7643d7", "ID_03b1afa64", "ID_03d09a412", "ID_03d6bc47f", "ID_03da9910e", "ID_03e4b34e8", "ID_03e55ddc9", "ID_03e60b208", "ID_03ebf1688", "ID_03f0a4032", "ID_03f8317df", "ID_0401fc5e0", "ID_040688f85", "ID_0415835d7", "ID_042285ea7", "ID_0423e4257", "ID_042b07d0c", "ID_04301cc70", "ID_0434e822f", "ID_044110b58", "ID_044474a5e", "ID_044b9aa5d", "ID_0451f8caa", "ID_0453e3512", "ID_0454bd375", "ID_04621a907", "ID_046541b74", "ID_0466a6e9b", "ID_0466d691e", "ID_046c0d8f1", "ID_04743df26", "ID_047b9e4e0", "ID_047f1a535", "ID_048404ad5", "ID_04a40f460", "ID_04a57cdb3", "ID_04a9f1ddd", "ID_04ad991e7", "ID_04d06c3a5", "ID_04df5fd68", "ID_04ed9bf42", "ID_04ef80edb", "ID_050523f24", "ID_05101b635", "ID_0518092ba", "ID_0521a637b", "ID_052e27134", "ID_053958ffc", "ID_053b860dd", "ID_054f8a333", "ID_054f8cdfe", "ID_0550ac742", "ID_055f33b69", "ID_05699b5d7", "ID_05783ef33", "ID_058afb61d", "ID_059674a1b", "ID_059a37437", "ID_05a0bf6aa", "ID_05ac33dd7", "ID_05bb89b07", "ID_05bf8f360", "ID_05ce3bcf4", "ID_05cef6e1f", "ID_05e448435", "ID_05ffb5433", "ID_060092069", "ID_0603de917", "ID_06197fc8a", "ID_06312ec79", "ID_063d04dbf", "ID_06409ccfe", "ID_0643b4ae1", "ID_06556bdfa", "ID_066010eaf", "ID_06666d25b", "ID_0674fbd5e", "ID_0677dbb33", "ID_068affe4e", "ID_06965a90b", "ID_06998c4b9", "ID_06a49313b", "ID_06aa44340", "ID_06ab62da9", "ID_06b62697c", "ID_06b7cf9df", "ID_06bcadfd0", "ID_06c23b8b5", "ID_06c9393d1", "ID_06d14e858", "ID_06d3b3897", "ID_06dee244f", "ID_06eb0e66c", "ID_06f75b771", "ID_070daab6e", "ID_070f40cc7", "ID_071333cfb", "ID_0719d4be1", "ID_071b1a476", "ID_0727d2497", "ID_072c04bb4", "ID_073fdf4e4", "ID_073ff5686", "ID_0754f0ba7", "ID_07598c3b2", "ID_075bf31d5", "ID_07678e57b", "ID_077373ae0", "ID_0773816f7", "ID_0778e408d", "ID_0787ef51d", "ID_0799558ea", "ID_079c526b2", "ID_07b2325fd", "ID_07b3e1951", "ID_07b63570b", "ID_07bc56d44", "ID_07c5827e5", "ID_07cefaf79", "ID_07d5efeab", "ID_07d914f5f", "ID_07e396e49", "ID_07e5f1dad", "ID_07e820dda", "ID_07ea770b2", "ID_07ec42327", "ID_07eff66b0", "ID_07ff6573c", "ID_080e55858", "ID_08171ed52", "ID_081965805", "ID_081ea2220", "ID_082b03a5e", "ID_083188e3b", "ID_083902813", "ID_084379495", "ID_08657edb7", "ID_0865fcbed", "ID_087208a31", "ID_088087420", "ID_088136e35", "ID_088c95e54", "ID_0899553ab", "ID_08a4b86ac", "ID_08a94bd98", "ID_08aae07e1", "ID_08bfbf01e", "ID_08cf89f6a", "ID_08d576bec", "ID_08d6cb0d4", "ID_08da8b6de", "ID_08df32924", "ID_08ed79d0a", "ID_08fbf5968", "ID_090322c5d", "ID_09041b2d8", "ID_0922dd7b6", "ID_0926b36ed", "ID_093931b23", "ID_0946cfa5c", "ID_094b57e71", "ID_094bdd994", "ID_095de695f", "ID_095e2aad9", "ID_09617c755", "ID_096b05798", "ID_096eacffe", "ID_0978305fb", "ID_0983ad15a", "ID_098450769", "ID_09a809353", "ID_09cf0ceab", "ID_09e0aa4e0", "ID_09e32f86c", "ID_09f428578", "ID_0a034e2db", "ID_0a0697fda", "ID_0a1282cf4", "ID_0a133af0b", "ID_0a2555e23", "ID_0a2f1cd26", "ID_0a328eb8c", "ID_0a3605ab1", "ID_0a36fc5f6", "ID_0a3a21d0d", "ID_0a3e35852", "ID_0a3e4b975", "ID_0a4147836", "ID_0a4ad0cba", "ID_0a662f025", "ID_0a68f5b4e", "ID_0a6e7bd5c", "ID_0a8447585", "ID_0a860b26d", "ID_0a89e0bd3", "ID_0a8e8a7f7", "ID_0aab0b72b", "ID_0aae2db1b", "ID_0ab18db3d", "ID_0abfcce29", "ID_0ac2a13bd", "ID_0ac7e65b7", "ID_0ad1f4f41", "ID_0aeed5483", "ID_0af6738d1", "ID_0b253de25", "ID_0b42d4eb5", "ID_0b4ce9fd4", "ID_0b5790597", "ID_0b7227779", "ID_0b78160e2", "ID_0b84dc98f", "ID_0b91a2684", "ID_0b998ae70", "ID_0bb49df46", "ID_0bbda34c8", "ID_0bc8a35c5", "ID_0bd4eaa89", "ID_0be362633", "ID_0be5c8278", "ID_0be659d92", "ID_0c02a053f", "ID_0c06283d8", "ID_0c1d6c209", "ID_0c2105f6c", "ID_0c2534ca4", "ID_0c2ec927c", "ID_0c346f539", "ID_0c412ba3c", "ID_0c44e28b1", "ID_0c4caf805", "ID_0c5466649", "ID_0c5f64453", "ID_0c600b0fa", "ID_0c61b3c45", "ID_0c636eba5", "ID_0c6f9f09e", "ID_0c78b074c", "ID_0c78ff097", "ID_0c881117b", "ID_0c8c31e57", "ID_0c8e8958c", "ID_0ca52e828", "ID_0ca835f60", "ID_0cb2dee34", "ID_0cbb735cd", "ID_0cc3a3039", "ID_0cc9a113d", "ID_0ccd3cd7a", "ID_0ce41f17f", "ID_0d0d2c928", "ID_0d19ea02c", "ID_0d227383d", "ID_0d24c1a89", "ID_0d2a747b5", "ID_0d3caf00a", "ID_0d4fe8907", "ID_0d54a6258", "ID_0d55b6b08", "ID_0d5fd059a", "ID_0d7baeb39", "ID_0d7da7c4f", "ID_0d8c90431", "ID_0d98c33c1", "ID_0d9caea0a", "ID_0da04ebc3", "ID_0daa739d6", "ID_0db2543b1", "ID_0db358436", "ID_0dc14755b", "ID_0dd62b7cd", "ID_0ddaa0725", "ID_0ddda8a0b", "ID_0df260dda", "ID_0df40bc05", "ID_0dfaf5dc3", "ID_0e025ea6e", "ID_0e09033c6", "ID_0e09fe613", "ID_0e0b086c1", "ID_0e1bef923", "ID_0e1eea48b", "ID_0e229f788", "ID_0e4aae42e", "ID_0e4cdd983", "ID_0e5243525", "ID_0e5846d16", "ID_0e62b7fef", "ID_0e6466e9e", "ID_0e69fc170", "ID_0e71459ff", "ID_0e719dc58", "ID_0e7a9a3ff", "ID_0e80f1b8f", "ID_0e833f159", "ID_0e9b9510e", "ID_0ea04eda4", "ID_0eb7fa79a", "ID_0eb92a230", "ID_0ebd689bc", "ID_0ec0c9d66", "ID_0ec0ecbd2", "ID_0ec9ec1f4", "ID_0eca906b9", "ID_0ee5bc77f", "ID_0eef97c45", "ID_0ef1fe9e5", "ID_0ef47a218", "ID_0efa070f3", "ID_0efebb9e3", "ID_0f0ec32b3", "ID_0f10e9b01", "ID_0f173600f", "ID_0f1c46b88", "ID_0f2308d16", "ID_0f2366d37", "ID_0f281ffc1", "ID_0f2da33fa", "ID_0f43308f1", "ID_0f487db67", "ID_0f55d6c4b", "ID_0f57df633", "ID_0f59f6913", "ID_0f6206b4d", "ID_0f69f6936", "ID_0f6fb2026", "ID_0f7080f41", "ID_0f7a502dd", "ID_0f7d34991", "ID_0f80bfe8f", "ID_0f8494464", "ID_0f856b98b", "ID_0f8b4cbf7", "ID_0f954ceed", "ID_0fa9d9ba5", "ID_0fb996be5", "ID_0fbd1f0db", "ID_0fbe1b020", "ID_0fbe4bb53", "ID_0feeecadc", "ID_0ff454d1e", "ID_0ffdc7b8d", "ID_102b4d202", "ID_102c5eaa4", "ID_102e9d995", "ID_1039d5cd2", "ID_1044bb5f9", "ID_104f5c86b", "ID_105126269", "ID_1058e6d4c", "ID_10680b1bf", "ID_106b804ee", "ID_1076e2148", "ID_107a7f332", "ID_10846086f", "ID_108652b88", "ID_10934c42b", "ID_109648ace", "ID_109bc3a0e", "ID_109cf9d95", "ID_109f99a38", "ID_10b1b0e55", "ID_10b58ea40", "ID_10be567aa", "ID_10bfe6767", "ID_10c158c17", "ID_10ca69e0a", "ID_10d1e694b", "ID_10d214381", "ID_10e2fe26d", "ID_10f7993b3", "ID_10f9fe44d", "ID_10ff196ec", "ID_1115885cd", "ID_111659347", "ID_111ea636e", "ID_1120c1e99", "ID_11215171c", "ID_11220282d", "ID_112574b43", "ID_11294c843", "ID_11368e68d", "ID_114bbeefa"]

    transforms=None
    if use_transforms:
        s = """{"__version__": "0.3.3", "transform": {"__class_fullname__": "albumentations.core.composition.Compose", "p": 1.0, "transforms": [{"__class_fullname__": "albumentations.augmentations.transforms.Resize", "always_apply": false, "p": 1, "height": 256, "width": 256, "interpolation": 1}, {"__class_fullname__": "albumentations.augmentations.transforms.RandomScale", "always_apply": false, "p": 0.5, "interpolation": 1, "scale_limit": [-0.19999999999999996, 0.19999999999999996]}, {"__class_fullname__": "albumentations.augmentations.transforms.Rotate", "always_apply": false, "p": 0.5, "limit": [-10, 10], "interpolation": 1, "border_mode": 4, "value": null, "mask_value": null}, {"__class_fullname__": "albumentations.augmentations.transforms.PadIfNeeded", "always_apply": false, "p": 1.0, "min_height": 224, "min_width": 224, "border_mode": 4, "value": null, "mask_value": null}, {"__class_fullname__": "albumentations.augmentations.transforms.RandomCrop", "always_apply": false, "p": 0.5, "height": 224, "width": 224}, {"__class_fullname__": "albumentations.augmentations.transforms.PadIfNeeded", "always_apply": false, "p": 1.0, "min_height": 224, "min_width": 224, "border_mode": 4, "value": null, "mask_value": null}, {"__class_fullname__": "albumentations.augmentations.transforms.CenterCrop", "always_apply": false, "p": 1.0, "height": 224, "width": 224}, {"__class_fullname__": "kaggle_lib.pytorch.augmentation.ChannelWindowing", "always_apply": false, "p": 1.0, "windows": ["soft_tissue", "f1000", "min_max"], "force_rgb": true, "min_pixel_value": 0, "max_pixel_value": 255}, {"__class_fullname__": "albumentations.augmentations.transforms.ToFloat", "always_apply": false, "p": 1.0, "max_value": 1.0}], "bbox_params": null, "keypoint_params": null, "additional_targets": {}}}"""
        tconfig = json.loads(s)
        transforms = A.from_dict(tconfig)

    model_params = {'encoder': 'resnet50',
                    'nclasses': 6,
                    'encoder_weights': 'imagenet',
                    'activation': 'sigmoid',
                    'model_dir': '/data/pretrained_weights/',
                    'classifier': 'srn'}



    model, model_preprocessing = get_model(**model_params)

    train_c =  datacatalog[dataset_map[ds]['train']]
    train_dataset = get_dataset(train_c, '/data', transforms=transforms,
                                preprocessing=get_preprocessing(model_preprocessing), debug=False, img_ids=img_ids,
                                limit=limit)

    import tqdm
    timer = Timer()
    timer.tic()
    if use_dataloader:
        dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                        pin_memory=pin_memory)
        tbar = tqdm.tqdm(dl, desc=h + '-' + ds + '-withloader-len{}-nworkers{}'.format(len(train_dataset), num_workers))
        for i, x in enumerate(tbar):
            s = ', '.join('{}=avg:{},t:{}'.format(name, t.average_time_str, t.total_time_str) for name, t in
                          dl.dataset.timers.items())
            print("index: {}".format(i))
            print("time string: {}".format(s))
            tbar.set_postfix_str(s)
            if i > N:
                break
        tbar.close()

    elif use_joblib:
        tbar = tqdm.tqdm(list(range(N)), desc=h + '-' + ds + '-joblib-len{}-nworkers{}'.format(len(train_dataset),
                                                                                                   num_workers))
        Parallel(n_jobs=num_workers, backend=joblib_backend)(delayed(run_batch)(i, batch_size, train_dataset)
                                                             for i in tbar)
        tbar.close()

    else:
        tbar = tqdm.tqdm(list(range(N)), desc=h + '-' + ds + '-noloader-len{}'.format(len(train_dataset)))
        nimages = len(train_dataset)
        batcher = list(range(nimages))
        for i in tbar:
            # Local batches and labels
            batch = batcher[i*batch_size:(i + 1)*batch_size]
            for x in batch:
                train_dataset[x]
            s = ', '.join('{}=avg:{},t:{}'.format(name, t.average_time_str, t.total_time_str) for name, t in
                          train_dataset.timers.items())
            tbar.set_postfix_str(s)
        tbar.close()
    timer.toc()

    print("Total Time: {}".format(timer.total_time_str))

#slow
#for limit in [674258, 168564, 42141, 10535, 2633]:
for limit in [479, None]:
    test(use_dataloader=False, use_joblib=True, joblib_backend='threading', small=False, N=10, limit=limit,
         num_workers=4)
    print()
    print()

#fast
#for limit in [674258, 168564, 42141, 10535, 2633]:
#test(use_dataloader=False, use_joblib=True, small=True, N=10, num_workers=4)
#print()
#print()
# test(use_dataloader=False, small=False)
