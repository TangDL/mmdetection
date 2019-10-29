#encoding=utf-8
import platform
from functools import partial

from mmcv.parallel import collate
from mmcv.runner import get_dist_info
from torch.utils.data import DataLoader

from .sampler import DistributedGroupSampler, DistributedSampler, GroupSampler

if platform.system() != 'Windows':
    # https://github.com/pytorch/pytorch/issues/973
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


def build_dataloader(dataset,
                     imgs_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     **kwargs):
    shuffle = kwargs.get('shuffle', True)
    if dist:                                     # 是否进行分布式训练-->否
        rank, world_size = get_dist_info()
        if shuffle:
            sampler = DistributedGroupSampler(dataset, imgs_per_gpu,
                                              world_size, rank)
        else:
            sampler = DistributedSampler(
                dataset, world_size, rank, shuffle=False)
        batch_size = imgs_per_gpu
        num_workers = workers_per_gpu
    else:                                               # 进入这一层，设定一些参数
        sampler = GroupSampler(dataset, imgs_per_gpu) if shuffle else None
        batch_size = num_gpus * imgs_per_gpu            # 定义batch_size=1*1
        num_workers = num_gpus * workers_per_gpu        # 定义workers=1*2

    data_loader = DataLoader(                           # 定义dataload，使用继承方法
        dataset,
        batch_size=batch_size,                          # 每次返回的图片数量
        sampler=sampler,                                # 抽取图片的方法
        num_workers=num_workers,                        # 进程数量
        collate_fn=partial(collate, samples_per_gpu=imgs_per_gpu),     # 抽取图片的具体函数，偏函数，设定传入函数的默认参数
        pin_memory=False,
        **kwargs)
    return data_loader
