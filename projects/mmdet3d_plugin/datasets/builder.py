import copy
import platform
import random
from functools import partial

import numpy as np
from mmcv.parallel import collate
from mmcv.runner import get_dist_info
from mmcv.utils import Registry, build_from_cfg
from torch.utils.data import DataLoader

# MMDetection 的普通分组采样器：组内随机打乱，按 samples_per_gpu 补齐并排列索引。
# 它逐个返回整数索引，不维护 batch 槽位的跨迭代时间连续性。
from mmdet.datasets.samplers import GroupSampler
from projects.mmdet3d_plugin.datasets.samplers import (
    # 本项目的流式批采样器：每个槽位独立推进序列，直接返回一批索引/增强配置。
    GroupInBatchSampler,
    DistributedGroupSampler,
    DistributedSampler,
    build_sampler
)


def build_dataloader(
    dataset,
    samples_per_gpu,
    workers_per_gpu,
    num_gpus=1,
    dist=True,
    shuffle=True,
    seed=None,
    shuffler_sampler=None,
    nonshuffler_sampler=None,
    runner_type="EpochBasedRunner",
    skip_prob = 0.5,
    sequence_flip_prob = 0.1,
    save_results = None,
    **kwargs
):
    """Build PyTorch DataLoader.
    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.
    Args:
        dataset (Dataset): A PyTorch dataset.
        samples_per_gpu (int): Number of training samples on each GPU, i.e.,
            batch size of each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        num_gpus (int): Number of GPUs. Only used in non-distributed training.
        dist (bool): Distributed training/test or not. Default: True.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Default: True.
        kwargs: any keyword argument to be used to initialize DataLoader
    Returns:
        DataLoader: A PyTorch dataloader.
    """
    rank, world_size = get_dist_info()
    # 两种组批模式二选一：普通 sampler 逐个给索引，batch_sampler 直接给一批索引。
    # 此变量默认为空；只有下面 IterBasedRunner 分支使用自定义批采样器。
    batch_sampler = None
    # 选择优先级：先看 runner_type，再看 dist/shuffle，并不是按“是否单卡”直接二选一。
    # 当前 customs/StreamOcc.py 的 runner.type=IterBasedRunner，因此训练使用本分支。
    # 该分支不根据 shuffle 改变模式，组顺序、跳帧和反序均由 GroupInBatchSampler 控制。
    if runner_type == 'IterBasedRunner':
        # GroupInBatchSampler：同一槽位连续消费一个序列/子序列，以匹配模型历史缓存。
        # 例如批次 [A0,B0] → [A1,B1]；每个槽位读完自己的组后才换组。
        # dataset.flag 表示序列组；每次 yield 一个列表，元素包含 idx 和 aug_config。
        # 与普通 GroupSampler 相比，关键目标是跨批次的槽位时序连续，而非批内同组。
        #* 槽位：是一个batch 中固定的位置**；“组”就是 dataset.flag 划分的一个序列或子序列。
        #* 采样器内部为每个槽位维护一个剩余帧列表：
        print("Use GroupInBatchSampler !!!")
        batch_sampler = GroupInBatchSampler(
            dataset,
            samples_per_gpu,
            world_size,
            rank,
            seed=seed,
            skip_prob=skip_prob,
            sequence_flip_prob=sequence_flip_prob,
        )
        # 自定义 batch_sampler 已按 samples_per_gpu 组批，DataLoader 不再二次组批。
        # batch_size=1 是 PyTorch 接口允许的默认占位值，不代表实际每批只有一个样本。
        # 启用 batch_sampler 时，sampler 必须为空，也不能另设 shuffle/drop_last。
        batch_size = 1
        sampler = None
        num_workers = workers_per_gpu
    elif dist:
        # 非 IterBasedRunner 的分布式分支：默认选 DistributedGroupSampler（shuffle=True）
        # 或 DistributedSampler（shuffle=False），不是下面的单进程 GroupSampler。
        # DistributedGroupSampler will definitely shuffle the data to satisfy
        # that images on each GPU are in the same group
        if shuffle:
            print("Use DistributedGroupSampler !!!")
            sampler = build_sampler(
                shuffler_sampler
                if shuffler_sampler is not None
                else dict(type="DistributedGroupSampler"),
                dict(
                    dataset=dataset,
                    samples_per_gpu=samples_per_gpu,
                    num_replicas=world_size,
                    rank=rank,
                    seed=seed,
                ),
            )
        else:
            sampler = build_sampler(
                nonshuffler_sampler
                if nonshuffler_sampler is not None
                else dict(type="DistributedSampler"),
                dict(
                    dataset=dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=shuffle,
                    seed=seed,
                ),
            )

        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        # assert False, 'not support in bevformer'
        print("WARNING!!!!, Only can be used for obtain inference speed!!!!")
        # 仅在非 IterBasedRunner、dist=False、shuffle=True 时使用 GroupSampler。
        # 它按 dataset.flag 分组，打乱组内帧，重复抽样补齐到 samples_per_gpu 的倍数，
        # 再打乱这些同组小批的顺序，最后逐个输出整数索引，由 DataLoader 组批。
        # 通用检测中 flag 常表示图像宽高比分组；这里的含义仍取决于 dataset 的定义。
        # 即便 flag 表示同一序列，它也不保持帧的时间顺序，不适合直接替代流式采样器。
        # 多 GPU 非分布式模式下，“同组”保证针对每个 samples_per_gpu 小批，而非整个总批次。
        # shuffle=False 时传 None，交由 DataLoader 默认顺序取样；也不启用时序增强管理。
        sampler = GroupSampler(dataset, samples_per_gpu) if shuffle else None
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    init_fn = (
        partial(worker_init_fn, num_workers=num_workers, rank=rank, seed=seed)
        if seed is not None
        else None
    )

    # 统一入口保留两个关键字，但不会同时传入两个非空采样器：
    # 1. IterBasedRunner：sampler=None，batch_sampler=GroupInBatchSampler 实例。
    #    每次直接获得一批索引，实际批大小由 GroupInBatchSampler 控制。
    # 2. 其他分支：batch_sampler=None，sampler 为普通/分布式采样器或 None。
    #    DataLoader 内部再用 BatchSampler 按 batch_size 将单个索引组成一批；
    #    sampler 也为 None 时，DataLoader 在未启用 shuffle 的情况下使用顺序采样器。
    # 因此不是“这两个都需要”：它们对应两种可选方式，不能同时提供非空对象。
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
        pin_memory=False,
        worker_init_fn=init_fn,
        **kwargs
    )

    return data_loader


def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# Copyright (c) OpenMMLab. All rights reserved.
import platform
from mmcv.utils import Registry, build_from_cfg

from mmdet.datasets import DATASETS
from mmdet.datasets.builder import _concat_dataset

if platform.system() != "Windows":
    # https://github.com/pytorch/pytorch/issues/973
    import resource

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    base_soft_limit = rlimit[0]
    hard_limit = rlimit[1]
    soft_limit = min(max(4096, base_soft_limit), hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

OBJECTSAMPLERS = Registry("Object sampler")


def custom_build_dataset(cfg, default_args=None):
    try:
        from mmdet3d.datasets.dataset_wrappers import CBGSDataset
    except:
        CBGSDataset = None
    from mmdet.datasets.dataset_wrappers import (
        ClassBalancedDataset,
        ConcatDataset,
        RepeatDataset,
    )

    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset(
            [custom_build_dataset(c, default_args) for c in cfg]
        )
    elif cfg["type"] == "ConcatDataset":
        dataset = ConcatDataset(
            [custom_build_dataset(c, default_args) for c in cfg["datasets"]],
            cfg.get("separate_eval", True),
        )
    elif cfg["type"] == "RepeatDataset":
        dataset = RepeatDataset(
            custom_build_dataset(cfg["dataset"], default_args), cfg["times"]
        )
    elif cfg["type"] == "ClassBalancedDataset":
        dataset = ClassBalancedDataset(
            custom_build_dataset(cfg["dataset"], default_args),
            cfg["oversample_thr"],
        )
    elif cfg["type"] == "CBGSDataset":
        dataset = CBGSDataset(
            custom_build_dataset(cfg["dataset"], default_args)
        )
    elif isinstance(cfg.get("ann_file"), (list, tuple)):
        dataset = _concat_dataset(cfg, default_args)
    else:
        dataset = build_from_cfg(cfg, DATASETS, default_args)

    return dataset
