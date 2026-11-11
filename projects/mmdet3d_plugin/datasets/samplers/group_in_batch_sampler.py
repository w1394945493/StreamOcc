# https://github.com/Divadi/SOLOFusion/blob/main/mmdet3d/datasets/samplers/infinite_group_each_sample_in_batch_sampler.py
import itertools
import copy

import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info
from torch.utils.data.sampler import Sampler

# https://github.com/open-mmlab/mmdetection/blob/3b72b12fe9b14de906d1363982b9fba05e7d47c1/mmdet/core/utils/dist_utils.py#L157
def sync_random_seed(seed=None, device="cuda"):
    """Make sure different ranks share the same seed.
    All workers must call this function, otherwise it will deadlock.
    This method is generally used in `DistributedSampler`,
    because the seed should be identical across all processes
    in the distributed group.
    In distributed sampling, different ranks should sample non-overlapped
    data in the dataset. Therefore, this function is used to make sure that
    each rank shuffles the data indices in the same order based
    on the same seed. Then different ranks could use different indices
    to select non-overlapped data from the same data list.
    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.
    Returns:
        int: Seed to be used.
    """
    # 分布式进程共用序列组排列的随机种子；rank 0 广播给其他进程。
    # 此处同步的是组排列的种子，不是下面 NumPy 跳帧/反序/增强的全部随机状态。
    if seed is None:
        seed = np.random.randint(2**31)
    assert isinstance(seed, int)

    rank, world_size = get_dist_info()

    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()


class GroupInBatchSampler(Sampler):
    """
    Pardon this horrendous name. Basically, we want every sample to be from its own group.
    If batch size is 4 and # of GPUs is 8, each sample of these 32 should be operating on
    its own group.

    Shuffling is only done for group order, not done within groups.
    """

    def __init__(
        self,
        dataset,
        batch_size=1,
        world_size=None,
        rank=None,
        seed=0,
        skip_prob=0.5,
        sequence_flip_prob=0.1,
    ):
        # 1. 确定当前进程与总进程数；batch_size 是每个进程的批次大小。
        _rank, _world_size = get_dist_info()
        if world_size is None:
            world_size = _world_size
        if rank is None:
            rank = _rank

        self.dataset = dataset
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank
        self.seed = sync_random_seed(seed)

        self.size = len(self.dataset)

        # 2. dataset.flag 为每帧标明所属序列/子序列；序列切分在 dataset 中完成。
        # 这里要求组编号是从 0 开始的连续整数，否则 bincount 会产生空组。
        assert hasattr(self.dataset, "flag")
        self.flag = self.dataset.flag #* dataset.flag: 数据集提前建立的一个数组：数组位置表示样本索引，数组值表示该样本所属的组号
        self.group_sizes = np.bincount(self.flag) #* 统计每个组号出现了多少次，也就是每组有多少帧：
        self.groups_num = len(self.group_sizes)   #* 有多少组
        self.global_batch_size = batch_size * world_size
        # 全局槽位数 = 每进程 batch_size × 进程数，至少需有这么多序列组。
        assert self.groups_num >= self.global_batch_size

        #*----------------------------------------------------------------------#
        #* 根据dataset.flag找出每组有哪些帧，这里“组”表示同一场景序列，或者从该序列切出来的子序列。
        #* 得到 “组号 → 该组所有帧的数据集索引” 字典。
        # 3. 预先建立“组号 → 帧索引列表”，避免每次取帧重新搜索整个数据集。
        # np.where 返回按数据集索引递增的顺序；本类不会按 timestamp 重新排序。
        # 因此组内时序正确的前提是 dataset 已将各帧按时间整理好。
        self.group_idx_to_sample_idxs = {
            group_idx: np.where(self.flag == group_idx)[0].tolist() # 找出“属于某个组”的所有样本索引。
            for group_idx in range(self.groups_num)
        }   #* 数据集按时间排序，这里np.where() 得到的升序索引，是建立在数据已经按时间排序的基础上的。

        # 4. 每个 batch 槽位拥有独立的序列组生成器。
        # 例如两卡、每卡两帧：rank 0 管全局槽位 0/1，rank 1 管槽位 2/3。
        # 同一个槽位跨迭代继续读取同一子序列，而不是每次随机换一个单帧。
        self.group_indices_per_global_sample_idx = [
            self._group_indices_per_global_sample_idx(
                self.rank * self.batch_size + local_sample_idx
            )
            for local_sample_idx in range(self.batch_size)
        ]

        #* 给每个槽位创建一个独立的帧缓存
        # 5. 为每个本地槽位维护“当前组剩余帧”和“该组的增强配置”。
        # buffer 中保存的是索引，不是已加载的图像，也不是模型的历史特征缓存。
        self.buffer_per_local_sample = [[] for _ in range(self.batch_size)]
        self.aug_per_local_sample = [None for _ in range(self.batch_size)]
        self.skip_prob = skip_prob
        self.sequence_flip_prob = sequence_flip_prob

    def _infinite_group_indices(self):
        # 不断产生新的组号随机排列；只打乱组的顺序，不打乱组内部帧的顺序。
        # 各槽位创建的生成器种子相同，因此拥有同一条概念上的无限组号流。
        g = torch.Generator()
        g.manual_seed(self.seed)
        while True:
            yield from torch.randperm(self.groups_num, generator=g).tolist()

    def _group_indices_per_global_sample_idx(self, global_sample_idx):
        # 从公共组号流中按全局槽位数进行步进分配：
        # 若全局有 4 个槽位，槽位 0 取流位置 0/4/8，槽位 1 取 1/5/9，以此类推。
        # 分配的是流中的位置；不同组长度不同，各槽位会独立决定何时切换组。
        # 无限流会重复各组，因此这不是全训练期间“同一组绝不重复”的保证。
        yield from itertools.islice(
            self._infinite_group_indices(),
            global_sample_idx,
            None,
            self.global_batch_size,
        )

    def __iter__(self):
        # 【四个接口的分工：本项目采用 map-style Dataset】
        # 1) Sampler：通常每次产生一个样本索引，只决定“取谁、按什么顺序取”。
        # 2) BatchSampler：每次产生一批索引。此类虽然继承 Sampler，实际由 builder.py
        #    作为 batch_sampler 传给 DataLoader，已经自行完成组批，无需再包一层。
        # 3) Dataset.__getitem__(index)：读取单个样本并执行 pipeline。这里 index 是
        #    {"idx": 帧索引, "aug_config": 增强参数}，Dataset 会拆出这两个字段。
        # 4) DataLoader：消费批索引，调度 Dataset 取样，用 collate_fn 合并成模型输入。
        #    num_workers>0 时取样通常在 worker 进程执行，采样器在主进程生成索引。
        #
        # 【iter / next / yield 的执行关系】
        # 因为函数中含 yield，iter(batch_sampler) 创建生成器，并不立即执行函数体。
        # DataLoader 内部对该生成器调用 next()，才开始执行或恢复以下代码。
        # 概念流程（省略多进程调度）：
        #   index_iter = iter(batch_sampler)
        #   batch_indices = next(index_iter)
        #   samples = [dataset[index] for index in batch_indices]
        #   batch = collate_fn(samples)
        # 训练循环拿到的是 DataLoader 返回的 batch 数据，不是这里的索引列表。
        # 多进程预取会提前请求多个索引批次，并非模型每完成一步才严格调用一次 next()。
        #
        # 数据流示例（不跳帧、不反序）：
        # 第一次 yield：[A0, B0]；第二次：[A1, B1]；第三次：[A2, B2]。
        # 若 A 先结束，槽位 0 换成 C0，槽位 1 仍继续 B，其他槽位无需等待。
        # 本类只生成数据索引；历史特征更新、梯度 detach、无效历史处理由模型负责。
        # 迭代器无限产生批次，实际停止位置由 IterBasedRunner 的 max_iters 控制。
        # 【while：每轮生成一批索引，而不是在这里完成一个 epoch 的训练】
        # 循环不会自行耗尽：某槽位序列读完后会换组，外层 Runner 控制训练结束。
        while True:
            # 仅新建本批输出列表；self 中的剩余帧 buffer 和增强配置继续保留。
            curr_batch = []
            # 阶段一：分别推进本进程的每个 batch 槽位，构造一个完整批次。
            for local_sample_idx in range(self.batch_size):
                # 1. 以 skip_prob 决定是否额外丢弃一帧；必须至少剩两帧，
                # 才能保证丢弃后仍可输出一帧。该判断发生在补充新组之前，
                # 因此空 buffer 新取入一个组时，本次不会跳过它的首帧。
                skip = (
                    np.random.uniform() < self.skip_prob
                    and len(self.buffer_per_local_sample[local_sample_idx]) > 1
                )
                if len(self.buffer_per_local_sample[local_sample_idx]) == 0:
                    # 2. 当前组读完：从此槽位的组号流取下一个组，复制其帧列表。
                    # 后续 pop 会修改副本，不影响 group_idx_to_sample_idxs 中的原始列表。
                    # 这里的 next() 消费的是“该槽位的组号生成器”，只取一个新组号；
                    # 与 DataLoader 对本 __iter__ 生成器调用 next() 不是同一层迭代。
                    new_group_idx = next(
                        self.group_indices_per_global_sample_idx[
                            local_sample_idx
                        ]
                    )
                    self.buffer_per_local_sample[local_sample_idx] = copy.deepcopy(self.group_idx_to_sample_idxs[new_group_idx])
                    # 3. 可选的时间反序：将整组帧倒着读取，不是图像的左右翻转。
                    # 概率为 0 时保持 dataset 的正向顺序。
                    if np.random.uniform() < self.sequence_flip_prob:
                        self.buffer_per_local_sample[
                            local_sample_idx
                        ] = self.buffer_per_local_sample[local_sample_idx][
                            ::-1
                        ]
                    # 4. 序列一致增强：只在进入新组时生成一次配置，组内后续帧复用。
                    # 这样可避免每帧随机改变几何增强，破坏时序特征之间的对应关系。
                    if self.dataset.keep_consistent_seq_aug:
                        self.aug_per_local_sample[
                            local_sample_idx
                        ] = self.dataset.get_augmentation()

                # 5. 如果关闭序列一致增强，每次输出一帧前重新生成增强参数。
                if not self.dataset.keep_consistent_seq_aug:
                    self.aug_per_local_sample[
                        local_sample_idx
                    ] = self.dataset.get_augmentation()

                # 6. 执行跳帧：丢弃队首一帧，然后再取下一帧作为本次训练输入。
                if skip:
                    self.buffer_per_local_sample[local_sample_idx].pop(0)
                # 7. 输出给 dataset 的不是单一整数，而是包含帧索引和增强参数的字典。
                # 真正的图像/标注读取及增强操作在 dataset/pipeline 中执行。
                curr_batch.append(
                    dict(
                        idx=self.buffer_per_local_sample[local_sample_idx].pop(
                            0
                        ),
                        aug_config=self.aug_per_local_sample[local_sample_idx],
                    )
                )

            # 阶段二：交出当前批次，长度为本进程 batch_size。下一次迭代继续消费 buffer。
            # 本类虽继承 Sampler，但在 DataLoader 中承担 batch_sampler 的职责。
            # 【yield：交出本批索引并暂停，不退出函数，也不清空序列状态】
            # 到这里时，各槽位已经 pop 掉本次使用的帧，剩余帧留在 self 的 buffer 中。
            # DataLoader 取得 curr_batch 后调度 Dataset.__getitem__ 并合并数据。
            # 注意 Dataset 可能因空标注重选样本；这是 Dataset 的逻辑，不是 yield 在换帧。
            yield curr_batch
            # 【恢复位置】下一次 next(index_iter) 从 yield 后面继续，走到 while 尾部，
            # 再进入下一轮并创建新的 curr_batch；不会从 __init__ 或函数开头重新执行。
            # 因此同一槽位可依次读 A0、A1、A2，模型据此在该槽位上延续历史状态。

    def __len__(self):
        """Length of base dataset."""
        # 返回原始数据集帧数作长度参考；并不意味着 __iter__ 在此数量后会终止，
        # 也不是考虑跳帧后精确的批次数。
        return self.size

    def set_epoch(self, epoch):
        # 仅记录 epoch；本实现不会按 epoch 重新设种子，也不会清空已有序列 buffer。
        self.epoch = epoch
