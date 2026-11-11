# Copyright (c) Horizon Robotics. All rights reserved.
from typing import Optional

import torch

from mmdet.core.bbox.builder import BBOX_CODERS

from projects.mmdet3d_plugin.core.box3d import *


@BBOX_CODERS.register_module()
class SparseBox3DDecoder(object):
    # 检测推理后处理：选择高分预测并还原三维框编码；这里不是 Transformer 解码层。
    def __init__(
        self,
        num_output: int = 300,
        score_threshold: Optional[float] = None,
        sorted: bool = True,
    ):
        super(SparseBox3DDecoder, self).__init__()
        self.num_output = num_output
        self.score_threshold = 0.3
        self.sorted = sorted

    def decode_box(self, box):
        # 将 log 尺寸取 exp，将 sin/cos 朝向还原成 yaw，拼接中心和速度得到物理框参数。
        yaw = torch.atan2(box[:, SIN_YAW], box[:, COS_YAW])
        box = torch.cat(
            [
                box[:, [X, Y, Z]],
                box[:, [W, L, H]].exp(),
                yaw[:, None],
                box[:, VX:],
            ],
            dim=-1,
        )
        return box

    def decode(
        self,
        cls_scores,
        box_preds,
        instance_id=None,
        qulity=None,
        output_idx=-1,
    ):
        # 默认使用最后一轮预测：sigmoid 分类分数 → top-k → 可选质量重评分
        # → 阈值过滤 → 框参数解码，返回每个样本的框、分数和类别。
        squeeze_cls = instance_id is not None

        cls_scores = cls_scores[output_idx].sigmoid()

        if squeeze_cls:
            cls_scores, cls_ids = cls_scores.max(dim=-1)
            cls_scores = cls_scores.unsqueeze(dim=-1)

        box_preds = box_preds[output_idx]
        bs, num_pred, num_cls = cls_scores.shape
        cls_scores, indices = cls_scores.flatten(start_dim=1).topk(
            self.num_output, dim=1, sorted=self.sorted
        )
        if not squeeze_cls:
            cls_ids = indices % num_cls
        if self.score_threshold is not None:
            mask = cls_scores >= self.score_threshold

        if qulity is not None:
            centerness = qulity[output_idx][..., CNS]
            centerness = torch.gather(centerness, 1, indices // num_cls)
            cls_scores_origin = cls_scores.clone()
            cls_scores *= centerness.sigmoid()
            cls_scores, idx = torch.sort(cls_scores, dim=1, descending=True)
            if not squeeze_cls:
                cls_ids = torch.gather(cls_ids, 1, idx)
            if self.score_threshold is not None:
                mask = torch.gather(mask, 1, idx)
            indices = torch.gather(indices, 1, idx)

        output = []
        for i in range(bs):
            category_ids = cls_ids[i]
            if squeeze_cls:
                category_ids = category_ids[indices[i]]
            scores = cls_scores[i]
            box = box_preds[i, indices[i] // num_cls]
            if self.score_threshold is not None:
                category_ids = category_ids[mask[i]]
                scores = scores[mask[i]]
                box = box[mask[i]]
            if qulity is not None:
                scores_origin = cls_scores_origin[i]
                if self.score_threshold is not None:
                    scores_origin = scores_origin[mask[i]]

            box = self.decode_box(box)
            output.append(
                {
                    "boxes_3d": box.cpu(),
                    "scores_3d": scores.cpu(),
                    "labels_3d": category_ids.cpu(),
                }
            )
            if qulity is not None:
                output[-1]["cls_scores"] = scores_origin.cpu()
            if instance_id is not None:
                ids = instance_id[i, indices[i]]
                if self.score_threshold is not None:
                    ids = ids[mask[i]]
                output[-1]["instance_ids"] = ids
        return output
