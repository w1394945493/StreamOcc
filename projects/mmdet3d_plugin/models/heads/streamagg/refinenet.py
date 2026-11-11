# Copyright (c) Phigent Robotics. All rights reserved.
# ---------------------------------------------
#  Modified by Seokha Moon
# ---------------------------------------------

import torch
from torch import nn
from mmcv.cnn import build_conv_layer, build_upsample_layer
from mmdet.models import BACKBONES

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        # 论文式(3)：先按通道筛选信息，再按三维空间位置生成残差门控权重。
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        # 1. M_c：强调有用的特征通道。
        x = self.channel_attention(x)
        # 2. M_s：强调有用的空间位置；额外返回 sigmoid 前的响应供几何分支使用。
        x, spatial_out = self.spatial_attention(x)
        return x, spatial_out


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        # 论文通道注意力：全局平均/最大池化汇总 D、H、W，保留每个通道的统计。
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        # 共享的两层映射，用 1×1×1 卷积实现通道压缩与恢复。
        self.fc1 = nn.Conv3d(channels, channels // reduction, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv3d(channels // reduction, channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1. 两种全局统计经过同一组映射，再相加得到通道评分。
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        # 2. sigmoid 得到 M_c，广播乘回三维特征。
        return x * self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # 论文空间注意力：拼接沿通道的均值/最大值，以 3D 卷积生成单通道 M_s。
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1. 沿通道汇总，保留三维空间结构，得到两张单通道响应图。
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        # 2. sigmoid(M_s) 对特征逐位置加权；out 保留未归一化的空间响应。
        return x * self.sigmoid(out), out


@BACKBONES.register_module()
class RefineNet(nn.Module):
    def __init__(self, channels, internal_channels, use_occupied_head=False):
        super(RefineNet, self).__init__()
        # 论文 Fig.3 的瓶颈修正网络：1×1×1 降维 → 3×3×3 → 1×1×1 升维。
        # 默认配置 channels=256、internal_channels=64，对应 C → C/4 → C。
        self.conv1 = nn.Conv3d(channels, internal_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm3d(internal_channels)

        self.conv2 = nn.Conv3d(internal_channels, internal_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(internal_channels)

        self.conv3 = nn.Conv3d(internal_channels, channels, kernel_size=1)
        self.bn3 = nn.BatchNorm3d(channels)

        # 论文 Adaptive Residual Refinement 中的三维通道/空间注意力。
        self.cbam = CBAM(channels)
        self.relu = nn.ReLU()

        # 论文 Geometry Supervision：由空间响应预测二值占用，辅助学习几何门控。
        # 损失在 StreamOccHead 中计算；当前实现只按此开关执行，未额外判断 self.training。
        self.use_occupied_head = use_occupied_head
        if self.use_occupied_head:
            deconv_cfg = dict(type='deconv3d', bias=False)
            conv3d_cfg = dict(type='Conv3d', bias=False)
            up_sample = build_upsample_layer(deconv_cfg, 1, internal_channels, kernel_size=2, stride=2)
            self.up_block = nn.Sequential(up_sample,
                                          nn.BatchNorm3d(internal_channels),
                                          nn.ReLU(inplace=True))
            self.pred_occ_mask = build_conv_layer(
                            conv3d_cfg,
                            in_channels=internal_channels,
                            out_channels=1,
                            kernel_size=1,
                            stride=1,
                            padding=0)
    #* 对历史特征进行修正
    def forward(self, x):
        # 1. 输入 x 为已对齐的 V_warp，形状 [B,C,D,H,W]；保存残差直连。
        pred_occ_mask = None
        identity = x
        #* 瓶颈卷积：生成需要补充的修正量
        # 2. 瓶颈网络生成修正特征 V_out，学习补偿插值引入的失真。
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        #* 通道注意力+空间注意力：控制哪些修正量更重要
        # 3. 论文式(3)：依次施加 M_c 和 sigmoid(M_s)，对修正量进行自适应加权。
        out, spatial_out = self.cbam(out)
        #*---------------------------------------------------------#
        #* 使用真值二值占据掩码监督：使得历史特征
        #* 起到几何引导的作用
        # 4. 几何辅助预测：将空间响应上采样到监督分辨率，返回占用概率。
        # 这里返回的是 sigmoid 后的概率；描述的是当前实现，不改变外层损失接口。
        if self.use_occupied_head:
            up_out = self.up_block(spatial_out) #* 对应论文的Geometry Supervision
            # 用于训练时的“占用/空闲”二分类辅助监督，帮助 RefineNet 学会关注有物体的区域。它不是最终的多类别占据预测。
            pred_occ_mask = self.pred_occ_mask(up_out).sigmoid()
        #* 将加权修正量加回历史特征能够区分占据区域与空闲区间
        # 5. 论文式(4)：加回 V_warp 得到修正后的历史特征；实现还追加了 ReLU。
        out += identity
        out = self.relu(out)
        # 6. 交回 StreamAgg：out 用于语义辅助预测和时序融合，mask 用于几何监督。
        return out, pred_occ_mask

__all__ = ["RefineNet"]

