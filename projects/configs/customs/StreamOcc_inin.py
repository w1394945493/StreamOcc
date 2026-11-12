nusc_class_frequencies = [944004, 1897170, 152386, 2391677, 16957802, 724139, 189027,
                          2074468, 413451, 2384460, 5916653, 175883646, 4275424, 51393615,
                          61411620, 105975596, 116424404, 1892500630]

# ================ base config ===================
plugin = True
plugin_dir = "projects/mmdet3d_plugin/"
dist_params = dict(backend="nccl")
log_level = "INFO"
work_dir = None

#*---------------------------------------------------------#
#* 训练按迭代步（iteration）训练和停止，num_epochs 只是用来换算总步数。
total_batch_size = 8
num_gpus = 4
batch_size = total_batch_size // num_gpus
num_iters_per_epoch = int(28130 // (num_gpus * batch_size))
num_epochs = 24
checkpoint_epoch_interval = 3

checkpoint_config = dict(
    interval=num_iters_per_epoch * checkpoint_epoch_interval
)
log_config = dict(
    interval=51,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
        dict(type="TensorboardLoggerHook"),
    ],
)
load_from = '/home/wangyushen/Data/Weights/streamocc/bevdet-r50-4d-cbgs.pth'
resume_from = None
resume_from_EMA = None
workflow = [("train", 1)]
input_shape = (704, 256)


skip_prob = 0.0
sequence_flip_prob = 0.0
# Model
grid_config = {
    'y': [-40.0, 40.0, 0.8],
    'x': [-40.0, 40.0, 0.8],
    'z': [-1, 5.4, 0.8],
    'depth': [1.0, 45.0, 0.5],
    'range': [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4],
}
grid_config_full = {
    'y': [-40.0, 40.0, 0.4],
    'x': [-40.0, 40.0, 0.4],
    'z': [-1, 5.4, 0.4],
    'depth': [1.0, 45.0, 0.5],
    'range': [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4],
}
point_cloud_range = [-40, -40, -1.0, 40, 40, 5.4]

bda_aug_conf = dict(
    rot_lim=(0.0, 0.0),
    scale_lim=(1.00, 1.00),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)

data_config = {
    'Ncams': 6,
    'cams': [
        'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'],
    'input_size': (256, 704),
}
# ================== model ========================
class_names = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]

occ_class_names = ['other', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle',
                   'pedestrian', 'traffic_cone', 'trailer', 'truck', 'driveable_surface',
                   'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation']


# group split
group_split = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2],  # front, back, empty
               [0, 0, 1, 2, 0, 3, 4, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 6],  # other, front(fine_less), empty
               [0, 1, 0, 0, 2, 0, 0, 3, 4, 0, 5, 0, 0, 0, 0, 0, 0, 6],  # other, front(fine_more), empty
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 4],  # other, back(fine_less), empty
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 3, 4]]  # other, back(fine_more), empty
group_detr = len(group_split) + 1
group_classes = [17] + [group[-1] for group in group_split]


# 3D 检测分支的类别数（来自 class_names），与占据分支的 18 类区分。
num_classes = len(class_names)
embed_dims = 256
numC_Trans = 64
num_groups = 8
# 稀疏检测解码共 6 轮：前 5 轮聚合图像，最后 1 轮聚合体素并执行 DQA。
num_decoder = 6
# 第一轮先建立当前帧目标表示，后续轮次再融合历史实例 query。
num_single_frame_decoder = 1
use_deformable_func = True  # mmdet3d_plugin/ops/setup.py needs to be executed
num_levels = 3
drop_out = 0.1
occ_pred_weight = 1.0
# 外层对 QueryAgg 返回的损失进行整体加权，平衡检测辅助任务与占据学习。
det_pred_weight = 0.2
# 开启历史实例缓存和时序注意力。
temporal = True
# 注意力中将实例内容特征与 anchor 几何编码拼接，因此注意力维度变为 2C。
decouple_attn = True
# 预测 centerness/yawness 质量分数，辅助评估框的位置与朝向质量。
with_quality_estimation = True
num_def_points = 6
pred_occ = True
num_without_img = 1
temp_cat_method = 'cat'
temporal_layers = 1
down_ratio = 4
use_nms_filter = True
_num_points_self_ = 6
use_different_aug = True
_pos_dim_ = [96, 96, 64]
use_mask_head = True
valid_for_detection = True
no_maskhead_infertime = True

model = dict(
    type="StreamOcc",
    use_grid_mask=False,
    num_levels=num_levels,
    valid_for_detection=valid_for_detection,
    use_deformable_func=use_deformable_func,
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch'),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[1024, 2048],
        out_channels=embed_dims,
        num_outs=1,
        start_level=0,
        out_ids=[0],
        use_DETR=True),
    multi_neck=dict(
        type='MultiFPN',
        in_channels=[512, 256, 256],
        out_channels=embed_dims,
        num_outs=num_levels,
        start_level=0,
        mid_level=num_levels-2,
        out_ids=[0, 1, 2],
    ),
    streamocc_head=dict(
        type='StreamOccHead',
        grid_config=grid_config,
        no_maskhead_infertime=no_maskhead_infertime,
        group_split=group_split,
        det_pred_weight=det_pred_weight,
        occ_pred_weight=occ_pred_weight,
        positional_encoding=dict(
            type='CustomLearnedPositionalEncoding3D',
            num_feats=_pos_dim_,
            row_num_embed=100,
            col_num_embed=100,
            tub_num_embed=8
        ),
        streamagg=dict(
            type='StreamAgg',
            embed_dims=embed_dims,
            temp_cat_method=temp_cat_method,
            num_classes=18,
            img_view_transformer=dict(
                type='LSSViewTransformerBEVDepth',
                grid_config=grid_config_full,
                input_size=data_config['input_size'],
                in_channels=embed_dims,
                out_channels=numC_Trans,
                sid=False,
                collapse_z=False,
                loss_depth_weight=0.05,
                depthnet_cfg=dict(use_dcn=False, aspp_mid_channels=96),
                downsample=16
            ),
            voxel_encoder_backbone=dict(
                type='CustomResNet3D',
                numC_input=numC_Trans,
                num_layer=[1,2, 2],
                with_cp=False,
                num_channels=[numC_Trans,numC_Trans*2,numC_Trans*2],
                stride=[1,2,2],
                backbone_output_ids=[0,1,2],
            ),
            voxel_encoder_neck=dict(
                type='LSSFPN3D_small',
                in_channels=numC_Trans*5,
                out_channels=embed_dims,
                size=(8, 100, 100)),
            refine_net_cfg=dict(
                type='RefineNet',
                channels=embed_dims,
                internal_channels=numC_Trans,
                use_occupied_head=True,
            ),
            grid_config=grid_config,
            use_forecast_head=True,
            surround_occ=False,
        ),
        # ===== 3D 检测与占据交互：论文 3.2 QueryAgg =====
        # Sparse4D 风格的稀疏实例查询：图像 → query → 体素几何增强 → DQA 回写体素。
        # 此分支在占据推理中仍参与计算，并非训练结束后可直接移除的辅助检测头。
        queryagg=dict(
            type="QueryAgg",
            # 训练时仅让匹配正样本中分类置信度超过该值的预测参与框回归损失。
            cls_threshold_to_reg=0.05,
            num_img_decoder=num_decoder-num_without_img,
            use_mask_head=use_mask_head,
            decouple_attn=decouple_attn,
            occ_pred_weight=occ_pred_weight,
            use_nms_filter=use_nms_filter,
            # 为体素/query 交互提供三维空间位置编码。
            positional_encoding=dict(
                type='CustomLearnedPositionalEncoding3D',
                num_feats=_pos_dim_,
                row_num_embed=100,
                col_num_embed=100,
                tub_num_embed=8
            ),
            # 1. 实例库：保存初始 query/anchor，以及跨帧传播的历史实例特征和框。
            instance_bank=dict(
                type="InstanceBank",
                num_anchor=900, # 常规候选 query 上限；训练中的去噪 query 另行追加。
                embed_dims=embed_dims,
                # K-means 生成的初始三维框参数，提供几何起点，不是模型 checkpoint。
                anchor="/home/wangyushen/Data/Datasets/nuscenes_anns/anns/streamocc/nuscenes_kmeans900_42m.npy",
                # 提供 anchor 关键点生成及跨帧坐标/运动传播能力。
                anchor_handler=dict(type="SparseBox3DKeyPointsGenerator"),
                # 从历史预测中缓存高置信度实例，下一帧与当前候选共同更新。
                num_temp_instances=600 if temporal else -1,
                confidence_decay=0.6, # 历史置信度衰减，避免旧目标长期占据缓存。
                feat_grad=False, # 固定初始 instance_feature 参数；不等于关闭整个检测分支梯度。
            ),
            # 2. 将框的位置、尺寸、朝向和速度编码成 query 的几何位置特征。
            anchor_encoder=dict(
                type="SparseBox3DEncoder",
                vel_dims=3,
                embed_dims=[128, 32, 32, 64] if decouple_attn else 256,
                mode="cat" if decouple_attn else "add",
                output_fc=not decouple_attn,
                in_loops=1,
                out_loops=4 if decouple_attn else 2,
            ),
            num_single_frame_decoder=num_single_frame_decoder,
            # 3. 检测解码流程。gnn=当前实例间自注意力；temp_gnn=当前与历史实例交互；
            # deformable=图像采样；deformable_vox=体素采样；refine=类别/框逐轮预测。
            # 最后的 dqa 将筛选出的实例特征注入相应体素区域。
            operation_order=(
                [
                    "gnn",
                    "norm",
                    "deformable",
                    "ffn",
                    "norm",
                    "refine",
                ]
                # 第一阶段：当前帧图像查询解码。
                * num_single_frame_decoder
                + [
                    "temp_gnn",
                    "gnn",
                    "norm",
                    "deformable",
                    "ffn",
                    "norm",
                    "refine",
                ]
                # 第二阶段：加入历史 query，继续从多视角图像获取证据。
                * (num_decoder - num_single_frame_decoder -1)
                + [
                    "temp_gnn",
                    "deformable_vox", # 第三阶段：Voxel-to-Query，补充三维几何信息。
                    "norm",
                    "gnn",
                    "ffn_vox",
                    "norm",
                    "refine",
                    "dqa",
                ]
            )[2:], # 跳过最开始的 gnn/norm，让初始 query 先获取图像特征。
            # temp_gnn：以当前实例为查询，聚合缓存的历史实例信息。
            temp_graph_model=dict(
                type="MultiheadAttention",
                embed_dims=embed_dims if not decouple_attn else embed_dims * 2,
                num_heads=num_groups,
                batch_first=True,
                dropout=drop_out,
            )
            if temporal
            else None,
            # 论文 Dynamic Query Aggregation：按预测框建立 query-voxel 连接，
            # 通过注意力与门控增强对应体素；这是占据交互模块，不是框预测头。
            dqa_module=dict(
                type="DQA",
                embed_dims=embed_dims,
                down_ratio=down_ratio,
                grid_config=grid_config,
            ),
            # gnn：当前帧候选实例之间的多头自注意力。
            graph_model=dict(
                type="MultiheadAttention",
                embed_dims=embed_dims if not decouple_attn else embed_dims * 2,
                num_heads=num_groups,
                batch_first=True,
                dropout=drop_out,
            ),
            norm_layer=dict(type="LN", normalized_shape=embed_dims),
            # 图像聚合使用 cat 残差，输入为 2C；FFN 将其映射回 query 的 C 维。
            ffn=dict(
                type="AsymmetricFFN",
                in_channels=embed_dims * 2,
                pre_norm=dict(type="LN"),
                embed_dims=embed_dims,
                feedforward_channels=embed_dims * 4,
                num_fcs=2,
                ffn_drop=drop_out,
                act_cfg=dict(type="ReLU", inplace=True),
            ),
            # 体素聚合使用 add 残差，FFN 输入仍为 C 维。
            ffn_vox=dict(
                type="AsymmetricFFN",
                in_channels=embed_dims,
                pre_norm=dict(type="LN"),
                embed_dims=embed_dims,
                feedforward_channels=embed_dims * 4,
                num_fcs=2,
                ffn_drop=drop_out,
                act_cfg=dict(type="ReLU", inplace=True),
            ),
            # 4. Image-to-Query：在三维 anchor 周围生成采样点，投影到六相机多尺度
            # 图像特征中采样并加权聚合，避免对全部图像像素执行全局注意力。
            deformable_model=dict(
                type="DeformableFeatureAggregation",
                use_voxel_feature=False,
                embed_dims=embed_dims,
                num_groups=num_groups,
                num_levels=num_levels,
                num_cams=6,
                attn_drop=0.15,
                use_deformable_func=use_deformable_func,
                use_camera_embed=True,
                residual_mode="cat",
                # residual_mode="add",
                attn_cfgs=dict(
                    type='DeformCrossAttention3D',
                    embed_dims=embed_dims,
                    grid_config=grid_config,
                    num_heads=num_groups,
                    num_levels=1,
                    num_points=num_def_points),
                # 图像采样点由可学习偏移与固定点组成；固定点包含中心和六个轴向点。
                kps_generator=dict(
                    type="SparseBox3DKeyPointsGenerator",
                    num_learnable_pts=num_def_points,
                    fix_scale=[
                        [0, 0, 0],
                        [0.45, 0, 0],
                        [-0.45, 0, 0],
                        [0, 0.45, 0],
                        [0, -0.45, 0],
                        [0, 0, 0.45],
                        [0, 0, -0.45],
                    ],
                ),
            ),
            # 5. Voxel-to-Query：启用体素特征分支，在 StreamAgg 输出的三维特征中
            # 用 DeformCrossAttention3D 采样，改善目标的深度定位与几何表示。
            deformable_model_vox=dict(
                type="DeformableFeatureAggregation",
                embed_dims=embed_dims,
                num_groups=num_groups,
                num_levels=num_levels,
                num_cams=6,
                attn_drop=0.15,
                use_voxel_feature=True,
                use_deformable_func=False,
                use_camera_embed=True,
                residual_mode="add",
                attn_cfgs=dict(
                    type='DeformCrossAttention3D',
                    embed_dims=embed_dims,
                    grid_config=grid_config,
                    num_heads=num_groups,
                    num_levels=1,
                    num_points=num_def_points),
                kps_generator=dict(
                    type="SparseBox3DKeyPointsGenerator",
                    num_learnable_pts=num_def_points,
                    fix_scale=[
                        [0, 0, 0],
                        [0.45, 0, 0],
                        [-0.45, 0, 0],
                        [0, 0.45, 0],
                        [0, -0.45, 0],
                        [0, 0, 0.45],
                        [0, 0, -0.45],
                    ],
                ),
            ),
            # 6. 每轮解码后的预测头：更新三维框，并预测类别和可选的质量分数。
            refine_layer=dict(
                type="SparseBox3DRefinementModule",
                embed_dims=embed_dims,
                num_cls=num_classes,
                refine_yaw=True,
                with_quality_estimation=with_quality_estimation,
            ),
            # 7. 训练目标构造：SparseBox3DTarget 内部使用匈牙利匹配分配 GT。
            # 同时生成带噪声的 GT query 做去噪训练；不是 CenterHead 的中心热力图监督。
            sampler=dict(
                type="SparseBox3DTarget",
                num_dn_groups=5, # 当前帧去噪 query 分组数。
                num_temp_dn_groups=3, # 用于跨帧去噪的分组数。
                dn_noise_scale=[2.0] * 3 + [0.5] * 7,
                max_dn_gt=32, # 去噪分支最多使用的 GT 数量，不限制常规检测目标数。
                add_neg_dn=True,
                # 以下 cls/box/reg 权重控制匹配代价；不要与后面的损失权重混淆。
                cls_weight=2.0,
                box_weight=0.25,
                reg_weights=[2.0] * 3 + [0.5] * 3 + [0.0] * 4,
                # 针对交通锥调整回归/匹配维度权重，降低无意义朝向的影响。
                cls_wise_reg_weights={
                    class_names.index("traffic_cone"): [
                        2.0,
                        2.0,
                        2.0,
                        1.0,
                        1.0,
                        1.0,
                        0.0,
                        0.0,
                        1.0,
                        1.0,
                    ],
                },
            ),
            # 8. 检测监督：逐 query 的 sigmoid Focal 分类损失。
            loss_cls=dict(
                type="FocalLoss",
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=2.0,
            ),
            # 三维框 L1 回归，以及 centerness/yawness 质量监督。
            loss_reg=dict(
                type="SparseBox3DLoss",
                loss_box=dict(type="L1Loss", loss_weight=0.25),
                loss_centerness=dict(type="CrossEntropyLoss", use_sigmoid=True),
                loss_yawness=dict(type="GaussianFocalLoss"),
                cls_allow_reverse=[class_names.index("barrier")], # 护栏允许反向朝向等价处理。
            ),
            # 9. 推理后处理：将内部框编码还原为物理框参数，选取并输出检测结果。
            decoder=dict(type="SparseBox3DDecoder"),
            # 检测回归损失各维权重：中心 xyz 加权为 2，其余参与监督的维度为 1。
            reg_weights=[2.0] * 3 + [1.0] * 7,
        ),
        cls_freq=nusc_class_frequencies,
        loss_occ=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=10.0
        ),
        mask_decoder_head=dict(
            type='MaskHead',
            in_channels=embed_dims,
            embed_dims=embed_dims,
            num_query=100,
            group_detr=group_detr,
            group_classes=group_classes,
            num_classes=17,
            transformer_decoder=dict(
                type='MaskOccDecoder',
                return_intermediate=True,
                num_layers=1,
                transformerlayers=dict(
                    type='MaskOccDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiScaleDeformableAttention3D',
                            embed_dims=embed_dims,
                            num_levels=1,
                            num_points=4,),
                        dict(
                            type='GroupMultiheadAttention',
                            group=group_detr,
                            embed_dims=embed_dims,
                            num_heads=8,
                            dropout=0.1),
                    ],
                    feedforward_channels=2*embed_dims,
                    ffn_dropout=0.1,
                    operation_order=('cross_attn', 'norm', 'self_attn', 'norm',
                                        'ffn', 'norm'))),
            predictor=dict(
                type='MaskPredictorHead_Group',
                nbr_classes=17,
                group_detr=group_detr,
                group_classes=group_classes,
                in_dims=embed_dims,
                hidden_dims=2*embed_dims,
                out_dims=embed_dims,
                mask_dims=embed_dims),
            use_camera_mask=True,
            use_lidar_mask=False,
            loss_cls=dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0,
                reduction='mean',
                class_weight=[1.0] * 17 + [0.1]),
            loss_mask=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                reduction='mean',
                loss_weight=20.0),
            loss_dice=dict(
            type='DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=1.0)),
        # model training and testing settings
        train_cfg=dict(
            pts=dict(
                out_size_factor=4,
                assigner=dict(
                    type='MaskHungarianAssigner3D',
                    cls_cost=dict(type='MaskClassificationCost', weight=1.0),
                    mask_cost=dict(type='MaskFocalLossCost', weight=20.0, binary_input=True),
                    dice_cost=dict(type='MaskDiceLossCost', weight=1.0, pred_act=True, eps=1.0),
                    use_camera_mask=True,
                    use_lidar_mask=False),
                sampler=dict(
                    type='MaskPseudoSampler',
                    use_camera_mask=True,
                    use_lidar_mask=False)
            )),
        test_cfg=dict(
            pts=dict(
                mask_threshold=0.7,
                overlap_threshold=0.8,
                occupy_threshold=0.3,
                inf_merge=True,
                only_encoder=True,
            )),
    ),
)

# ================== data ========================
dataset_type = "NuScenes3DDetTrackDataset"
data_root = "/home/wangyushen/Data/Datasets/Nuscenes/"
file_client_args = dict(backend="disk")

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)


train_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args,
    ),
    dict(type='LoadOccupancy_OCC'),
    dict(type="ResizeCropFlipImage"),
    dict(type='PointToMultiViewDepth', downsample=1, grid_config=grid_config),
    dict(type="BBoxRotation_StreamOcc"),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(
        type="CircleObjectRangeFilter",
        class_dist_thred=[42] * len(class_names),
    ),
    dict(type="InstanceNameFilter", classes=class_names),
    dict(type="NuScenesSparse4DAdaptor"),
    dict(
        type="Collect",
        keys=[
            "img",
            "timestamp",
            "projection_mat",
            "image_wh",
            "gt_depth",
            "gt_bboxes_3d",
            "gt_labels_3d",
            "view_tran_comp",
            "voxel_semantics",
            "mask_lidar",
            "mask_camera",
        ],
        meta_keys=["T_global", "T_global_inv", "timestamp", "instance_id"],
    ),
]
test_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="ResizeCropFlipImage"),
    dict(type='LoadOccupancy_OCC'),

    dict(type="BBoxRotation_StreamOcc"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(
        type="CircleObjectRangeFilter",
        class_dist_thred=[42] * len(class_names),
    ),
    dict(type="NuScenesSparse4DAdaptor"),
    dict(
        type="Collect",
        keys=[
            "img",
            "timestamp",
            "projection_mat",
            "image_wh",
            "gt_bboxes_3d",
            "gt_labels_3d",
            "view_tran_comp",
            "voxel_semantics",
            "mask_lidar",
            "mask_camera",
        ],
        meta_keys=["T_global", "T_global_inv", "timestamp"],
    ),
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False,
)

data_basic_config = dict(
    type=dataset_type,
    data_root=data_root,
    classes=class_names,
    occ_classes=occ_class_names,
    modality=input_modality,
    version="v1.0-trainval",
    skip_prob=skip_prob,
    sequence_flip_prob=sequence_flip_prob,
    use_different_aug=use_different_aug,
    valid_for_detection=valid_for_detection,
)

data_aug_conf = {
    "resize_lim": (0.40, 0.47),
    "final_dim": input_shape[::-1],
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (-5.4, 5.4),
    "H": 900,
    "W": 1600,
    "rand_flip": True,
    "rot3d_range": [-0.3925, 0.3925],
    "bda_aug": bda_aug_conf,
}


data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=batch_size,
    train=dict(
        **data_basic_config,
        ann_file="/home/wangyushen/Data/Datasets/nuscenes_anns/anns/streamocc/nuscenes_infos_aug_train.pkl",
        pipeline=train_pipeline,
        test_mode=False,
        data_aug_conf=data_aug_conf,
        with_seq_flag=True,
        sequences_split_num=2, #* 训练时将每个视频序列分为两个子序列
        keep_consistent_seq_aug=True,
    ),
    val=dict(
        **data_basic_config,
        ann_file="/home/wangyushen/Data/Datasets/nuscenes_anns/anns/streamocc/nuscenes_infos_aug_val.pkl",
        pipeline=test_pipeline,
        data_aug_conf=data_aug_conf,
        test_mode=True,
    ),
    test=dict(
        **data_basic_config,
        ann_file="/home/wangyushen/Data/Datasets/nuscenes_anns/anns/streamocc/nuscenes_infos_aug_val.pkl",
        pipeline=test_pipeline,
        data_aug_conf=data_aug_conf,
        test_mode=True,
    ),
)
# ================== training ========================
optimizer = dict(
    type="AdamW",
    lr=2e-4,
    weight_decay=0.01,
)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=1.0 / 3,
    step=[200*num_iters_per_epoch,230*num_iters_per_epoch]
)
runner = dict(
    type="IterBasedRunner", #* IterBasedRunner
    max_iters=num_iters_per_epoch * num_epochs,
)

# ================== eval ========================
vis_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(
        type="Collect",
        keys=["img"],
        meta_keys=["timestamp", "lidar2img"],
    ),
]
evaluation = dict(
    interval=num_iters_per_epoch * checkpoint_epoch_interval,
    pipeline=vis_pipeline,
)


custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
        split_iter=num_iters_per_epoch * checkpoint_epoch_interval,
        resume=resume_from_EMA,
    ),
    dict(
        type='SyncbnControlHook',
        syncbn_start_iter=0,
    ),
]