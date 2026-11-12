#*-----------------------------------------------------#
#* wsl2本地
conda activate bdcocc-cu121
python -m pip install torch_geometric==2.5.3 yapf==0.33.0
conda activate bdcocc-cu121

cd /home/wys/wangyushen/forks/StreamOcc/projects/mmdet3d_plugin/ops
python setup.py build_ext --inplace

cd /home/wys/wangyushen/forks/StreamOcc/deform_attn_3d
python setup.py build_ext --inplace

cd /home/wys/wangyushen/forks/StreamOcc/projects/mmdet3d_plugin/models/bev_pool_v2
python setup.py build_ext --inplace

#*-----------------------------------------------------#
#* inin 3090
conda activate OccMamba
python -m pip install torch_geometric==2.5.3 yapf==0.33.0 --proxy http://127.0.0.1:17891

cd /home/wangyushen/forks/StreamOcc/projects/mmdet3d_plugin/ops
python setup.py build_ext --inplace

cd /home/wangyushen/forks/StreamOcc/deform_attn_3d
python setup.py build_ext --inplace

cd /home/wangyushen/forks/StreamOcc/projects/mmdet3d_plugin/models/bev_pool_v2
python setup.py build_ext --inplace


#*-----------------------------------------------------#
#* wsl2本地
# 生成数据集pkl文件
cd /home/wys/wangyushen/forks/StreamOcc
python tools/create_data_streamocc.py \
  --root_path /home/wys/wangyushen/data/Datasets/nuscenes/v1.0-mini \
  --version v1.0-mini \
  --info_prefix nuscenes_mini \
  --out_dir /home/wys/wangyushen/data/Datasets/nuscenes/anns/streamocc \
  --occ_root /home/wys/wangyushen/data/Datasets/nuscenes/gts

# 生成 nuscenes_kmeans900_42m.npy
python tools/anchor_generator.py \
  --ann_file /home/wys/wangyushen/data/Datasets/nuscenes/anns/streamocc/nuscenes_mini_infos_aug_train.pkl \
  --output_file_name /home/wys/wangyushen/data/Datasets/nuscenes/anns/streamocc/nuscenes_kmeans900_42m.npy

#*-----------------------------------------------------#
#* inin 3090
# 生成数据集pkl文件
cd /home/wangyushen/forks/StreamOcc
python tools/create_data_streamocc.py \
  --root_path /home/wangyushen/Data/Datasets/Nuscenes \
  --version v1.0-trainval \
  --info_prefix nuscenes \
  --out_dir /home/wangyushen/Data/Datasets/nuscenes_anns/anns/streamocc \
  --occ_root /home/wangyushen/Data/Datasets/nuscenes_anns/gts

# 生成 nuscenes_kmeans900_42m.npy
# 加上空的 __init__.py 后，StreamOcc 的 projects 也成为普通包，排在前面就能优先被采用。 需要增加此文件，否则可能导入其他文件夹充满文件
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python tools/anchor_generator.py \
  --ann_file /home/wangyushen/Data/Datasets/nuscenes_anns/anns/streamocc/nuscenes_infos_aug_train.pkl \
  --output_file_name /home/wangyushen/Data/Datasets/nuscenes_anns/anns/streamocc/nuscenes_kmeans900_42m.npy

#* 推理
#* wsl2 本地
conda activate bdcocc-cu121
cd /home/wys/wangyushen/forks/StreamOcc
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python tools/test.py \
  /home/wys/wangyushen/forks/StreamOcc/projects/configs/customs/StreamOcc.py \
  --eval bbox

#* ----------------------------------------------------#
#* inin-3090
conda activate OccMamba
cd /home/wangyushen/forks/StreamOcc
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python tools/test.py \
  /home/wangyushen/forks/StreamOcc/projects/configs/customs/StreamOcc_inin.py \
  --eval bbox


#* ----------------------------------------------------#
#* inin-3090
#* 训练
conda activate OccMamba
cd /home/wangyushen/forks/StreamOcc
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

BS=1
GPUS=2
EPOCHS=24
LOG_INTERVAL=50
KEEP_CKPTS=1
ITERS_PER_EPOCH=$((28130 / (BS * GPUS)))
MAX_ITERS=$((ITERS_PER_EPOCH * EPOCHS))

CUDA_VISIBLE_DEVICES=5,7 python -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${GPUS}" \
  tools/train.py \
  /home/wangyushen/forks/StreamOcc/projects/configs/customs/StreamOcc_inin.py \
  --launcher pytorch \
  --work-dir /home/wangyushen/Data/Outputs/streamocc/outputs/train \
  --cfg-options \
  data.samples_per_gpu="${BS}" \
  runner.max_iters="${MAX_ITERS}" \
  log_config.interval="${LOG_INTERVAL}" \
  checkpoint_config.by_epoch=False \
  checkpoint_config.interval="${ITERS_PER_EPOCH}" \
  checkpoint_config.max_keep_ckpts="${KEEP_CKPTS}" \
  evaluation.interval="${ITERS_PER_EPOCH}"