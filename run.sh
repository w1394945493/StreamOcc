
conda activate bdcocc-cu121
python -m pip install torch_geometric==2.5.3 yapf==0.33.0
conda activate bdcocc-cu121

cd /home/wys/wangyushen/forks/StreamOcc/projects/mmdet3d_plugin/ops
python setup.py build_ext --inplace

cd /home/wys/wangyushen/forks/StreamOcc/deform_attn_3d
python setup.py build_ext --inplace

cd /home/wys/wangyushen/forks/StreamOcc/projects/mmdet3d_plugin/models/bev_pool_v2
python setup.py build_ext --inplace

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


conda activate bdcocc-cu121
cd /home/wys/wangyushen/forks/StreamOcc
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python tools/test.py \
  /home/wys/wangyushen/forks/StreamOcc/projects/configs/customs/StreamOcc.py \
  --eval bbox