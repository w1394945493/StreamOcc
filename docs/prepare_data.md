### Prepare the data

### Dowload the nuScenes dataset

Download the [nuScenes dataset](https://www.nuscenes.org/nuscenes#download) and create symbolic links.

```bash
cd ${streamocc_path}
mkdir data
ln -s path/to/nuscenes ./data
```

### Dowload the Occ3D-nuScenes dataset

For Occupancy Prediction task, download Occ3D-nuScenes from [CVPR2023-3D-Occupancy-Prediction](https://github.com/CVPR2023-3D-Occupancy-Prediction/CVPR2023-3D-Occupancy-Prediction) and place it in data/nuscenes/gts

### Prepare pkl files

Pack the meta-information and labels of the dataset, and generate the required .pkl files.

```bash
python3 tools/create_data_streamocc.py
```

For nuScenes mini, use the root containing `v1.0-mini`, `samples`, and
`sweeps` (not the `v1.0-mini` metadata directory itself):

```bash
python3 tools/create_data_streamocc.py \
    --root_path /path/to/nuscenes \
    --version v1.0-mini \
    --info_prefix nuscenes_mini \
    --out_dir ./data/nuscenes_mini_anno_pkls \
    --occ_root /path/to/occupancy/gts
```

This writes only `nuscenes_mini_infos_aug_{train,val}.pkl` to `--out_dir`.
Base info generation and augmentation run together in memory, loading the
nuScenes database once per version and writing no intermediate pkl files.
Set the training configuration's annotation paths to these augmented files.
By default, output goes to the data root's sibling `nuscenes_anno_pkls`,
and occupancy paths use `--root_path/gts`. Only `--out_dir` is needed;
the former intermediate `--info_dir` option has been removed.
Stored sensor and occupancy paths are absolute. Occupancy labels are not
generated or checked by this script; the corresponding scene/sample labels
must be supplied separately for occupancy training.

Comma-separated versions are processed separately, with the version appended
to the prefix when more than one version is requested to prevent overwrites.
`v1.0-test` produces only base test infos, since it has no ground-truth
annotations for augmentation.

### Generate anchors by K-means

```bash
python3 tools/anchor_generator.py --ann_file ./data/nuscenes_anno_pkls/nuscenes_occ_infos_aug_train.pkl
```

### Download pre-trained weights

Download the required backbone [BEVDet serires](https://github.com/HuangJunJie2017/BEVDet) and place it in ckpts folder.

### The Overall Structure

Please make sure the structure of StreamOcc is as follows:

```shell
StreamOcc
├── projects/
├── tools/
├── ckpts/
│   ├── bevdet-r50-4d-depth-cbgs.pth
├── data/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── v1.0-test/
│   │   ├── v1.0-trainval/
│   ├── nuscenes_anno_pkls/
│   │   ├── nuscenes_occ_infos_aug_train.pkl
│   │   ├── nuscenes_occ_infos_aug_val.pkl
├── nuscenes_kmeans900_42m.npy
└── others
```

→ Back to: [Training &amp; Inference](/#%EF%B8%8F-training--inference)
