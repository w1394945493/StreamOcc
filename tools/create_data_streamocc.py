# Copyright (c) OpenMMLab. All rights reserved.

import numpy as np
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
from nuscenes_converter import create_nuscenes_infos as create_nuscenes_infos
import os

map_name_from_general_to_detection = {
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.wheelchair': 'ignore',
    'human.pedestrian.stroller': 'ignore',
    'human.pedestrian.personal_mobility': 'ignore',
    'human.pedestrian.police_officer': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'animal': 'ignore',
    'vehicle.car': 'car',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.truck': 'truck',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.emergency.ambulance': 'ignore',
    'vehicle.emergency.police': 'ignore',
    'vehicle.trailer': 'trailer',
    'movable_object.barrier': 'barrier',
    'movable_object.trafficcone': 'traffic_cone',
    'movable_object.pushable_pullable': 'ignore',
    'movable_object.debris': 'ignore',
    'static_object.bicycle_rack': 'ignore',
}
classes = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

CLASS_IDS = {name: index for index, name in enumerate(classes)}


def augment_info(nuscenes, sample, info, annotations, velocities, occ_root):
    """Add ego annotations in-place, reusing the converter's annotation data."""
    gt_boxes, gt_labels, instance_inds = [], [], []
    trans = -np.asarray(info['ego2global_translation'])
    rot = Quaternion(info['ego2global_rotation']).inverse
    for ann, velocity, instance in zip(annotations, velocities, info['instance_inds']):
        label = CLASS_IDS.get(map_name_from_general_to_detection[ann['category_name']])
        if label is None or ann['num_lidar_pts'] + ann['num_radar_pts'] <= 0:
            continue
        if np.any(np.isnan(velocity)):
            velocity = np.zeros(3)
        box = Box(ann['translation'], ann['size'],
                  Quaternion(ann['rotation']), velocity=velocity)
        box.translate(trans)
        box.rotate(rot)
        gt_boxes.append(np.concatenate([
            box.center, np.asarray(box.wlh)[[1, 0, 2]],
            [box.orientation.yaw_pitch_roll[0]], box.velocity[:2]]))
        gt_labels.append(label)
        instance_inds.append(int(instance))
    info['gt_boxes_ego'] = gt_boxes
    # Keep the existing spelling for compatibility with dataset consumers.
    info['gt_lables_ego'] = gt_labels
    info['instance_inds_ego'] = instance_inds
    scene = nuscenes.get('scene', sample['scene_token'])
    info['occ_path'] = os.path.join(occ_root, scene['name'], info['token'])


def main():
    import argparse

    parser = argparse.ArgumentParser(description="nuscenes converter")
    parser.add_argument("--root_path", type=str, default="./data/nuscenes/")
    parser.add_argument("--info_prefix", type=str, default="nuscenes_occ")
    parser.add_argument("--version", type=str, default="v1.0-trainval")
    parser.add_argument("--max_sweeps", type=int, default=10)
    parser.add_argument('--out_dir', '--out-dir', default=None,
                        help='Augmented pkl directory (default: sibling nuscenes_anno_pkls)')
    parser.add_argument('--occ_root', '--occ-root', default=None,
                        help='Occupancy label root (default: root_path/gts)')
    args = parser.parse_args()

    versions = list(dict.fromkeys(v.strip() for v in args.version.split(',')))
    if any(v not in ('v1.0-trainval', 'v1.0-mini', 'v1.0-test') for v in versions):
        parser.error('--version must contain v1.0-trainval, v1.0-mini or v1.0-test')
    if args.max_sweeps < 0:
        parser.error('--max_sweeps must be non-negative')
    from functools import partial
    root_path = os.path.abspath(os.path.expanduser(args.root_path))
    out_dir = args.out_dir or os.path.join(os.path.dirname(root_path), 'nuscenes_anno_pkls')
    occ_root = os.path.abspath(os.path.expanduser(
        args.occ_root or os.path.join(root_path, 'gts')))
    for version in versions:
        # Different versions share split names, so isolate their file prefixes.
        prefix = (args.info_prefix if len(versions) == 1
                  else args.info_prefix + '_' + version)
        create_nuscenes_infos(
            root_path,
            prefix,
            version=version,
            max_sweeps=args.max_sweeps,
            out_dir=out_dir,
            info_transform=partial(augment_info, occ_root=occ_root),
            augmented=True,
        )



if __name__ == '__main__':
    main()
