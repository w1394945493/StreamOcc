"""Small synthetic nuScenes regression tests; no dataset or CUDA required."""
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import create_data_streamocc as entry
import nuscenes_converter as converter


class FakeNuScenes:
    instances = []

    def __init__(self, version, dataroot, **kwargs):
        self.version, self.dataroot = version, dataroot
        self.velocity_calls = 0
        self.box_calls = 0
        self.instances.append(self)
        self.scene = [dict(name='scene-0061', token='train', first_sample_token='train'),
                      dict(name='scene-0103', token='val', first_sample_token='val')]
        cameras = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                   'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        self.sample = [dict(token=split, scene_token=split, timestamp=0,
                            anns=['ann'] if split == 'train' else [],
                            data=dict(LIDAR_TOP=split, **{c: c for c in cameras}))
                       for split in ['train', 'val']]

    def get(self, table, token):
        if table == 'sample':
            return next(s for s in self.sample if s['token'] == token)
        if table == 'scene':
            return next(s for s in self.scene if s['token'] == token)
        if table == 'sample_data':
            return dict(token=token, calibrated_sensor_token='sensor',
                        ego_pose_token='pose', timestamp=0, prev='')
        if table == 'calibrated_sensor':
            return dict(translation=[0, 0, 0], rotation=[1, 0, 0, 0],
                        camera_intrinsic=np.eye(3).tolist())
        if table == 'ego_pose':
            return dict(translation=[1, 2, 0],
                        rotation=Quaternion(axis=[0, 0, 1], angle=np.pi / 2).elements)
        if table == 'sample_annotation':
            return dict(token='ann', translation=[3, 4, 0], size=[2, 4, 2],
                        rotation=[1, 0, 0, 0], category_name='vehicle.car',
                        instance_token='instance', num_lidar_pts=1, num_radar_pts=0)
        raise AssertionError(table)

    def getind(self, table, token):
        return 7

    def get_sample_data_path(self, token):
        return str(Path(self.dataroot) / 'sensor.bin')

    def get_sample_data(self, token):
        self.box_calls += 1
        boxes = [Box([2, -2, 0], [2, 4, 2],
                     Quaternion(axis=[0, 0, 1], angle=-np.pi / 2),
                     name='vehicle.car')] if token == 'train' else []
        return self.get_sample_data_path(token), boxes, None

    def box_velocity(self, token):
        self.velocity_calls += 1
        return np.array([2., 0., 0.])


class ConversionTests(unittest.TestCase):
    def test_single_pass_outputs_and_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'dataset'
            root.mkdir()
            (root / 'sensor.bin').touch()
            out = Path(tmp) / 'output'
            argv = ['convert', '--root_path', str(root), '--version', 'v1.0-mini',
                    '--out_dir', str(out), '--occ_root', str(Path(tmp) / 'labels')]
            FakeNuScenes.instances = []
            with patch.object(sys, 'argv', argv), patch('nuscenes.nuscenes.NuScenes', FakeNuScenes):
                entry.main()
            self.assertEqual(len(FakeNuScenes.instances), 1)
            nusc = FakeNuScenes.instances[0]
            self.assertEqual(nusc.velocity_calls, 1)
            self.assertEqual(nusc.box_calls, 2)  # Only LiDAR, never camera/scene checks.
            self.assertEqual(sorted(p.name for p in out.iterdir()),
                             ['nuscenes_occ_infos_aug_train.pkl', 'nuscenes_occ_infos_aug_val.pkl'])
            self.assertFalse(list(root.glob('*.pkl')))
            with open(out / 'nuscenes_occ_infos_aug_train.pkl', 'rb') as f:
                train = pickle.load(f)
            info = train['infos'][0]
            self.assertEqual(train['metadata']['version'], 'v1.0-mini')
            np.testing.assert_allclose(info['gt_boxes_ego'][0],
                                       [2, -2, 0, 4, 2, 2, -np.pi/2, 0, -2], atol=1e-12)
            np.testing.assert_allclose(info['gt_velocity'], [[0, -2]], atol=1e-12)
            self.assertEqual(info['gt_lables_ego'], [0])
            self.assertEqual(info['instance_inds_ego'], [7])
            self.assertEqual(info['occ_path'], str(Path(tmp) / 'labels/scene-0061/train'))
            np.testing.assert_equal(info['cams']['CAM_FRONT']['cam_intrinsic'], np.eye(3))
            with open(out / 'nuscenes_occ_infos_aug_val.pkl', 'rb') as f:
                self.assertEqual(pickle.load(f)['infos'][0]['gt_boxes_ego'], [])

    def test_empty_training_filtered_and_unselected_skipped(self):
        nusc = FakeNuScenes('v1.0-mini', '/unused')
        # The val sample has no annotations; the train sample must not be read.
        with patch.object(converter.mmcv, 'check_file_exist'):
            train, val = converter._fill_trainval_infos(
                nusc, {'val'}, set(), info_transform=lambda *args: entry.augment_info(*args, occ_root='/occ'),
                augmented=True)
        self.assertEqual((train, val), ([], []))
        self.assertEqual(nusc.box_calls, 1)

    def test_nan_velocity_and_ignored_annotations(self):
        nusc = FakeNuScenes('v1.0-mini', '/unused')
        ann = nusc.get('sample_annotation', 'ann')
        info = dict(ego2global_translation=[0, 0, 0], ego2global_rotation=[1, 0, 0, 0],
                    instance_inds=[7, 8], token='train')
        ignored = dict(ann, category_name='animal')
        entry.augment_info(nusc, nusc.sample[0], info, [ann, ignored],
                           [np.array([np.nan, 0, 0]), np.zeros(3)], '/occ')
        self.assertEqual(len(info['gt_boxes_ego']), 1)
        np.testing.assert_equal(info['gt_boxes_ego'][0][-2:], [0, 0])

    def test_test_version_and_multi_version_prefixes(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch('nuscenes.nuscenes.NuScenes', FakeNuScenes), \
                 patch.object(converter, 'get_available_scenes', return_value=[]), \
                 patch.object(converter, '_fill_trainval_infos', return_value=([{'token': 'test'}], [])):
                converter.create_nuscenes_infos(tmp, 'test', version='v1.0-test',
                                               augmented=True, info_transform=entry.augment_info)
            self.assertEqual([p.name for p in Path(tmp).iterdir()], ['test_infos_test.pkl'])
        with patch.object(sys, 'argv', ['convert', '--version', 'v1.0-mini,v1.0-test']), \
             patch.object(entry, 'create_nuscenes_infos') as create:
            entry.main()
        self.assertEqual(create.call_count, 2)
        self.assertNotEqual(create.call_args_list[0].args[1], create.call_args_list[1].args[1])


if __name__ == '__main__':
    unittest.main()
