"""
nuScenes Dataset

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com), Zheng Zhang
Please cite our work if the code is helpful to you.
"""

import os
import numpy as np
from collections.abc import Sequence
import pickle

import open3d as o3d
from .default import BaseDataset
from nuscenes.nuscenes import NuScenes
import torch
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
import cv2
from utils.utils import (
    save_all_point_cloud, 
    save_error_point_cloud,
    save_common_entropy,
    save_custom_npy
)
from pathlib import Path
import yaml 

class NuScenesDataset(BaseDataset):
    """用于处理和保存NuScenes数据集的处理器。"""
    def __init__(self, config, sequence, save_settings=None, global_output_dir=None):
        super().__init__(config, save_settings, sequence, global_output_dir) # scene_name is the sequence
        
        # 合并 _init_reader 的逻辑
        nusc_cfg = config.get('dataset', {})
        self.nusc = NuScenes(version=nusc_cfg['version'], dataroot=nusc_cfg['dataroot'], verbose=False)
        
        dataset_config_path = self._get_dataset_config_path('nuscenes')
        with open(dataset_config_path, 'r') as f:
            nuscenes_yaml = yaml.safe_load(f)
        self.cam_names = nuscenes_yaml.get('cameras', ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 
                                                  'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'])
        if sequence:
            self.samples = self._get_samples_for_scene(sequence)
        else:
            self.samples = self.nusc.sample
        if not self.samples:
            raise ValueError(f"No samples found for scene '{sequence or 'any scene'}' in NuScenes dataset.")
        
        self.use_net = self.config.get('pcnetwork', {}).get('use_net', False)
        if self.use_net:
            net_feature_base_path = self.config.get('pcnetwork', {}).get('net_feature_path')
            if not net_feature_base_path:
                raise ValueError("pcnetwork.use_net 为 True, 但 pcnetwork.net_feature_path 未指定。")
            # NuScenes 的 net_feature 通常不按 sequence 划分，直接使用基础路径
            self.net_feature_path = Path(net_feature_base_path)
        else:
            self.net_feature_path = None

        # 配置外部特征加载 (用于 model_predict)
        self.load_external_feature = self.config.get('external_feature', {}).get('load_feature', False)
        feature_cfg = self.config.get('external_feature', {})
        # 使用 'feature_path' 作为键，与 kitti 统一
        self.feature_path = Path(feature_cfg.get('feature_path')) if feature_cfg.get('feature_path') else None
        if feature_cfg.get('save_feature', False) and not self.feature_path:
            print("Warning: external_feature.save_feature is True, but external_feature.feature_path is not specified. Features will not be saved.")
            self.feature_path = None # 确保路径无效
            
    def __len__(self):
        return len(self.samples)

    def _setup_output_dir(self, global_output_dir):
        """对于 NuScenes，序列特定的目录就是全局输出目录。"""
        # NuScenes 的保存方法会自己添加 sequence 子目录
        
        return global_output_dir

    def _get_samples_for_scene(self, scene_name):
        """
        根据场景名称获取该场景下的所有样本。
        """
        # 1. 找到与名称匹配的场景记录
        scenes = [s for s in self.nusc.scene if s['name'] == scene_name]
        if not scenes:
            return []  # 如果找不到场景，返回空列表

        scene = scenes[0]
        
        # 2. 从第一个样本开始，遍历链表以收集所有样本
        samples = []
        current_sample_token = scene['first_sample_token']
        while current_sample_token:
            sample = self.nusc.get('sample', current_sample_token)
            samples.append(sample)
            current_sample_token = sample['next']
            
        return samples
    
    @staticmethod
    def get_sequences(dataset_cfg):
        sequences = dataset_cfg.get('scenes', [])
        if not sequences:
            print("No scenes specified, discovering all scenes...")
            nusc = NuScenes(version=dataset_cfg.get('version', 'v1.0-trainval'), dataroot=dataset_cfg.get('dataroot'), verbose=False)
            sequences = [scene['name'] for scene in nusc.scene]
        return sequences
    
    def _get_lidar_data(self, sample):
        """获取激光雷达点云和位姿。"""
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_path = self.nusc.get_sample_data_path(lidar_token)
        cloud = LidarPointCloud.from_file(lidar_path).points.T[:, :4]

        sd_record_lidar = self.nusc.get('sample_data', lidar_token)
        cs_record_lidar = self.nusc.get('calibrated_sensor', sd_record_lidar['calibrated_sensor_token'])
        ego_pose_lidar = self.nusc.get('ego_pose', sd_record_lidar['ego_pose_token'])
        
        lidar_to_ego = transform_matrix(cs_record_lidar['translation'], Quaternion(cs_record_lidar['rotation']), inverse=False)
        ego_to_global = transform_matrix(ego_pose_lidar['translation'], Quaternion(ego_pose_lidar['rotation']), inverse=False)
        
        transformed_cloud = self.transform(cloud, ego_to_global @ lidar_to_ego)
        return cloud, transformed_cloud, ego_to_global, lidar_token

    def _get_labels(self, lidar_token, num_points):
        """获取激光雷达分割标签。"""
        raw_label = np.zeros(num_points, dtype=np.uint8)
        lidarseg_token = None
        if 'lidarseg' in self.nusc.table_names:
            try:
                lidarseg_record = self.nusc.get('lidarseg', lidar_token)
                lidarseg_token = lidarseg_record['token']
                lidarseg_path = os.path.join(self.nusc.dataroot, lidarseg_record['filename'])
                raw_label = np.fromfile(lidarseg_path, dtype=np.uint8)
            except KeyError:
                pass  # 某些样本可能没有lidarseg标签
            except Exception as e:
                print(f"Warning: Could not load lidarseg labels for sample_data {lidar_token}. Error: {e}")

        label = np.array([self.label_to_index.get(l, 0) for l in raw_label], dtype=np.int32)
        # print("raw_label unique ids:", np.unique(raw_label))
        # print("label unique ids:", np.unique(label))
        return label, lidarseg_token

    def _get_image_and_projections(self, sample, transformed_cloud):
        """获取所有相机图像和点云投影信息。"""
        images = {}
        proj_cloud = {}
        for cam_name in self.cam_names:
            cam_token = sample['data'][cam_name]
            cam_path = self.nusc.get_sample_data_path(cam_token)
            image = cv2.imread(cam_path)
            if image is None:
                continue
            # images[cam_name] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images[cam_name] = image
            
            sd_record_cam = self.nusc.get('sample_data', cam_token)
            cs_record_cam = self.nusc.get('calibrated_sensor', sd_record_cam['calibrated_sensor_token'])
            ego_pose_cam = self.nusc.get('ego_pose', sd_record_cam['ego_pose_token'])

            global_to_ego = transform_matrix(ego_pose_cam['translation'], Quaternion(ego_pose_cam['rotation']), inverse=True)
            ego_to_cam = transform_matrix(cs_record_cam['translation'], Quaternion(cs_record_cam['rotation']), inverse=True)
            
            cam_intrinsics = np.eye(4)
            cam_intrinsics[:3, :3] = cs_record_cam['camera_intrinsic']
            global_to_image_transform = cam_intrinsics @ ego_to_cam @ global_to_ego

            pixels, points, indices = self.project_points_to_camera(transformed_cloud, global_to_image_transform, image.shape)
            proj_cloud[cam_name] = {'pixels': pixels, 'points': points, 'indices': indices}
        return images, proj_cloud

    def __getitem__(self, index):
        sample = self.samples[index]
        
        cloud, transformed_cloud, ego_to_global, lidar_token = self._get_lidar_data(sample)
        label, lidarseg_token = self._get_labels(lidar_token, len(cloud))
        images, proj_cloud = self._get_image_and_projections(sample, transformed_cloud)

        # 加载 pcnetwork 的预计算特征 -> net_output
        net_output_data = np.array([])
        if self.use_net and self.net_feature_path:
            # NuScenes 的特征通常以 lidarseg_token 命名
            feature_file = self.net_feature_path / f"{lidarseg_token}_lidarseg.npz"
            if feature_file.exists():
                with np.load(feature_file) as data:
                    net_output_data = data.get('pred_prob', np.array([]))
            else:
                print(f"警告: PC网络特征文件 {feature_file} 未找到。")

        # [修改] 加载所有相机的 external_feature
        model_predict_data = {}
        if self.load_external_feature:
            for cam_name in self.cam_names:
                model_predict_data[cam_name] = self.load_features(index, cam_name)

        return {
            "images": images,
            "cloud": cloud.astype(np.float32),
            "transformed_cloud": transformed_cloud.astype(np.float32),
            "label": label.astype(np.int32),
            "pose": ego_to_global.astype(np.float32),
            "proj_cloud": proj_cloud,
            "net_output": net_output_data.astype(np.float32),
            "model_predict": model_predict_data,
            "label_token": lidarseg_token,
            "index": index
        }

    def save_label(self, global_pred, frame_id):
        lidarseg_token = frame_id  # 在 NuScenes 中, frame_id 是 label_token
        if lidarseg_token is None:
            return
        
        # 路径: base_dir/lidarseg/label/v1.0-trainval/lidarseg_token.bin
        nusc_version = self.config.get('data', {}).get('nuscenes', {}).get('version', 'v1.0-trainval')
        label_dir = self.output_dir / "lidarseg" / "label" / nusc_version
        label_dir.mkdir(parents=True, exist_ok=True)
        
        pred_label_path = label_dir / f"{lidarseg_token}_lidarseg.bin"
        global_pred_inv = np.array([self.index_to_label.get(l, 0) for l in global_pred])
        pred_label_path.write_bytes(global_pred_inv.astype(np.uint8).tobytes())

    def save_global_points(self, cloud, pred, frame_id):
        # 路径: base_dir/global_points/seq_name/06d:index.pcd
        index = frame_id # 在 NuScenes 中, 我们将 index 作为 frame_id 传入
        output_dir = self.output_dir / "global_points" / self.sequence
        
        colors = self.get_colors(pred)
        labeled_points = np.concatenate([cloud[:, :3], colors], axis=1)
        save_all_point_cloud(labeled_points, index, output_dir)

    def save_error_points(self, cloud, pred, label, frame_id, active_classes):
        # 路径: base_dir/error_points/class/seq_name/06d:index.pcd
        index = frame_id
        output_dir = self.output_dir / "error_points"
        # save_error_point_cloud 内部会处理 class 和 seq_name 子目录
        save_error_point_cloud(cloud[:, :3], pred, label, index, output_dir, active_classes)

    def save_entropy(self, cloud, entropy_data, frame_id, entropy_type):
        # 路径: base_dir/{entropy_type}/seq_name/06d:index.pcd
        index = frame_id
        entropy_dir = self.output_dir / entropy_type / self.sequence
        entropy_dir.mkdir(parents=True, exist_ok=True)

        save_common_entropy(cloud[:, :3], entropy_data, index, entropy_dir)
        save_custom_npy(entropy_data, entropy_dir / f"{index:06d}.npy")

    def save_pred_prob(self, pred_prob, frame_id):
        lidarseg_token = frame_id
        if lidarseg_token is None:
            return
        # 路径: base_dir/lidarseg/pred_prob/v1.0-trainval/lidarseg_token.npz
        nusc_version = self.config.get('data', {}).get('nuscenes', {}).get('version', 'v1.0-trainval')
        prob_dir = self.output_dir / "lidarseg" / "pred_prob" / nusc_version
        prob_dir.mkdir(parents=True, exist_ok=True)
        
        prob_path = prob_dir / f"{lidarseg_token}_lidarseg.npz"
        np.savez_compressed(prob_path, pred_prob=pred_prob)

    def save_visualizations(self, detection_img, segmentation_img, frame_id, cam_name):
        index = frame_id
        # 保存分割图像
        if self.flags.get('save_segmented_images') and segmentation_img is not None:
            # 路径: base_dir/samples/seq_name/segementation_result/cam_name/06d:index.jpg
            img_dir = self.output_dir / 'samples' / self.sequence / 'segementation_result' / cam_name
            img_dir.mkdir(exist_ok=True, parents=True)
            img_path = img_dir / f"{index:06d}.jpg"
            cv2.imwrite(str(img_path), segmentation_img)
        
        # 保存检测图像
        if self.flags.get('save_detection_images') and detection_img is not None:
            # 路径: base_dir/samples/seq_name/detection_result/cam_name/06d:index.jpg
            img_dir = self.output_dir / 'samples' / self.sequence / 'detection_result' / cam_name
            img_dir.mkdir(exist_ok=True, parents=True)
            img_path = img_dir / f"{index:06d}.jpg"
            cv2.imwrite(str(img_path), detection_img)

    def save_projected_image(self, image, proj_points, frame_id, cam_name):
        """保存带有投影点云的图像。"""
        if not self.flags.get('save_local_points', False):
            return
        
        # 路径: basedir/model_name/sequences/seq_id/cam_name/proj_images/frame_id.png
        output_dir = self.output_dir / 'samples' / self.sequence / "proj_images" / cam_name 
        output_dir.mkdir(parents=True, exist_ok=True)
        
        vis_img = image.copy()
        for (v, u, r, g, b) in proj_points:
            cv2.circle(vis_img, (int(u), int(v)), 1, (int(b), int(g), int(r)), -1)
        
        output_path = output_dir / f"{frame_id}.png"
        cv2.imwrite(str(output_path), vis_img)

    def save_local_point_cloud(self, labeled_points, frame_id, cam_name):
        if not self.flags.get('save_local_points', False):
            return
        index = frame_id
        # 路径: base_dir/local_points/cam_name/seq_name/06d:index.pcd
        output_dir = self.output_dir / "local_points" / cam_name / self.sequence
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{index:06d}.pcd"

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(labeled_points[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(labeled_points[:, 3:6] / 255.0)
        o3d.io.write_point_cloud(str(output_path), pcd)
    
    def save_features(self, index, cam_name, cam_ground_indices, ground_probs, cam_nonground_indices, nonground_probs):
        """
        将来自单个相机视角的模型输出作为命名数组保存到 .npz 文件中。
        """
        try:
            sample = self.samples[index]
            camera_token = sample['data'][cam_name]
        except (IndexError, KeyError) as e:
            print(f"Warning: Could not get camera token for frame {index}, cam {cam_name}. Error: {e}")
            return
        
        # 路径: feature_output_path/features/sequence/cam_name/frame_id.npz
        output_dir = self.feature_path / "features/nuscenes" / cam_name 
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"{camera_token}.npz"
        
        # 构建要保存的数据字典，并过滤掉值为None的项
        features_to_save = {
            'cam_ground_indices': cam_ground_indices,
            'cam_ground_probs': ground_probs,
            'cam_nonground_indices': cam_nonground_indices,
            'cam_nonground_probs': nonground_probs
        }
        
        np.savez_compressed(output_path, features_to_save)

    def load_features(self, frame_id, cam_name):
        """
        从 .npz 文件中加载单个相机视角的模型输出。
        """
        # [修正] 简化和修正加载逻辑
        if not (self.load_external_feature and self.feature_path):
            return {}

        index = frame_id
        try:
            sample = self.samples[index]
            camera_token = sample['data'][cam_name]
        except (IndexError, KeyError) as e:
            print(f"Warning: Could not get camera token for frame {frame_id}, cam {cam_name} during load. Error: {e}")
            return {}

        # 构建与 save_features 相同的路径
        feature_file = self.feature_path / "features/nuscenes" / cam_name / f"{camera_token}.npz"
        
        if feature_file.exists():
            try:
                # 直接加载为字典
                data = np.load(feature_file, allow_pickle=True)
                return dict(data['arr_0'].item())
            except Exception as e:
                print(f"Warning: Could not load feature file {feature_file}. Error: {e}")
                return {}
        
        # 文件不存在是正常情况，返回空字典
        return {}