"""
Default Datasets

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import glob
import numpy as np
import torch
from copy import deepcopy
from torch.utils.data import Dataset
from collections.abc import Sequence

from pathlib import Path
from abc import ABC, abstractmethod
import joblib
import yaml
import cv2
import open3d as o3d

class BaseDataset(Dataset, ABC):
    """数据集的抽象基类，定义所有数据集通用的接口。"""
    def __init__(self, config, save_settings, sequence, global_output_dir):
        print("init default")
        self.config = config
        self.save_settings = save_settings
        self.flags = save_settings['flags']
        self.sequence = sequence
        self.n_jobs = joblib.cpu_count()
        self.dataset_type = self.config.get('dataset', {}).get('type', 'semantic_kitti').lower()
        self.dataset_config_path = self._get_dataset_config_path(self.dataset_type)
        with open(self.dataset_config_path, 'r') as f:
            self.yaml_config = yaml.safe_load(f)
        self._init_label_mapping()
        self._init_relation_mapping()
        self.output_dir = self._setup_output_dir(global_output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    
    def _init_relation_mapping(self):
        # relation 映射为类别索引
        raw_relation = self.yaml_config.get('relation', {})
        # name_to_final_id: 类名 -> 索引（从1开始）
        self.relation = {}
        all_keys = list(self.name_to_index.keys())
        for k, v_list in raw_relation.items():
            # k, v_list 都是类别名
            if k in self.name_to_index:
                k_idx = self.name_to_index[k]
                v_idx_list = [self.name_to_index[v] for v in v_list if v in self.name_to_index]
                self.relation[k_idx] = v_idx_list
            
    # def _init_label_mapping(self):
        """初始化标签映射功能，类别定义完全来自数据集的YAML配置。"""

        # # 2. 从YAML的name_map直接确定所有活动类别
        # self.name_to_original_id = self.yaml_config.get('name_map', {})
        # self.active_classes = list(self.name_to_original_id.keys())

        # ground_class_map = self.yaml_config.get('ground_class', {})
        # self.ground_classes = sorted([name for name in self.active_classes if ground_class_map.get(name)])
        # self.nonground_classes = sorted([name for name in self.active_classes if not ground_class_map.get(name)])
        
        # self.ground_class_indices = [self.active_classes.index(name) for name in self.ground_classes]
        # self.nonground_class_indices = [self.active_classes.index(name) for name in self.nonground_classes]

        # # 3. 构建所有映射
        # original_id = np.unique(list(self.name_to_original_id.values()))
        # label_to_id = {v: k + 1 for k, v in enumerate(original_id)}  # 从1开始编号
        # self.name_to_final_id = {name: label_to_id[orig_id] for name, orig_id in self.name_to_original_id.items()}
        # self.learning_map = self.yaml_config.get('learning_map', {}) 

        # report_name_map = self.yaml_config.get('report_name_map', {})
        # self.report_name = {}
        # for name, idx in report_name_map.items():
        #     self.report_name[idx] = name
        
        # color_map_original = self.yaml_config.get('color_map', {})
        # self.final_id_to_color = {0: [0, 0, 0]}
        # self.final_id_to_color.update({
        #     self.name_to_final_id[name]: color_map_original.get(self.name_to_original_id[name], np.random.randint(0, 255, 3).tolist())
        #     for name in self.active_classes if name in self.name_to_original_id
        # })
        
        # self.order_label = {final_id: self.name_to_original_id.get(name, 0) for name, final_id in self.name_to_final_id.items()}
        # self.order_label[0] = 0
        # self.label_order = {v: k for k, v in self.order_label.items()}
        
    def _init_label_mapping(self):
        self.name_to_label = self.yaml_config.get('name_map', {})
        self.active_classes = list(self.name_to_label.keys())

        self.label_to_index = self.yaml_config.get('learning_map', {})
        self.index_to_label = {v: k for k, v in self.label_to_index.items()}
        
        self.name_to_index = {name: self.label_to_index[label] for name, label in self.name_to_label.items()}
        
        ground_class_map = self.yaml_config.get('ground_class', {})
        self.ground_classes = sorted([name for name in self.active_classes if ground_class_map.get(name)])
        self.nonground_classes = sorted([name for name in self.active_classes if not ground_class_map.get(name)])
        
        self.ground_indices = [self.name_to_index[name] for name in self.ground_classes]
        self.nonground_indices = [self.name_to_index[name] for name in self.nonground_classes]
        
        color_map = self.yaml_config.get('color_map', {})
        self.index_to_color = {0: [0, 0, 0]}
        self.index_to_color.update({
            self.name_to_index[name]: color_map.get(self.name_to_label[name], np.random.randint(0, 255, 3).tolist())
            for name in self.active_classes if name in self.name_to_label
        })
        
        report_name_map = self.yaml_config.get('report_name_map', {})
        self.report_name = {}
        for name, idx in report_name_map.items():
            self.report_name[self.label_to_index[idx]] = name

    def _get_dataset_config_path(self, dataset_type):
        """获取配置文件路径"""
        config_map = {
            'semantic_kitti': 'dataset/config/semantickitti.yaml',
            'nuscenes': 'dataset/config/nuscenes.yaml',
            'meituan': 'dataset/config/meituan.yaml'  # ✅ 新增
        }
        base_path = Path(__file__).parent.parent
        return base_path / config_map.get(dataset_type, config_map['semantic_kitti'])
    
    @staticmethod
    def transform(points, transform_matrix_val):
        """应用变换矩阵到点云"""
        points_h = np.hstack((points[:, :3], np.ones((points.shape[0], 1))))
        transformed_points = (transform_matrix_val @ points_h.T).T
        return np.hstack((transformed_points[:, :3], points[:, 3:]))

    @staticmethod
    def transform_inv(points, transform_matrix_val):
        """应用逆变换矩阵到点云"""
        inv_transform = np.linalg.inv(transform_matrix_val)
        return BaseDataset.transform(points, inv_transform)

    @staticmethod
    def project_points_to_camera(points_global: np.ndarray, global_to_image_transform: np.ndarray, image_shape: tuple):
        """将全局坐标系下的点云投影到单个相机图像。"""
        if points_global.shape[0] == 0:
            return {}, []
        #if points_global is None or points_global.shape[0] == 0:
            #return np.zeros((0, 2), np.int32), points_global[:0], np.zeros((0,), np.int64) # 防止没有点云投影而无法运行

        points_h = np.hstack([points_global[:, :3], np.ones((points_global.shape[0], 1))])
        points_image_h = (global_to_image_transform @ points_h.T).T
        depths = points_image_h[:, 2]
        pixels = points_image_h[:, :2] / (depths[:, np.newaxis] + 1e-8)

        img_h, img_w = image_shape[:2]
        mask = (depths > 1.0) & (pixels[:, 0] >= 0) & (pixels[:, 0] < img_w) & (pixels[:, 1] >= 0) & (pixels[:, 1] < img_h)
        
        valid_indices = np.where(mask)[0]
        if valid_indices.size == 0:
            return {}, []
            
        valid_points = points_global[valid_indices]
        valid_pixels_xy = pixels[valid_indices].astype(np.int32)
        valid_pixels_vu = valid_pixels_xy[:, ::-1] 
        
        return valid_pixels_vu, valid_points, valid_indices

    def get_classes_by_type(self, class_type='all'):
        """根据类型获取类别列表"""
        if class_type == 'ground':
            return [name for name in self.active_classes if name in self.ground_classes]
        elif class_type == 'non_ground':
            return [name for name in self.active_classes if name in self.nonground_classes]
        else:
            return self.active_classes
    
    def _get_config_path(self, dataset_type):
        """获取配置文件路径"""
        config_map = {
            'semantic_kitti': 'dataset/config/semantickitti.yaml',
            'nuscenes': 'dataset/config/nuscenes.yaml',
            'meituan': 'dataset/config/meituan.yaml'  # ✅ 新增
        }
        base_path = Path(__file__).parent.parent
        return base_path / config_map.get(dataset_type, config_map['semantic_kitti'])
    
    def get_colors(self, idxs):
        """获取颜色"""
        return np.array([self.index_to_color.get(idx, [0, 0, 0]) for idx in idxs], dtype=np.uint8)
    
    def colorize_points(self, points, label_ids):
        if points.size == 0 or label_ids.size == 0:
            return np.array([])
        
        colors = self.get_colors(label_ids)
        return np.hstack((points, colors))

    def init_time_log(self):
        """初始化时间记录文件。"""
        if self.flags.get('save_time'):
            self.time_file = self.output_dir / "time.txt"
            with open(self.time_file, 'w') as f:
                f.write("frame_id detection_time segment_time\n")


    def log_time(self, frame_id, detection_time, segment_time):
        """记录一帧的处理时间。"""
        if self.flags.get('save_time'):
            with open(self.time_file, 'a') as f:
                f.write(f"{frame_id} {detection_time:.4f} {segment_time:.4f}\n")

    @staticmethod
    def file_exist(file_path):
        """检查文件是否存在。"""
        return Path(file_path).is_file()
    
    def save(self, output):
        """统一的保存框架"""
        # 1. 预处理
        frame_id = output['frame_id']
        label_token = output.get('label_token', frame_id)
        final_cloud = self.transform_inv(output['cloud'], output['pose'])
        
        # 2. 定义保存配置
        saves = [
            ('save_label', self.save_label, 
             [output['pred'], label_token]),
            
            ('save_global_points', self.save_global_points, 
             [final_cloud, output['pred'], frame_id]),
            
            ('save_error_points', self.save_error_points, 
             [final_cloud, output['pred'], output['label'], frame_id, self.active_classes]),
            
            # ('save_common_entropy', self.save_entropy, 
            #  [final_cloud, output['common_entropy'], frame_id, 'common_entropy', self.seq_output_dir]),
            
            # ('save_dir_expected_entropy', self.save_entropy, 
            #  [final_cloud, output['dir_expected_entropy'], frame_id, 'dir_expected_entropy', self.seq_output_dir]),
            
            # ('save_diff_dir_entropy', self.save_entropy, 
            #  [final_cloud, output['diff_dir_entropy'], frame_id, 'diff_dir_entropy', self.seq_output_dir]),
            ('save_pred_prob', self.save_pred_prob, 
             [output['pred_prob'], label_token])
        ]

        # 3. 执行保存
        for flag, method, args in saves:
            if self.flags.get(flag):
                method(*args)
    
    @abstractmethod
    def save_visualizations(self, detection_img, segmentation_img, frame_id, cam_name):
        """保存每个相机视角的可视化结果。"""
        pass
        
    @abstractmethod
    def save_projected_image(self, image, proj_points, frame_id, cam_name):
        """保存带有投影点云的图像。"""
        pass
    
    @abstractmethod
    def save_local_point_cloud(self, labeled_points, frame_id, cam_name):
        """保存相机视角的局部点云。"""
        pass
        
    @abstractmethod
    def save_label(self, global_pred, frame_id, seq_output_dir):
        """保存预测的标签。"""
        pass

    @abstractmethod
    def save_global_points(self, cloud, pred, frame_id, seq_output_dir):
        """保存带颜色的全局点云。"""
        pass

    @abstractmethod
    def save_error_points(self, cloud, pred, label, frame_id, seq_output_dir, active_classes):
        """保存误差点云。"""
        pass

    @abstractmethod
    def save_entropy(self, cloud, entropy_data, frame_id, entropy_type, seq_output_dir):
        """保存熵的可视化点云和原始数据。"""
        pass

    @abstractmethod
    def save_pred_prob(self, pred_prob, frame_id, seq_output_dir):
        """保存预测的概率分布。"""
        pass

    @abstractmethod
    def get_sequences(dataset_cfg):
        """根据数据集配置获取序列/场景列表。"""
        pass
    
    @abstractmethod
    def _setup_output_dir(self, global_output_dir):
        """设置并返回特定于此序列的输出目录。"""
        pass
    
    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, index):
        pass

    @abstractmethod
    def save_features(self, output_data, frame_id,  cam_name):
        """
        保存模型特征到文件。
        """
        pass