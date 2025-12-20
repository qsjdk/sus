import numpy as np
import yaml
from .default import BaseDataset
import cv2
import torch
from pathlib import Path
from utils.utils import (
    save_all_point_cloud,
    save_error_point_cloud,
    save_common_entropy,
    save_custom_npy
)
import open3d as o3d


class MeiTuanDataset(BaseDataset):
    """用于处理和加载美团数据集的类。"""

    def __init__(self,
                 config,
                 save_settings,
                 sequence,
                 global_output_dir):
        super().__init__(config=config, save_settings=save_settings, sequence=sequence,
                         global_output_dir=global_output_dir)

        dataset_cfg = self.config.get('dataset', {})
        if not dataset_cfg:
            raise ValueError("在主配置文件中未找到 'dataset' 配置。")

        # 从数据集特定YAML获取相机列表
        dataset_config_path = self._get_dataset_config_path('meituan')

        with open(dataset_config_path, 'r') as f:
            kitti_yaml = yaml.safe_load(f)
        self.cam_names = kitti_yaml.get('images', ['image_2'])

        # 根据新的目录结构构建路径
        dataset_root = Path(dataset_cfg['dataset_path']).expanduser().resolve()
        base_path = dataset_root / str(self.sequence)

        # 加载点云和标签路径
        self.cloud_paths = sorted(list((base_path / 'velodyne').glob('*.bin')))
        self.label_paths = sorted(list((base_path / 'labels').glob('*.label')))

        if not self.cloud_paths:
            raise FileNotFoundError(f"在路径 {base_path / 'velodyne'} 下找不到点云文件。请检查 'dataset_path' 配置。")

        # 为每个指定的相机加载图像路径
        self.image_paths = {}
        for cam_name in self.cam_names:
            self.image_paths[cam_name] = sorted(list((base_path / cam_name).glob('*.png')))

        # 加载标定和位姿 (假设它们在序列文件夹的根目录下)
        self.calib = self._read_calibration(base_path / 'calib.txt')
        self.poses = self._load_poses(base_path / 'poses.txt')
        # self.poses = self._load_poses_or_identity(base_path / 'poses.txt', n_frames=len(self.cloud_paths))

        # 配置网络特征加载 (用于 net_output)
        self.use_net = self.config.get('pcnetwork', {}).get('use_net', False)
        if self.use_net:
            net_feature_base_path = self.config.get('pcnetwork', {}).get('net_feature_path')
            if not net_feature_base_path:
                raise ValueError("pcnetwork.use_net 为 True, 但 pcnetwork.net_feature_path 未指定。")
            self.net_feature_path = Path(net_feature_base_path) / str(self.sequence)
        else:
            self.net_feature_path = None

        self.load_external_feature = self.config.get('external_feature', {}).get('load_feature', False)
        feature_cfg = self.config.get('external_feature', {})
        self.feature_path = Path(feature_cfg.get('feature_path')) if feature_cfg.get('feature_path') else None
        if feature_cfg.get('save_feature', False) and not self.feature_path:
            print(
                "Warning: external_feature.save_feature is True, but external_feature.feature_path is not specified. Features will not be saved.")
            self.feature_path = None  # 确保路径无效

        self.min_points = int(self.config.get("processing", {}).get("min_points", 2000)) # 设置2000与之，少于2000点的点云帧直接跳过

    def _setup_output_dir(self, global_output_dir):
        """设置并返回特定于此序列的输出目录。"""
        # 路径: basedir/model_name/sequences/seq_id
        seq_dir = global_output_dir / "sequences" / str(self.sequence)
        seq_dir.mkdir(parents=True, exist_ok=True)
        return seq_dir

    @staticmethod
    def get_sequences(dataset_cfg):
        return dataset_cfg.get('sequences', [])

    def _read_calibration(self, filepath):
        """
        读取 calib.txt，支持:
          - P0..P3 (3x4)
          - Tr0..Tr3 (3x4 -> 4x4)
        兼容：如果有 Tr 且没有 Tr0，就用 Tr 作为 Tr0；最后补齐 calib['Tr']=calib['Tr0']。
        """
        calib = {}
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or ':' not in line:
                    continue
                key, value = line.split(':', 1)
                key = key.strip()
                vals = value.strip().split()
                if len(vals) == 0:
                    continue
                calib[key] = np.array([float(x) for x in vals], dtype=float)

        # Reshape P matrices
        for i in range(4):
            k = f'P{i}'
            if k in calib:
                if calib[k].size != 12:
                    raise ValueError(f"[calib] {k} should have 12 numbers, got {calib[k].size}")
                calib[k] = calib[k].reshape(3, 4)

        def _reshape_Tr(k):
            if k in calib:
                if calib[k].size != 12:
                    raise ValueError(f"[calib] {k} should have 12 numbers, got {calib[k].size}")
                calib[k] = np.vstack((calib[k].reshape(3, 4), [0, 0, 0, 1]))

        # 兼容：如果你某天又加回 Tr，也能用
        _reshape_Tr('Tr')
        if 'Tr0' not in calib and 'Tr' in calib:
            calib['Tr0'] = calib['Tr']

        # 只读 Tr0..Tr3（你现在删掉 Tr 也完全OK）
        for i in range(4):
            _reshape_Tr(f'Tr{i}')

        # 如果缺 Tr0，直接报错（因为后面至少要有一个外参）
        if 'Tr0' not in calib:
            raise ValueError("[calib] Missing Tr0 in calib.txt (need at least Tr0).")

        # 若 Tr1/Tr2/Tr3 缺失，fallback 用 Tr0，保证程序能继续跑完
        for i in range(1, 4):
            k = f'Tr{i}'
            if k not in calib:
                print(f"[WARN] {k} missing in calib.txt, fallback to Tr0")
                calib[k] = calib['Tr0'].copy()

        # 兼容旧代码：如果别处还在用 self.calib['Tr']，这里也给它补齐
        calib['Tr'] = calib['Tr0']

        return calib

    def _load_poses(self, filepath):
        """读取poses.txt文件。"""
        poses = []
        with open(filepath, 'r') as f:
            for line in f:
                pose = np.fromstring(line.strip(), sep=' ').reshape(3, 4)
                poses.append(np.vstack((pose, [0, 0, 0, 1])))
        return poses

    def __len__(self):
        return len(self.cloud_paths)

    def __getitem__(self, index):
        # 1. 加载点云
        cloud_path = self.cloud_paths[index]
        cloud = np.fromfile(cloud_path, dtype=np.float32).reshape(-1, 4)

        # ✅ 帧级跳过：点数太少直接跳过（避免后续所有流程）
        npts = 0 if cloud is None else int(cloud.shape[0])
        frame_id = cloud_path.stem
        if cloud is None or npts < self.min_points:
            return {
                "skip": True,
                "reason": f"too_few_points(<{self.min_points})",
                "num_points": npts,
                "label_token": frame_id,
                "index": index,
            }

        # 2. 加载标签并映射
        label_path = self.label_paths[index]
        raw_label = np.fromfile(label_path, dtype=np.uint32).reshape(-1)
        raw_label = raw_label & 0xFFFF  # 取低16位为语义标签
        label = np.array([self.label_to_index.get(l, 0) for l in raw_label], dtype=np.int32)

        # ✅ 宽泛一点：标签和点云长度不一致也跳过/或截断（二选一）
        if label.shape[0] != cloud.shape[0]:
            # 更“保守不出错”的做法：直接跳过
            return {
                "skip": True,
                "reason": "label_cloud_length_mismatch",
                "num_points": npts,
                "label_len": int(label.shape[0]),
                "label_token": frame_id,
                "index": index,
            }

        # 3. 获取位姿和标定
        ego_to_global = self.poses[index]  # 这是从当前帧相机坐标系到第一帧相机坐标系的变换
        lidar_to_cam0 = self.calib['Tr']  # 从雷达坐标系到相机0坐标系的变换

        # 4. 将点云转换到全局坐标系 (世界坐标系为第一帧的相机坐标系)
        cloud_cam0 = self.transform(cloud, lidar_to_cam0)
        transformed_cloud = self.transform(cloud_cam0, ego_to_global)

        images = {}
        proj_cloud = {}
        # 5. 为每个相机加载图像并计算投影
        for cam_name in self.cam_names:  # e.g., 'image_2'
            image_path = self.image_paths[cam_name][index]
            image = cv2.imread(str(image_path))
            # images[cam_name] = torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            images[cam_name] = torch.from_numpy(image)

            # 获取对应相机的投影矩阵
            cam_idx = int(cam_name[-1])  # 'image_2' -> 2
            P = self.calib[f'P{cam_idx}']

            # 构建从雷达到该相机图像的完整变换
            P_4x4 = np.eye(4)
            P_4x4[:3, :] = P
            # 变换链：点云(雷达) -> cam0 -> 图像
            lidar_to_image_transform = P_4x4 @ lidar_to_cam0


            # ✅ project_points_to_camera 可能返回 2 个值/或异常
            try:
                ret = self.project_points_to_camera(cloud, lidar_to_image_transform, image.shape)
            except Exception:
                ret = None

            if ret is None or (isinstance(ret, (tuple, list)) and len(ret) != 3):
                pixels = np.empty((0, 2), dtype=np.int32)
                points = np.empty((0, 3), dtype=np.float32)
                indices = np.empty((0,), dtype=np.int64)
            else:
                pixels, points, indices = self.project_points_to_camera(cloud, lidar_to_image_transform, image.shape)

            proj_cloud[cam_name] = {'pixels': pixels, 'points': points, 'indices': indices}

        frame_id = cloud_path.stem

        # 加载 pcnetwork 的预计算特征 -> net_output
        net_output_data = np.array([])
        if self.use_net and self.net_feature_path:
            feature_file = self.net_feature_path / f"{frame_id}.npz"
            if feature_file.exists():
                with np.load(feature_file) as data:
                    # 假设键名是 'pred_prob'
                    net_output_data = data.get('pred_prob', np.array([]))
            else:
                print(f"警告: PC网络特征文件 {feature_file} 未找到。")

        # 加载所有相机的 external_feature
        model_predict_data = {}
        if self.load_external_feature:
            for cam_name in self.cam_names:
                model_predict_data[cam_name] = self.load_features(index, cam_name)

        return {
            "images": {k: v.numpy() for k, v in images.items()},
            "cloud": cloud.astype(np.float32),
            "transformed_cloud": transformed_cloud.astype(np.float32),
            "label": label.astype(np.int32),
            "pose": ego_to_global.astype(np.float32),
            "proj_cloud": proj_cloud,
            "net_output": torch.from_numpy(net_output_data.astype(np.float32)),
            "label_token": frame_id,
            "model_predict": model_predict_data,
            "index": index
        }

    def save_label(self, global_pred, frame_id):
        """保存标签到预设目录"""
        # 路径: basedir/model_name/sequences/seq_id/pred_labels/frame_id.label
        pred_label_dir = self.output_dir / "pred_labels"
        pred_label_dir.mkdir(parents=True, exist_ok=True)

        pred_label_path = pred_label_dir / f"{frame_id}.label"
        global_pred_inv = np.array([self.index_to_label.get(l, 0) for l in global_pred])
        pred_label_path.write_bytes(global_pred_inv.astype(np.uint32).tobytes())

    def save_global_points(self, cloud, pred, frame_id):
        """保存带颜色的全局点云。"""
        # 路径: basedir/model_name/sequences/seq_id/all_points/frame_id.pcd
        colors = self.get_colors(pred)
        labeled_points = np.concatenate([cloud[:, :3], colors], axis=1)
        save_all_point_cloud(labeled_points, frame_id, self.output_dir)

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

    def save_error_points(self, cloud, pred, label, frame_id, active_classes):
        """保存误差点云。"""
        # 路径: basedir/model_name/sequences/seq_id/error_points/class/frame_id_errornums.label
        save_error_point_cloud(cloud[:, :3], pred, label, frame_id, self.output_dir, active_classes)

    def save_entropy(self, cloud, entropy_data, frame_id, entropy_type):
        """保存熵的可视化点云和原始数据。"""
        # 路径: basedir/model_name/sequences/seq_id/{entropy_type}/frame_id.pcd
        entropy_dir = self.output_dir / entropy_type
        save_common_entropy(cloud[:, :3], entropy_data, frame_id, entropy_dir)
        save_custom_npy(entropy_data, entropy_dir / f"{frame_id}.npy")

    def save_pred_prob(self, pred_prob, frame_id):
        """保存预测的概率分布。"""
        # 路径: basedir/model_name/sequences/seq_id/pred_probs/frame_id.npz
        prob_dir = self.output_dir / "pred_probs"
        prob_dir.mkdir(parents=True, exist_ok=True)
        prob_path = prob_dir / f"{frame_id}.npz"
        np.savez_compressed(prob_path, pred_prob=pred_prob)

    def save_visualizations(self, detection_img, segmentation_img, frame_id, cam_name):
        """保存每个相机视角的可视化结果。"""
        # 保存检测图像
        if self.flags.get('save_detection_images') and detection_img is not None:
            # 路径: basedir/model_name/sequences/seq_id/cam_name/detection_images/frame_id.jpg
            img_dir = self.output_dir / cam_name / 'detection_images'
            img_dir.mkdir(exist_ok=True, parents=True)
            img_path = img_dir / f"{frame_id}.jpg"
            cv2.imwrite(str(img_path), detection_img)

        # 保存分割图像
        if self.flags.get('save_segmented_images') and segmentation_img is not None:
            # 路径: basedir/model_name/sequences/seq_id/cam_name/segmentation_images/frame_id.jpg
            img_dir = self.output_dir / cam_name / 'segmentation_images'
            img_dir.mkdir(exist_ok=True, parents=True)
            img_path = img_dir / f"{frame_id}.jpg"
            cv2.imwrite(str(img_path), segmentation_img)

    def save_projected_image(self, image, proj_points, frame_id, cam_name):
        """保存带有投影点云的图像。"""
        if not self.flags.get('save_local_points', False):
            return

        # 路径: basedir/model_name/sequences/seq_id/cam_name/proj_images/frame_id.png
        output_dir = self.output_dir / cam_name / "proj_images"
        output_dir.mkdir(parents=True, exist_ok=True)

        vis_img = image.copy()
        for (v, u, r, g, b) in proj_points:
            cv2.circle(vis_img, (int(u), int(v)), 1, (int(b), int(g), int(r)), -1)

        output_path = output_dir / f"{frame_id}.png"
        cv2.imwrite(str(output_path), vis_img)

    def save_features(self, index, cam_name, cam_ground_indices, ground_probs, cam_nonground_indices, nonground_probs):
        """
        将来自单个相机视角的模型输出作为命名数组保存到 .npz 文件中。
        """
        if self.feature_path is None:
            return

        # 路径: feature_output_path/features/sequence/cam_name/frame_id.npz
        output_dir = self.feature_path / "features/meituan" / str(self.sequence) / cam_name
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"{index:06d}.npz"

        # 构建要保存的数据字典，并过滤掉值为None的项
        features_to_save = {
            'cam_ground_indices': cam_ground_indices,
            'cam_ground_probs': ground_probs,
            'cam_nonground_indices': cam_nonground_indices,
            'cam_nonground_probs': nonground_probs
        }

        if features_to_save:
            np.savez_compressed(output_path, features_to_save)

    def load_features(self, frame_id, cam_name):
        """
        为指定的相机和帧加载预计算的特征。
        """
        # [修正] 简化加载逻辑，移除内部循环
        if not (self.load_external_feature and self.feature_path):
            return {}

        feature_file = self.feature_path / "features/meituan" / str(
            self.sequence) / cam_name / f"{frame_id:06d}.npz"

        if feature_file.exists():
            try:
                # 直接加载为字典
                data = np.load(feature_file, allow_pickle=True)
                return dict(data['arr_0'].item())
            except Exception as e:
                print(f"警告: 无法加载特征文件 {feature_file}。错误: {e}")
                return {}

        # 文件不存在是正常情况，返回空字典
        return {}