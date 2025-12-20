import numpy as np
import yaml
import cv2
import torch
from pathlib import Path
from bisect import bisect_left
import open3d as o3d

from .default import BaseDataset
from utils.utils import (
    save_all_point_cloud,
    save_error_point_cloud,
    save_common_entropy,
    save_custom_npy
)


class MeiTuanDataset(BaseDataset):
    """用于处理和加载美团数据集的类（兼容序号命名 & 时间戳命名）。"""

    def __init__(self, config, save_settings, sequence, global_output_dir):
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

        # 1) 读取原始路径（先不假设一一对齐）
        self.cloud_paths_raw = sorted(list((base_path / 'velodyne').glob('*.bin')))
        self.label_paths_raw = sorted(list((base_path / 'labels').glob('*.label')))

        if not self.cloud_paths_raw:
            raise FileNotFoundError(f"在路径 {base_path / 'velodyne'} 下找不到点云文件。请检查 'dataset_path' 配置。")
        if not self.label_paths_raw:
            raise FileNotFoundError(f"在路径 {base_path / 'labels'} 下找不到label文件。请检查 'dataset_path' 配置。")

        # 读取每个相机的图片路径
        self.image_paths_raw = {}
        for cam_name in self.cam_names:
            self.image_paths_raw[cam_name] = sorted(list((base_path / cam_name).glob('*.png')))

        # 2) 加载标定和位姿
        self.calib = self._read_calibration(base_path / 'calib.txt')
        self.poses = self._load_poses(base_path / 'poses.txt')

        # 3) ✅ 关键：以 label 的文件名为索引，对齐 cloud & images（兼容两种命名）
        # 可选：你也可以在 config.yaml 里加 max_time_diff_ms 来限制匹配跨度
        # 例如：dataset: { max_time_diff_ms: 20 }
        self.max_time_diff_ms = dataset_cfg.get('max_time_diff_ms', None)
        self.max_time_diff_key = None
        if self.max_time_diff_ms is not None:
            # 对时间戳（ns）有意义；对序号命名一般也没坏处（只是更严格）
            self.max_time_diff_key = float(self.max_time_diff_ms)

        self._align_by_label_index()

        # 输出目录
        self.output_dir = self._setup_output_dir(global_output_dir)

    # ----------------------------
    # 输出目录
    # ----------------------------
    def _setup_output_dir(self, global_output_dir):
        """设置并返回特定于此序列的输出目录。"""
        seq_dir = global_output_dir / "sequences" / str(self.sequence)
        seq_dir.mkdir(parents=True, exist_ok=True)
        return seq_dir

    @staticmethod
    def get_sequences(dataset_cfg):
        return dataset_cfg.get('sequences', [])

    # ----------------------------
    # Key parsing + nearest search
    # ----------------------------
    @staticmethod
    def _parse_key_from_stem(stem: str):
        """
        支持：
          - '000000' -> 0
          - '1747106157757263374' -> 1747106157757263374
          - '1747106157.757263374' -> 转成 ns int
        返回 (key_int, token_str)；失败返回 (None, token_str)
        """
        token = stem
        s = stem.strip()

        if s.isdigit():
            return int(s), token

        # 支持带小数点的秒表示
        # 1747106157.757263374 -> 1747106157757263374
        if "." in s:
            a, b = s.split(".", 1)
            if (a.isdigit()) and (b.isdigit()):
                b = (b + "0" * 9)[:9]
                return int(a) * 1_000_000_000 + int(b), token

        # 兜底：抽取数字
        digits = "".join(ch for ch in s if ch.isdigit())
        if digits:
            return int(digits), token

        return None, token

    @staticmethod
    def _build_index(paths):
        """
        输入 Path 列表
        输出：
          keys_sorted: [k...]
          key2path: {k: Path}
          key2pos: {k: index_in_sorted}
          key2token: {k: original_stem_str}
        """
        tmp = []
        for p in paths:
            k, token = MeiTuanDataset._parse_key_from_stem(p.stem)
            if k is None:
                continue
            tmp.append((k, p, token))

        tmp.sort(key=lambda x: x[0])
        key2path = {}
        key2token = {}
        keys_sorted = []
        for k, p, token in tmp:
            # 去重：保留第一个
            if k in key2path:
                continue
            key2path[k] = p
            key2token[k] = token
            keys_sorted.append(k)

        key2pos = {k: i for i, k in enumerate(keys_sorted)}
        return keys_sorted, key2path, key2pos, key2token

    @staticmethod
    def _nearest_key(target, keys_sorted):
        if not keys_sorted:
            return None
        pos = bisect_left(keys_sorted, target)
        if pos == 0:
            return keys_sorted[0]
        if pos >= len(keys_sorted):
            return keys_sorted[-1]
        a = keys_sorted[pos - 1]
        b = keys_sorted[pos]
        return b if abs(b - target) < abs(target - a) else a

    def _key_is_timestamp_like(self, keys_sorted):
        """
        简单判断：如果 key 很大（> 1e12），通常是 ns 时间戳；否则更像序号。
        """
        if not keys_sorted:
            return False
        return keys_sorted[len(keys_sorted)//2] > 1_000_000_000_000

    def _align_by_label_index(self):
        """
        ✅ 核心对齐逻辑：
        以 label 为索引，label_key -> 最近 cloud_key / image_key（每个cam）
        数据集长度 == label 数量（自然跳过 label 不存在的帧）
        """
        # 建索引
        label_keys, label_k2p, _, label_k2token = self._build_index(self.label_paths_raw)
        cloud_keys, cloud_k2p, cloud_k2pos, cloud_k2token = self._build_index(self.cloud_paths_raw)

        cam_index = {}
        for cam in self.cam_names:
            cam_keys, cam_k2p, _, cam_k2token = self._build_index(self.image_paths_raw.get(cam, []))
            cam_index[cam] = (cam_keys, cam_k2p, cam_k2token)

        if not label_keys:
            raise RuntimeError("无法从 label 文件名解析出有效 key（可能文件名不是数字/时间戳）。")
        if not cloud_keys:
            raise RuntimeError("无法从 bin 文件名解析出有效 key（可能文件名不是数字/时间戳）。")

        is_ts = self._key_is_timestamp_like(label_keys) or self._key_is_timestamp_like(cloud_keys)

        # 如果你设置了 max_time_diff_ms，且是时间戳模式：把 ms 转成 ns 的 key 差
        max_delta_key = None
        if self.max_time_diff_ms is not None and is_ts:
            max_delta_key = int(float(self.max_time_diff_ms) * 1e6)  # ms -> ns
        elif self.max_time_diff_ms is not None and not is_ts:
            # 序号模式下 max_time_diff_ms 没意义，这里忽略
            max_delta_key = None

        # 对齐输出
        aligned_label_paths = []
        aligned_label_tokens = []
        aligned_cloud_paths = []
        aligned_cloud_pose_idx = []
        aligned_image_paths = {cam: [] for cam in self.cam_names}

        drop_no_cloud = 0
        drop_over_threshold = 0

        for lk in label_keys:
            lp = label_k2p[lk]
            ltoken = label_k2token[lk]

            ck = self._nearest_key(lk, cloud_keys)
            if ck is None:
                drop_no_cloud += 1
                continue

            delta = abs(ck - lk)
            if max_delta_key is not None and delta > max_delta_key:
                # 超阈值：认为不匹配，直接丢弃这个 label
                drop_over_threshold += 1
                continue

            aligned_label_paths.append(lp)
            aligned_label_tokens.append(ltoken)

            cp = cloud_k2p[ck]
            aligned_cloud_paths.append(cp)

            # poses 默认与 cloud（按 key 排序后）一一对应
            pose_idx = cloud_k2pos[ck]
            if pose_idx >= len(self.poses):
                # poses 行数不足时兜底：用最后一个
                pose_idx = len(self.poses) - 1
            aligned_cloud_pose_idx.append(pose_idx)

            # images per cam
            for cam in self.cam_names:
                cam_keys, cam_k2p, _ = cam_index[cam]
                ik = self._nearest_key(lk, cam_keys)
                if ik is None:
                    aligned_image_paths[cam].append(None)
                    continue
                if max_delta_key is not None:
                    if abs(ik - lk) > max_delta_key:
                        aligned_image_paths[cam].append(None)
                        continue
                aligned_image_paths[cam].append(cam_k2p[ik])

        # 覆盖到 dataset 使用的字段
        self.label_paths = aligned_label_paths
        self.cloud_paths = aligned_cloud_paths
        self.image_paths = aligned_image_paths
        self._pose_indices = aligned_cloud_pose_idx
        self._label_tokens = aligned_label_tokens

        print(f"[ALIGN] mode={'timestamp(ns)' if is_ts else 'index'} "
              f"labels(raw)={len(self.label_paths_raw)} -> labels(used)={len(self.label_paths)} "
              f"clouds(raw)={len(self.cloud_paths_raw)} imgs(raw)={ {k:len(v) for k,v in self.image_paths_raw.items()} }")
        if max_delta_key is not None:
            print(f"[ALIGN] max_time_diff_ms={self.max_time_diff_ms} -> max_delta_ns={max_delta_key}, "
                  f"dropped(no_cloud)={drop_no_cloud}, dropped(over_threshold)={drop_over_threshold}")

        # 额外：打印各相机缺失情况
        for cam in self.cam_names:
            miss = sum(1 for p in self.image_paths[cam] if p is None)
            print(f"[ALIGN] {cam}: missing_images={miss}/{len(self.label_paths)}")

    # ----------------------------
    # calib / poses
    # ----------------------------
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

        for i in range(4):
            _reshape_Tr(f'Tr{i}')

        if 'Tr0' not in calib:
            raise ValueError("[calib] Missing Tr0 in calib.txt (need at least Tr0).")

        # 若某些 Tr 缺失，fallback 到 Tr0
        for i in range(4):
            k = f'Tr{i}'
            if k not in calib:
                print(f"[WARN] {k} missing in calib.txt, fallback to Tr0")
                calib[k] = calib['Tr0'].copy()

        # 兼容旧代码：别处还在用 self.calib['Tr']
        calib['Tr'] = calib['Tr0']
        return calib

    def _load_poses(self, filepath):
        """读取poses.txt文件。"""
        poses = []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                pose = np.fromstring(line, sep=' ').reshape(3, 4)
                poses.append(np.vstack((pose, [0, 0, 0, 1])))
        return poses

    # ----------------------------
    # Dataset API
    # ----------------------------
    def __len__(self):
        # ✅ 以 label 为主：长度就是对齐后的 label 数量
        return len(self.label_paths)

    def __getitem__(self, index):
        # label token（保留原文件名样式：000000 或 时间戳）
        label_token = self._label_tokens[index]

        # 1) 加载点云
        cloud_path = self.cloud_paths[index]
        cloud = np.fromfile(cloud_path, dtype=np.float32).reshape(-1, 4)

        # ✅ 帧级跳过：点数太少
        npts = 0 if cloud is None else int(cloud.shape[0])
        if cloud is None or npts < self.min_points:
            return {
                "skip": True,
                "reason": f"too_few_points(<{self.min_points})",
                "num_points": npts,
                "label_token": label_token,
                "index": index,
            }

        # 2) 加载标签并映射
        label_path = self.label_paths[index]
        raw_label = np.fromfile(label_path, dtype=np.uint32).reshape(-1)
        raw_label = raw_label & 0xFFFF
        label = np.array([self.label_to_index.get(l, 0) for l in raw_label], dtype=np.int32)

        # 标签点数不等于点云点数 -> 跳过（保守策略）
        if label.shape[0] != cloud.shape[0]:
            return {
                "skip": True,
                "reason": "label_cloud_length_mismatch",
                "num_points": npts,
                "label_len": int(label.shape[0]),
                "label_token": label_token,
                "index": index,
            }

        # 3) 获取位姿和标定（pose 用对齐后的 cloud pose index）
        pose_idx = self._pose_indices[index]
        ego_to_global = self.poses[pose_idx]
        lidar_to_cam0 = self.calib['Tr']

        # 4) 转到全局
        cloud_cam0 = self.transform(cloud, lidar_to_cam0)
        transformed_cloud = self.transform(cloud_cam0, ego_to_global)

        # 5) 为每个相机加载图像并计算投影（缺图则跳过该cam）
        images = {}
        proj_cloud = {}

        for cam_name in self.cam_names:
            image_path = self.image_paths[cam_name][index]
            if image_path is None:
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                continue

            images[cam_name] = torch.from_numpy(image)

            cam_idx = int(cam_name[-1])  # image_2 -> 2
            P = self.calib[f'P{cam_idx}']

            P_4x4 = np.eye(4, dtype=np.float32)
            P_4x4[:3, :] = P
            lidar_to_image_transform = P_4x4 @ lidar_to_cam0

            try:
                ret = self.project_points_to_camera(cloud, lidar_to_image_transform, image.shape)
            except Exception:
                ret = None

            if ret is None or (isinstance(ret, (tuple, list)) and len(ret) != 3):
                pixels = np.empty((0, 2), dtype=np.int32)
                points = np.empty((0, 3), dtype=np.float32)
                indices = np.empty((0,), dtype=np.int64)
            else:
                pixels, points, indices = ret

            proj_cloud[cam_name] = {'pixels': pixels, 'points': points, 'indices': indices}

        # pcnetwork 特征（用 cloud token 读取更合理）
        cloud_token = cloud_path.stem
        net_output_data = np.array([])
        if self.use_net and self.net_feature_path:
            feature_file = self.net_feature_path / f"{cloud_token}.npz"
            if feature_file.exists():
                with np.load(feature_file) as data:
                    net_output_data = data.get('pred_prob', np.array([]))
            else:
                # 不强制报错
                pass

        # external_feature（也用 label_token/cam 对应的文件名 token 更稳）
        model_predict_data = {}
        if self.load_external_feature:
            for cam_name in self.cam_names:
                model_predict_data[cam_name] = self.load_features(label_token, cam_name)

        return {
            "images": {k: v.numpy() for k, v in images.items()},
            "cloud": cloud.astype(np.float32),
            "transformed_cloud": transformed_cloud.astype(np.float32),
            "label": label.astype(np.int32),
            "pose": ego_to_global.astype(np.float32),
            "proj_cloud": proj_cloud,
            "net_output": torch.from_numpy(net_output_data.astype(np.float32)),
            "label_token": label_token,
            "cloud_token": cloud_token,   # ✅ 额外返回（不影响原逻辑）
            "model_predict": model_predict_data,
            "index": index
        }

    # ----------------------------
    # Save helpers（兼容两种命名：直接用 token）
    # ----------------------------
    def save_label(self, global_pred, frame_id):
        """保存标签到预设目录"""
        pred_label_dir = self.output_dir / "pred_labels"
        pred_label_dir.mkdir(parents=True, exist_ok=True)

        pred_label_path = pred_label_dir / f"{frame_id}.label"
        global_pred_inv = np.array([self.index_to_label.get(l, 0) for l in global_pred])
        pred_label_path.write_bytes(global_pred_inv.astype(np.uint32).tobytes())

    def save_global_points(self, cloud, pred, frame_id):
        """保存带颜色的全局点云。"""
        colors = self.get_colors(pred)
        labeled_points = np.concatenate([cloud[:, :3], colors], axis=1)
        save_all_point_cloud(labeled_points, frame_id, self.output_dir)

    def save_local_point_cloud(self, labeled_points, frame_id, cam_name):
        """保存局部点云（按 token 命名，兼容 000000 / timestamp）。"""
        if not self.flags.get('save_local_points', False):
            return

        output_dir = self.output_dir / "local_points" / cam_name / str(self.sequence)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{frame_id}.pcd"

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(labeled_points[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(labeled_points[:, 3:6] / 255.0)
        o3d.io.write_point_cloud(str(output_path), pcd)

    def save_error_points(self, cloud, pred, label, frame_id, active_classes):
        """保存错误点云。"""
        if not self.flags.get('save_error_points', False):
            return
        save_error_point_cloud(cloud[:, :3], pred, label, frame_id, active_classes, self.output_dir)

    def save_entropy(self, entropy, frame_id):
        """保存熵结果。"""
        if not self.flags.get('save_entropy', False):
            return
        save_common_entropy(entropy, frame_id, self.output_dir)

    def save_pred_prob(self, pred_prob, frame_id):
        """保存预测概率。"""
        if not self.flags.get('save_pred_prob', False):
            return
        save_custom_npy(pred_prob, frame_id, self.output_dir, subdir="pred_prob")

    def save_visualizations(self, vis_data, frame_id):
        """保存可视化相关内容（按需扩展）。"""
        if not self.flags.get('save_visualizations', False):
            return
        save_custom_npy(vis_data, frame_id, self.output_dir, subdir="visualizations")

    def save_projected_image(self, image, frame_id, cam_name):
        """保存投影图（如果需要）。"""
        if not self.flags.get('save_projected_image', False):
            return
        out_dir = self.output_dir / "projected_images" / cam_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{frame_id}.png"
        cv2.imwrite(str(out_path), image)

    def save_features(self, frame_id, cam_name, cam_ground_indices, ground_probs,
                      cam_nonground_indices, nonground_probs):
        """保存特征（按 token 命名）。"""
        if not self.flags.get('save_features', False):
            return
        if self.feature_path is None:
            return

        output_dir = self.feature_path / "features/meituan" / str(self.sequence) / cam_name
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"{frame_id}.npz"
        features_to_save = {
            'cam_ground_indices': cam_ground_indices,
            'cam_ground_probs': ground_probs,
            'cam_nonground_indices': cam_nonground_indices,
            'cam_nonground_probs': nonground_probs
        }
        np.savez_compressed(output_path, features_to_save)

    def load_features(self, frame_id, cam_name):
        """
        为指定的相机和帧加载预计算的特征。
        ✅ 兼容：frame_id 可以是 '000000' 或 时间戳字符串
        """
        if not (self.load_external_feature and self.feature_path):
            return {}

        feature_file = self.feature_path / "features/meituan" / str(self.sequence) / cam_name / f"{frame_id}.npz"
        if feature_file.exists():
            try:
                data = np.load(feature_file, allow_pickle=True)
                # 兼容你之前的保存格式：arr_0 是 dict
                if 'arr_0' in data:
                    return dict(data['arr_0'].item())
                # 或者是直接 keys
                return {k: data[k] for k in data.files}
            except Exception as e:
                print(f"警告: 无法加载特征文件 {feature_file}。错误: {e}")
                return {}
        return {}
