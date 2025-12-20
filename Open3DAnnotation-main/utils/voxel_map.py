import torch
import numpy as np
from collections import deque, defaultdict
import nearest_neighbors as nearest_neighbors
import numpy as np

class ProbFusion:
    def __init__(self, num_classes, mode='mean', init_alpha=0.01):
        self.num_classes = num_classes
        self.mode = mode
        self.init_alpha = init_alpha
    
    def fuse_prob(self, data1, data2):
        """
        Fuses two sets of probability distributions.
        Args:
        data1 (np.ndarray): Probabilities of shape (num_points, num_classes).
        data2 (np.ndarray): Probabilities of shape (num_points, num_classes).
        Returns:
        np.ndarray: Fused probabilities of shape (num_points, num_classes).
        """
        if self.mode == 'mean':
            return (data1 + data2) / 2
        if self.mode == 'bayesian':
            return self._bayesian_fusion(data1, data2)
        if self.mode == 'dirichlet':
            # For DST (Dempster-Shafer Theory) based fusion using Dirichlet parameters
            alpha1 = torch.from_numpy(data1)
            alpha2 = torch.from_numpy(data2)
            return self._dst_fusion(alpha1, alpha2).numpy()
        raise ValueError(f"Unsupported fusion method: {self.mode}")
    


    @staticmethod

    def _bayesian_fusion(data1, data2):
        """
        Performs Bayesian fusion directly using NumPy for efficiency.
        Assumes inputs are probabilities.
        """
        fused_prob = np.softmax(data1, axis=1) * np.softmax(data2, axis=1)
        # Normalize to ensure the probabilities sum to 1.
        return fused_prob / (fused_prob.sum(axis=1, keepdims=True) + 1e-10)
    
    

    def _dst_fusion(self, alpha1, alpha2):
        """
        :param alpha1: Dirichlet distribution parameters of view 1
        :param alpha2: Dirichlet distribution parameters of view 2
        :return: Combined Dirichlet distribution parameters
        """
        alpha = dict()
        alpha[0], alpha[1] = alpha1, alpha2
        b, S, E, u = dict(), dict(), dict(), dict()
        for v in range(2):
            S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
            E[v] = alpha[v]-self.init_alpha
            b[v] = E[v]/(S[v].expand(E[v].shape))
            u[v] = (self.num_classes * self.init_alpha) / S[v]

        # b^0 @ b^(0+1)
        bb = torch.bmm(b[0].view(-1, self.num_classes, 1), b[1].view(-1, 1, self.num_classes))
        # b^0 * u^1
        uv1_expand = u[1].expand(b[0].shape)
        bu = torch.mul(b[0], uv1_expand)
        # b^1 * u^0
        uv_expand = u[0].expand(b[0].shape)
        ub = torch.mul(b[1], uv_expand)
        # calculate C
        bb_sum = torch.sum(bb, dim=(1, 2), out=None)
        bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
        C = bb_sum - bb_diag

        # calculate b^a
        b_a = (torch.mul(b[0], b[1]) + bu + ub)/((1-C).view(-1, 1).expand(b[0].shape))
        # calculate u^a
        u_a = torch.mul(u[0], u[1])/((1-C).view(-1, 1).expand(u[0].shape))

        # calculate new S
        S_a = self.num_classes / u_a
        # calculate new e_k
        e_a = torch.mul(b_a, S_a.expand(b_a.shape))
        alpha_a = e_a + self.init_alpha
        return alpha_a
        
        
class Voxel:
    def __init__(self, use_key_frame, dist_thres, angle_thres, num_classes, mode, init_alpha, update_threshold, use_net=False, device='cpu'):
        self.use_key_frame = use_key_frame
        self.dist_thres = dist_thres
        self.angle_thres = angle_thres
        self.num_classes = num_classes
        self.device = device
        self.prob = np.ones(num_classes) / num_classes
        self.current_points = 0
        self.update_num = 0
        self.threshold = 1 / num_classes
        self.status = {}
        self.update_threshold = update_threshold
        self.uncertainty_vector = np.zeros(num_classes, dtype=np.float32)
        self.mode = mode
        self.pose = np.zeros((4,4), dtype=np.float32)
        if self.mode == 'dirichlet':
            self.net_prob = np.ones(num_classes, dtype=np.float32) * init_alpha
        else:
            self.net_prob = np.ones(num_classes) / num_classes
        
        # 将更新方法映射到模式
        self._update_methods = {
            'bayesian': self._bayesian_update,
            'maximal': self._maximal_update,
            'dirichlet': self._dirichlet_update,
        }
        
        self.use_net = use_net

    def _check_key_frame(self, current_pose):
        
        if np.all(self.pose == 0) or not self.use_key_frame:
            self.pose = current_pose
            return True
        
        translation_diff = np.linalg.norm(current_pose[:3, 3] - self.pose[:3, 3])
        # 修正：clip到[-1, 1]
        cos_theta = (np.trace(current_pose[:3, :3].T @ self.pose[:3, :3]) - 1) / 2
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        rotation_diff = np.arccos(cos_theta)
        if translation_diff < self.dist_thres and rotation_diff < self.angle_thres:
            return False
        self.pose = current_pose
        return True
    
    def update(self, data, pose):
        """
        Update the voxel's probability using the specified mode. After updating,
        log the updated probability and effective update count if a logger is provided.
        """
        mask = data > self.threshold
        if not np.any(mask):
            return
        
        update_method = self._update_methods.get(self.mode)
        is_key_frame = self._check_key_frame(pose)
        if not is_key_frame:
            return
        
        if update_method:
            update_method(data)
            self.update_num += 1
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")
    
    
    def update_net_prob(self, data):
        if self.mode == 'dirichlet':
            self.net_prob += data
        else:
            likelihood = data
            prior = self.net_prob
            evidence = prior * likelihood + (1 - prior) * (1 - likelihood) + 1e-10
            self.net_prob = (prior * likelihood) / evidence
            self.net_prob /= self.net_prob.sum()  # Normalize.

    def _dirichlet_update(self, data):
        mask = data > self.threshold
        self.uncertainty_vector[mask] += data[mask]

    def _bayesian_update(self, data):
        """Bayesian update: update probabilities."""
        
        data_copy = np.ones_like(data) * self.threshold
        mask = data > self.threshold
        data_copy[mask] = data[mask]
        data_copy /= data_copy.sum()  # Normalize.
        
        likelihood = data_copy
        prior = self.prob
        evidence = prior * likelihood + (1 - prior) * (1 - likelihood) + 1e-10
        self.prob = (prior * likelihood) / evidence
        self.prob /= self.prob.sum()  # Normalize.
    
    def _maximal_update(self, data):
        """Maximal update: update probabilities with the maximum value between current and new data."""
        data = np.asarray(data)
        self.prob = np.maximum(self.prob, data)
    
    def get_prob(self):
        """Return the current probability distribution."""
        if self.mode == 'dirichlet':
            if self.update_num < self.update_threshold:
                # Return a uniform distribution if not updated enough
                return np.zeros(self.num_classes)
            
            # Return normalized probabilities from Dirichlet parameters
            dirichlet_params = self.uncertainty_vector.copy()
            return dirichlet_params
        
        if self.update_num < self.update_threshold:
            return np.zeros(self.num_classes)
        
        return self.prob
    
class VoxelMap:
    def __init__(self, relation, voxel_cfg, num_classes, device):
        self.use_key_frame = voxel_cfg.get('use_key_frame', False)
        self.dist_thres = voxel_cfg.get('key_frame_dist_thres', 0.3)
        self.angle_thres = voxel_cfg.get('key_frame_angle_thres', 0.3)
        self.voxel_size = voxel_cfg.get('voxel_size', 0.1)
        self.window_size = voxel_cfg.get('max_frames', 5)
        self.mode = voxel_cfg.get('mode', 'frequency')
        self.init_alpha = voxel_cfg.get('init_alpha', 0.01)
        self.use_net = voxel_cfg.get('use_net', False)
        self.update_threshold = voxel_cfg.get('update_threshold', 3)
        self.relation = relation
        self.use_relation = voxel_cfg.get('use_relation', False)
        self.relation_threshold = voxel_cfg.get('relation_threshold', 0.3)


        self.num_classes = num_classes
        self.device = device

        self.voxels = defaultdict(lambda: Voxel(self.use_key_frame, self.dist_thres, self.angle_thres, self.num_classes, self.mode, self.init_alpha, self.update_threshold, self.use_net, self.device))
        self.point_window = deque(maxlen=self.window_size)
        self.label_window = deque(maxlen=self.window_size)
        self.pose_window = deque(maxlen=self.window_size)
        self.point_index_window = deque(maxlen=self.window_size)
        self.lidarseg_token_window = deque(maxlen=self.window_size)
        self.window_index = []

        self.inverse_voxel_size = 1.0 / self.voxel_size
        self.prob_fusion = ProbFusion(self.num_classes, self.mode, self.init_alpha)

    def _aggregate_by_voxel(self, points, data):
        """Helper to aggregate point-wise data by voxel."""
        if points.shape[0] == 0:
            return np.empty((0, 3), dtype=np.int64), np.empty((0, data.shape[1]), dtype=np.float32)
        
        voxel_coords = self._to_voxel_coords(points[:, :3])
        unique_coords, inv_idx, counts = np.unique(voxel_coords, axis=0, return_inverse=True, return_counts=True)
        
        sums = np.zeros((unique_coords.shape[0], data.shape[1]), dtype=np.float32)
        np.add.at(sums, inv_idx, data)
        
        return unique_coords, sums / counts[:, None]
    
    def combine_probs(self, smoothed_probs, net_pred_probs):
        return self.prob_fusion.fuse_prob(smoothed_probs, net_pred_probs)

    def _calculate_probabilities(self, unique_coords, inv):
        """Helper method to calculate point-wise probabilities."""
        unique_smoothed_probs = np.array([self.voxels[tuple(coord)].get_prob() for coord in unique_coords], dtype=np.float32)
        smoothed_probs = unique_smoothed_probs[inv]

        if self.use_net:
            net_pred_probs = np.array([self.voxels[tuple(coord)].net_prob for coord in unique_coords], dtype=np.float32)
            
            if self.mode == 'dirichlet':
                # For dirichlet, net_prob stores parameters, needs normalization
                net_pred_probs = net_pred_probs / (net_pred_probs.sum(axis=1, keepdims=True) + 1e-10)
            
            smoothed_net_probs = net_pred_probs[inv]
            # Combine smoothed probabilities with network predictions
            final_probs = self.combine_probs(smoothed_probs, smoothed_net_probs)
        else:
            final_probs = smoothed_probs
            
        return final_probs

    def step_frame_window(self, points, point_idxs, global_label, pose, index, mode, lidarseg_token=None, net_output_subset=None):
        """
        将一个时间帧的子集（地面或非地面）推入窗口。
        这个方法现在接收已经分割好的点云子集和对应的网络预测。
        """
        # 使用 net_output_subset 更新体素的网络预测概率
        if self.use_net and net_output_subset is not None and net_output_subset.size > 0:
            # 确保 net_output_subset 的点数与输入的 points 子集匹配
            if len(net_output_subset) == len(points):
                voxel_coords = self._to_voxel_coords(points[:, :3])
                unique_coords, inv_idx = np.unique(voxel_coords, axis=0, return_inverse=True)
                counts = np.bincount(inv_idx)
                
                sums = np.zeros((unique_coords.shape[0], self.num_classes), dtype=np.float32)
                np.add.at(sums, inv_idx, net_output_subset)
                avg_probs = sums / counts[:, None]

                for coord, avg_prob in zip(unique_coords, avg_probs):
                    self.voxels[tuple(coord)].update_net_prob(avg_prob)
        
        # 1. 将新帧数据添加到窗口
        self.point_window.append(points.copy())
        self.label_window.append(global_label)
        self.window_index.append(index)
        self.pose_window.append(pose)
        self.point_index_window.append(point_idxs)
        self.lidarseg_token_window.append(lidarseg_token)

        # 2. 更新所有点的体素占用计数
        voxel_coords = self._to_voxel_coords(points[:, :3])
        unique_coords, counts = np.unique(voxel_coords, axis=0, return_counts=True)
        for coord, count in zip(unique_coords, counts):
            self.voxels[tuple(coord)].current_points += count

 
    def update_probabilities(self, pose, point_coords, point_probs, net_pred_coords=None, net_pred_probs=None):
        """
        仅使用来自一个视角的数据更新体素的概率，不操作时间窗口。
        此方法现在接收点坐标和概率的NumPy数组。
        """
        # 更新网络预测概率
        if self.use_net and net_pred_probs is not None and net_pred_probs.size > 0:
            unique_coords, avg_net_probs = self._aggregate_by_voxel(net_pred_coords, net_pred_probs)
            for coord, avg_prob in zip(unique_coords, avg_net_probs):
                self.voxels[tuple(coord)].update_net_prob(avg_prob, pose)

        # 更新主要概率
        if point_probs is not None and point_probs.size > 0:
            unique_coords, avg_probs = self._aggregate_by_voxel(point_coords, point_probs)
            for coord, avg_prob in zip(unique_coords, avg_probs):
                self.voxels[tuple(coord)].update(avg_prob, pose)
               
    # def get_labels(self, probs):
    #     """
    #     根据概率数组返回预测类别。
    #     如果 use_relation=True，且 relation 存在，则当子类别概率大于等于主类别概率 * relation_threshold 时，优先选择子类别。
    #     """
    #     if self.use_relation and self.relation:
    #         pred_labels = probs.argmax(axis=1) + 1
    #         pred_labels[np.all(probs <= 1 / self.num_classes, axis=1)] = 0
            
    #         for i, prob in enumerate(probs):
    #             label = pred_labels[i]
    #             related_classes = self.relation.get(label, [])
    #             if label == 0 or not related_classes:
    #                 continue
    #             main_prob = prob[label - 1]
    #             # 记录最大概率的子类别
    #             best_sub_label = label
    #             best_sub_prob = main_prob
    #             for sub_label in related_classes:
    #                 sub_prob = prob[sub_label - 1]
    #                 if sub_prob >= main_prob * self.relation_threshold and sub_prob > best_sub_prob:
    #                     best_sub_label = sub_label
    #                     best_sub_prob = sub_prob
    #             pred_labels[i] = best_sub_label
    #         return pred_labels
    #     else:
    #         threshold = 1.0 / self.num_classes
    #         pred_labels = np.where(np.all(probs <= threshold, axis=1),
    #                                     0,
    #                                     probs.argmax(axis=1) + 1)
    #         return pred_labels
    
    def get_labels(self, probs):
        threshold = 1.0 / self.num_classes
        pred_labels = np.where(np.all(probs <= threshold, axis=1),
                                    0,
                                    probs.argmax(axis=1) + 1)
        return pred_labels
    
    def pop_frame(self):
        if len(self.point_window) >= self.window_size:
            return self._pop_frame()
        return None
    
    def _pop_frame(self):
        """优化过期帧处理"""
        if not all([self.point_window, self.point_index_window, self.label_window, 
                    self.pose_window, self.window_index, self.lidarseg_token_window]):
            return None
        
        oldest_points = self.point_window.popleft()
        oldest_point_idx = self.point_index_window.popleft()
        global_label = self.label_window.popleft()
        global_pose = self.pose_window.popleft()
        window_index = self.window_index.pop(0)
        lidarseg_token = self.lidarseg_token_window.popleft()

        oldest_voxel_coords = self._to_voxel_coords(oldest_points[:, :3])
        unique_coords, inv, counts = np.unique(oldest_voxel_coords, axis=0, return_inverse=True, return_counts=True)
        
        # Calculate final probabilities using the helper method
        final_probs = self._calculate_probabilities(unique_coords, inv)
    
        # Determine prediction labels based on final probabilities
        # threshold = 1.0 / self.num_classes
        # pred_labels = np.where(np.all(final_probs <= threshold, axis=1),
        #                             0,
        #                             final_probs.argmax(axis=1) + 1)
        pred_labels = self.get_labels(final_probs)
        if self.mode == 'dirichlet':
            final_probs = final_probs + np.ones(self.num_classes, dtype=np.float32) * self.init_alpha
            
        # Decrement point counts and identify voxels to delete
        keys_to_delete = []
        for i, coord_tuple in enumerate(map(tuple, unique_coords)):
            voxel = self.voxels.get(coord_tuple)
            if voxel:
                voxel.current_points -= counts[i]
                if voxel.current_points <= 0:
                    keys_to_delete.append(coord_tuple)

        for key in keys_to_delete:
            del self.voxels[key]

        return oldest_points, pred_labels, global_label, global_pose, window_index, final_probs, oldest_point_idx, lidarseg_token
    
    def _to_voxel_coords(self, points):
        """优化坐标转换计算"""
        if isinstance(points, np.ndarray):
            return (points * self.inverse_voxel_size).astype(np.int64)
        return (np.array(points) * self.inverse_voxel_size).astype(np.int64)
    
    def flush(self):
        """清空剩余帧"""
        results = []
        while self.point_window:
            results.append(self._pop_frame())
        return [r for r in results if r is not None]