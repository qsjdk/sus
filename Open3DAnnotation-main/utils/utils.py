from pathlib import Path
import cv2
import numpy as np
import open3d as o3d
import torch
from scipy.stats import entropy, dirichlet
from scipy.special import digamma

def calc_entropy(data, mode, num_classes):
    """
    Calculates the entropy for a batch of distributions.
    Args:
        data (np.ndarray): Array of shape (num_points, num_classes).
        mode (str): The mode of entropy calculation ('common', 'dirichlet', 'diff_dirichlet').
        num_classes (int): The number of classes.
    Returns:
        np.ndarray: An array of entropy values of shape (num_points,).
    """
    # Ensure data is a 2D array for consistent processing
    if data.ndim == 1:
        data = data[np.newaxis, :]

    # Check for uniform distribution for each point
    is_uniform = np.all(data <= (1 / num_classes) + 1e-4, axis=1)
    
    entropies = np.zeros(data.shape[0])

    if mode == 'common':
        # Add a small epsilon to avoid log(0)
        entropies = entropy(data + 1e-10, axis=1)
            
    elif mode == 'dirichlet':
        alpha_0 = np.sum(data, axis=1)
        # Avoid division by zero or log of zero for invalid alpha_0
        valid_mask = alpha_0 > 0
        
        term1 = digamma(alpha_0[valid_mask])
        term2_numerator = np.sum(data[valid_mask] * digamma(data[valid_mask]), axis=1)
        term2 = term2_numerator / alpha_0[valid_mask]
        
        entropies[valid_mask] = term1 - term2
        # For invalid alpha_0, entropy can be considered 0 or some other placeholder
        entropies[~valid_mask] = 0

    elif mode == 'diff_dirichlet':
        alpha_0 = np.sum(data, axis=1)
        valid_mask = alpha_0 > 0
        if np.any(valid_mask):
            data_to_process = data[valid_mask]
            calculated_entropies = np.array([dirichlet.entropy(row) for row in data_to_process])
            entropies[valid_mask] = calculated_entropies
    
    else:
        raise ValueError(f"Unsupported entropy mode: {mode}")

    # Set entropy to infinity for uniform distributions
    entropies[is_uniform] = np.inf
    
    return entropies



def save_projected_image(image, proj_cloud, frame_id, output_dir, cam_name):
    """Saves an image with projected point cloud points, including camera name in the filename."""
    output_dir = Path(output_dir) / "projected_images" / cam_name 
    output_dir.mkdir(parents=True, exist_ok=True)
    projected_img = image.copy()
    for v, u, r, g, b in proj_cloud:
        if 0 <= v < image.shape[0] and 0 <= u < image.shape[1]:
            cv2.circle(projected_img, (int(u), int(v)), 2, (int(b), int(g), int(r)), -1)
    # 【修改】直接使用 frame_id 字符串
    cv2.imwrite(str(output_dir / f"projected_points_{frame_id}.png"), projected_img)

def save_pts_bin(pc_data, filename, base_dir):
    if pc_data.size == 0:
        print("Warning: Point cloud data is empty. Skipping saving.")
        return
    pcd = o3d.geometry.PointCloud()
    pts = pc_data[:, :3].astype(np.float64)
    
    # Intensity is now the 4th column, colors start from the 5th
    colors = pc_data[:, 4:7].astype(np.float64) if pc_data.shape[1] >= 7 else np.tile(np.array([0.5, 0.5, 0.5]),
                                                                                        (pts.shape[0], 1))
    intensity = pc_data[:, 3].astype(np.float64)
    if colors.max() > 1.0:
        colors /= 255.0

    # Filter out points where all color channels are zero
    mask = np.all(colors == 0, axis=1)
    pts = pts[~mask]
    colors = colors[~mask]
    intensity = intensity[~mask]

    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    filepath = base_dir / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to binary format
    points = np.asarray(pcd.points).astype(np.float32)
    colors = np.asarray(pcd.colors).astype(np.float32)
    intensity = intensity.astype(np.float32).reshape(-1, 1)
    
    # Combine points, colors and intensity
    combined_data = np.hstack((points, intensity, colors))
    
    # Save as bin file
    bin_filepath = filepath.with_suffix(".bin")
    combined_data.flatten().tofile(str(bin_filepath))

def save_pts_pcd(pc_data, filename, base_dir):
    """pc_data: (N, 3) or (N, 6), x, y , z, r, g, b"""

    pcd = o3d.geometry.PointCloud()
    pts = pc_data[:, :3].astype(np.float64)
    colors = pc_data[:, 3:6].astype(np.float64) if pc_data.shape[1] >= 6 else np.tile(np.array([0.5, 0.5, 0.5]),
                                                                                        (pts.shape[0], 1))
    if colors.max() > 1.0:
        colors /= 255.0
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    filepath = base_dir / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(filepath), pcd)

def save_point_cloud_visualization(visible_points, idx, output_dir="outputs"):
    if visible_points.shape[0] == 0:
        print(f"No visible points to save for index {idx}")
        return

    base_dir = Path(output_dir) / "visible_points"
    # base_dir = Path(output_dir)

    save_pts_pcd(visible_points, f"{idx:06}.pcd", base_dir)

def save_local_point_cloud(self, labeled_points, frame_id, cam_name):
    """保存相机视角的局部点云。"""
    if not self.flags.get('save_local_points', False):
        return

    # 路径: basedir/model_name/sequences/seq_id/cam_name/local_points/frame_id.pcd
    output_dir = self.seq_output_dir / cam_name / "local_points"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{frame_id}.pcd"

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(labeled_points[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(labeled_points[:, 3:6] / 255.0)
    o3d.io.write_point_cloud(str(output_path), pcd)


def save_all_point_cloud(cloud, frame_id, output_dir="outputs"):
    if cloud.shape[0] == 0:
        print(f"No points to save for frame {frame_id}")
        return
    base_dir = Path(output_dir) / "all_points"
    # 【修改】直接使用 frame_id 字符串
    save_pts_pcd(cloud, f"{frame_id:06d}.pcd", base_dir)

def prase_extension(extension):
    """
    Given an extension (list of categories), return:
    1. classes: a flat list of unique names/words for each category (including "name", synonyms, subcategories, and descriptions)
    2. num_classes: the number of main categories (length of the input list)
    3. classes_idx: a list of the same length as classes where each element is the index (in the input list) of the main category the word came from.
    """
    classes = []
    classes_idx = []
    original_classes = []
    for idx, cat in enumerate(extension):
        exts = []
        if "name" in cat:
            exts.append(cat["name"])
            original_classes.append(cat["name"])
        if "synonyms" in cat:
            exts.extend(cat["synonyms"])
        if "subcategories" in cat:
            exts.extend(cat["subcategories"])
        if "descriptions" in cat:
            exts.extend(cat["descriptions"])
        
        # Remove duplicates while preserving the order
        seen = set()
        unique_exts = []
        for word in exts:
            if word not in seen:
                seen.add(word)
                unique_exts.append(word)
        
        classes.extend(unique_exts)
        classes_idx.extend([idx] * len(unique_exts))
    
    num_classes = len(extension)
    return original_classes, classes, num_classes, classes_idx


def save_error_point_cloud(transformed_cloud, global_pred, global_label, frame_id, output_dir, class_names):
    # Flatten class_names in case it is nested (e.g., a list of lists)
    flattened = []
    for item in class_names:
        if isinstance(item, (list, tuple)):
            flattened.extend(item)
        else:
            flattened.append(item)
    class_names = flattened
    # Identify error points where both predictions and labels are non-zero and not equal.
    error_boolean = (global_pred != 0) & (global_label != 0) & (global_pred != global_label)

    if error_boolean.sum() == 0:
        # print(f"No error points for index {curr_index}")
        return

    # For each unique error predicted label, highlight its error points.
    unique_labels = np.unique(global_pred[error_boolean])
    for label in unique_labels:
        # Start with a full point cloud colored black.
        colors = np.tile(np.array([0.5, 0.5, 0.5]), (transformed_cloud.shape[0], 1))
        # Mark the error points corresponding to the current label in red.
        err_indices = np.where((global_pred == label) & error_boolean)[0]
        colors[err_indices] = np.array([1.0, 0.0, 0.0])
        
        if len(err_indices) < 20:
            continue
        # Combine all points (black) with the red error points.
        pc_data = np.hstack((transformed_cloud.astype(np.float64), colors))

        # Folder is based on the category name (using the integer cast of label).
        folder = Path(output_dir) / "error_points" / str(class_names[int(label - 1)])
        folder.mkdir(parents=True, exist_ok=True)

        filename = f"{frame_id:06d}_{class_names[int(label - 1)]}_{len(err_indices)}.pcd"
        save_pts_pcd(pc_data, filename, folder)

def save_common_entropy(transformed_cloud, global_entropy, frame_id, output_dir):
    if transformed_cloud.shape[0] == 0:
        print(f"No points to save for frame {frame_id}")
        return

    # base_dir = Path(output_dir) / "entropy_points"
    base_dir = Path(output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    colors = np.zeros((transformed_cloud.shape[0], 3))
    
    # Identify points with zero and non-zero entropy
    zero_entropy_mask = (global_entropy == np.inf)
    non_zero_entropy_mask = ~zero_entropy_mask

    # Assign grey color to points with zero entropy
    colors[zero_entropy_mask] = np.array([0.5, 0.5, 0.5])

    # Process points with non-zero entropy
    if np.any(non_zero_entropy_mask):
        non_inf_entropy = global_entropy[non_zero_entropy_mask]
        
        # Normalize non-zero entropy to [0, 1] for color mapping
        min_entropy = non_inf_entropy.min()
        max_entropy = non_inf_entropy.max()
        
        if max_entropy > min_entropy:
            normalized_entropy = (non_inf_entropy - min_entropy) / (max_entropy - min_entropy)
        else:
            # All non-zero entropies are the same, map to a mid-range color
            normalized_entropy = np.full_like(non_inf_entropy, 0.5)

        # Use JET colormap to convert entropy values to colors.
        # Blue (low entropy) -> Red (high entropy)
        x = normalized_entropy
        r = np.clip(1.5 - np.abs(4 * x - 3.0), 0, 1)
        g = np.clip(1.5 - np.abs(4 * x - 2.0), 0, 1)
        b = np.clip(1.5 - np.abs(4 * x - 1.0), 0, 1)
        jet_colors = np.stack([r, g, b], axis=1)
        
        colors[non_zero_entropy_mask] = jet_colors

    # Combine point cloud data with colors
    pc_data = np.hstack((transformed_cloud.astype(np.float64), colors))

    filename = f"{frame_id:06d}.pcd"
    save_pts_pcd(pc_data, filename, base_dir)


def save_custom_npy(data, filename):
    filepath = Path(filename)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    np.save(filepath, data)
    
    
# def combine_outputs(ground_output, nonground_output, name_to_final_id):
#     """
#     [修改后] 将来自地面和非地面VoxelMap的输出合并。
#     这个函数现在直接合并已经在全局类别空间中的概率。
#     """
#     if ground_output is None and nonground_output is None:
#         return None

#     num_global_classes = len(name_to_final_id)
    
#     all_clouds, all_labels, all_point_indices = [], [], []
#     all_probs = []
    
#     frame_info = {} # 用于存储pose, index, token等信息

#     # 处理 ground output
#     if ground_output is not None:
#         g_cloud, _, g_label, g_pose, g_index, g_pred_prob, g_point_idx, g_lidarseg_token = ground_output
        
#         all_clouds.append(g_cloud)
#         all_labels.append(g_label)
#         all_point_indices.append(g_point_idx)
#         all_probs.append(g_pred_prob) # 概率已经是全局的
        
#         # 保存帧信息 (假设两个输出的帧信息是一致的)
#         frame_info['pose'] = g_pose
#         frame_info['frame_id'] = g_index
#         frame_info['label_token'] = g_lidarseg_token

#     # 处理 non-ground output
#     if nonground_output is not None:
#         ng_cloud, _, ng_label, ng_pose, ng_index, ng_pred_prob, ng_point_idx, ng_lidarseg_token = nonground_output
        
#         all_clouds.append(ng_cloud)
#         all_labels.append(ng_label)
#         all_point_indices.append(ng_point_idx)
#         all_probs.append(ng_pred_prob) # 概率已经是全局的

#         if not frame_info: # 如果ground_output为None，从nonground_output获取帧信息
#             frame_info['pose'] = ng_pose
#             frame_info['frame_id'] = ng_index
#             frame_info['label_token'] = ng_lidarseg_token

#     # 合并所有数据
#     merged_cloud = np.concatenate(all_clouds, axis=0)
#     merged_label = np.concatenate(all_labels, axis=0)
#     merged_point_indices = np.concatenate(all_point_indices, axis=0)
#     merged_pred_prob = np.concatenate(all_probs, axis=0)

#     # 根据合并后的全局概率计算最终预测
#     merged_pred = np.argmax(merged_pred_prob, axis=1) + 1
#     # 如果所有类别的概率都非常低，则认为是 "unlabeled" (0)
#     merged_pred[np.max(merged_pred_prob, axis=1) < (1.0 / num_global_classes)] = 0

#     # 按照原始点云索引排序，以确保数据对齐
#     sort_indices = np.argsort(merged_point_indices)
    
#     output = {
#         'cloud': merged_cloud[sort_indices],
#         'pred': merged_pred[sort_indices],
#         'label': merged_label[sort_indices],
#         'pose': frame_info.get('pose'),
#         'frame_id': frame_info.get('frame_id'),
#         'pred_prob': merged_pred_prob[sort_indices],
#         'label_token': frame_info.get('label_token')
#     }
    
    # return output
    

def combine_outputs(ground_output, nonground_output, num_classes):
    if ground_output is None and nonground_output is None:
        return None
    
    if ground_output is None:
        return nonground_output
    
    if nonground_output is None:
        return ground_output

    g_cloud, g_pred, g_label, g_pose, g_index, g_pred_prob, g_point_idx, g_lidarseg_token = ground_output
    ng_cloud, ng_pred, ng_label, _, _, ng_pred_prob, ng_point_idx, _ = nonground_output

    total_points = len(g_point_idx) + len(ng_point_idx)
    
    merged_cloud = np.zeros((total_points, g_cloud.shape[1]), dtype=g_cloud.dtype)
    merged_pred = np.zeros(total_points, dtype=g_pred.dtype)
    merged_label = np.zeros(total_points, dtype=g_label.dtype)
    
    merged_pred_prob = np.zeros((total_points, num_classes), dtype=g_pred_prob.dtype)

    merged_cloud[g_point_idx] = g_cloud
    merged_cloud[ng_point_idx] = ng_cloud

    merged_pred[g_point_idx] = g_pred
    merged_pred[ng_point_idx] = ng_pred

    merged_pred_prob[g_point_idx] = g_pred_prob
    merged_pred_prob[ng_point_idx] = ng_pred_prob

    merged_label[g_point_idx] = g_label
    merged_label[ng_point_idx] = ng_label
    
    merged_pose = g_pose
    curr_index = g_index

    output = {
        'cloud': merged_cloud,
        'pred': merged_pred,
        'label': merged_label,
        'pose': merged_pose,
        'frame_id': curr_index,
        'pred_prob': merged_pred_prob,
        'label_token': g_lidarseg_token
    }
    
    return output
