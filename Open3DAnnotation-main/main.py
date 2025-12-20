import os
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['HF_HUB_DISABLE_SSL'] = 'True'

import torch
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.metric import IOUCalculator
from dataset.semantic_kitti import SemanticKITTIDataset
from dataset.nuscenes import NuScenesDataset
from dataset.meituan import MeiTuanDataset
from utils.model_process import ModelProcess
from utils.voxel_map import VoxelMap
from utils.utils import combine_outputs
import pypatchworkpp
import gc
import argparse

def load_config(config_path):
    """加载主配置文件。"""
    with open(config_path) as f:
        return yaml.safe_load(f)

def get_dataset_class(dataset_type):
    """根据类型获取数据集类。"""
    "新增美团数据集类"
    if dataset_type == 'semantic_kitti':
        return SemanticKITTIDataset
    elif dataset_type == 'nuscenes':
        return NuScenesDataset
    elif dataset_type == 'meituan':
        return MeiTuanDataset
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

def main(config_path):
    # 1. 加载主配置和保存设置
    config = load_config(config_path)
    save_settings = config.get('save', {})
    save_settings['flags'] = {k: v for k, v in save_settings.items() if k.startswith('save_')}
    save_settings['output_dir'] = Path(save_settings.get('output_path', 'outputs'))
    
    model_cfg = config.get('model_config', {})
    save_settings['detection_model'] = model_cfg.get('DetectionModel', {}).get('model_name', 'unknown_det')
    save_settings['segment_model'] = model_cfg.get('SegmentationModel', {}).get('model_name', 'unknown_seg')

    # 2. 获取数据集类和序列列表
    dataset_cfg = config.get('dataset', {})
    dataset_type = dataset_cfg.get('type', 'semantic_kitti').lower()
    DatasetClass = get_dataset_class(dataset_type)
    sequences = DatasetClass.get_sequences(dataset_cfg)

    # [新设计] 获取评估配置
    eval_cfg = config.get('evaluation', {})
    reporting_mode = eval_cfg.get('reporting_mode', 'per_sequence') # 默认为 per_sequence

    global_output_dir = save_settings['output_dir'] / f"{save_settings['detection_model']}_{save_settings['segment_model']}"
    
    temp_dataset = DatasetClass(config=config, save_settings=save_settings, sequence=sequences[0], global_output_dir=global_output_dir)
    label_ids = sorted(temp_dataset.report_name.keys())
    report_names = [temp_dataset.report_name[i] for i in label_ids]
    
    if reporting_mode == 'global' or reporting_mode == 'all':
        # print("Metrics reporting mode: global")
        global_iou_calculator = IOUCalculator(class_names=report_names)
    else:
        # print("Metrics reporting mode: per_sequence")
        global_iou_calculator = None # 在全局模式下不使用

    # 3. 初始化地面分割
    use_patchwork = config.get('processing', {}).get('use_patchwork', True)

    if use_patchwork:
        params = pypatchworkpp.Parameters()
        params.verbose = False
        PatchworkPLUSPLUS = pypatchworkpp.patchworkpp(params)
    else:
        PatchworkPLUSPLUS = None

    
    load_feature = config.get('external_feature', {}).get('load_feature', False)
    model = ModelProcess(config=config)
    
    # 4. 遍历每个序列进行处理
    for seq in tqdm(sequences, desc="Processing Sequences", unit="sequence"):
        # 初始化数据集和数据加载器
        dataset = DatasetClass(config=config, save_settings=save_settings, sequence=seq, global_output_dir=global_output_dir)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: x[0])
        model.update_dataset(dataset)

        # 初始化时间记录文件
        dataset.init_time_log()
        
        if reporting_mode == 'per_sequence' or reporting_mode == 'all':
            # print("Metrics reporting mode: per_sequence")
            seq_iou_calculator = IOUCalculator(class_names=report_names)
        else:
            seq_iou_calculator = None

        # 初始化VoxelMap
        voxel_cfg = config.get('processing', {}).get('voxelmap', {})
        num_global_classes = len(dataset.report_name)
        voxel_map_ground = VoxelMap(
            relation=dataset.relation,
            voxel_cfg=voxel_cfg,
            num_classes=num_global_classes,
            device=model.device
        )
        voxel_map_nonground = VoxelMap(
            relation=dataset.relation,
            voxel_cfg=voxel_cfg,
            num_classes=num_global_classes,
            device=model.device
        )
        
        for idx, data in enumerate(tqdm(dataloader, desc=f"Processing {seq}", unit="frame")):
            # ✅ 放在你取出 cloud/label 之前，越早越好
            if isinstance(data, dict) and data.get("skip", False):
                print(f"[SKIP] seq={seq} frame={data.get('label_token')} "
                      f"points={data.get('num_points')} reason={data.get('reason')}")
                continue

            images = {k: v for k, v in data['images'].items()}
            cloud = data['cloud']
            label = data['label']
            pose = data['pose']
            model_predict = data['model_predict']
            transformed_cloud = data['transformed_cloud']
                
            # 获取两种类型的帧ID
            frame_idx = data['index']
            frame_token = data['label_token']



            # 地面分割
            if use_patchwork and PatchworkPLUSPLUS is not None:
                PatchworkPLUSPLUS.estimateGround(cloud)
                ground_indices = PatchworkPLUSPLUS.getGroundIndices()
                nonground_indices = PatchworkPLUSPLUS.getNongroundIndices()
            else:
                # 不分割，全部作为非地面
                ground_indices = np.array([], dtype=int)
                nonground_indices = np.arange(cloud.shape[0])


            # 更新VoxelMap (使用 token 作为唯一标识符)
            voxel_map_ground.step_frame_window(
                points=transformed_cloud[ground_indices], 
                point_idxs=ground_indices, 
                global_label=label[ground_indices], 
                pose=pose, 
                index=frame_idx, 
                mode='ground', 
                lidarseg_token=frame_token
            )

            print("shape(label)=", label.shape, "shape(ground_indices)=", ground_indices.shape)

            voxel_map_nonground.step_frame_window(
                points=transformed_cloud[nonground_indices], 
                point_idxs=nonground_indices, 
                global_label=label[nonground_indices], 
                pose=pose, 
                index=frame_idx, 
                mode='nonground', 
                lidarseg_token=frame_token
            )
            
            frame_total_detection_time = 0
            frame_total_segment_time = 0
            for cam_name, image in images.items():
                proj_info = data['proj_cloud'].get(cam_name)
                if not proj_info or proj_info['indices'].size == 0:
                    continue
                
                cam_results = {}
                if load_feature and len(model_predict) > 0 and len(model_predict[cam_name]) > 0:
                    cam_results = model_predict[cam_name]
                    cam_ground_indices = cam_results.get('cam_ground_indices', None)
                    cam_ground_probs = cam_results.get('cam_ground_probs', None)
                    cam_nonground_indices = cam_results.get('cam_nonground_indices', None)
                    cam_nonground_probs = cam_results.get('cam_nonground_probs', None)
                    detection_time = 0
                    segment_time = 0
                    
                if len(cam_results) == 0:
                    (cam_ground_indices, cam_ground_probs,
                     cam_nonground_indices, cam_nonground_probs,
                     detection_time, segment_time) = model.predict_for_camera(
                        image=image, 
                        proj_info=proj_info,
                        ground_indices=ground_indices,
                        nonground_indices=nonground_indices,
                        frame_id=frame_idx,
                        cam_name=cam_name
                    )
                    cam_results = {
                        'cam_ground_indices': cam_ground_indices,
                        'cam_ground_probs': cam_ground_probs,
                        'cam_nonground_indices': cam_nonground_indices,
                        'cam_nonground_probs': cam_nonground_probs,
                        'detection_time': detection_time,
                        'segment_time': segment_time
                    }
                    
                frame_total_detection_time += detection_time
                frame_total_segment_time += segment_time
                
                # 更新VoxelMap概率
                if cam_ground_indices is not None and cam_ground_indices.size > 0:
                    voxel_map_ground.update_probabilities(
                        pose=pose,
                        point_coords=transformed_cloud[cam_ground_indices],
                        point_probs=cam_ground_probs
                    )

                if cam_nonground_indices is not None and cam_nonground_indices.size > 0:
                    voxel_map_nonground.update_probabilities(
                        pose=pose,
                        point_coords=transformed_cloud[cam_nonground_indices],
                        point_probs=cam_nonground_probs
                    )
                    
                all_cam_indices_parts = []
                all_cam_probs_parts = []
                if cam_ground_indices is not None and cam_ground_indices.size > 0 and cam_ground_probs is not None and cam_ground_probs.shape[0] > 0:
                    all_cam_indices_parts.append(cam_ground_indices)
                    all_cam_probs_parts.append(cam_ground_probs)
                if cam_nonground_indices is not None and cam_nonground_indices.size > 0 and cam_nonground_probs is not None and cam_nonground_probs.shape[0] > 0:
                    all_cam_indices_parts.append(cam_nonground_indices)
                    all_cam_probs_parts.append(cam_nonground_probs)
                if len(all_cam_indices_parts) == 0 or len(all_cam_probs_parts) == 0:
                    continue
                all_cam_indices = np.concatenate(all_cam_indices_parts)
                if all_cam_indices.size > 0:
                    all_cam_probs = np.concatenate(all_cam_probs_parts, axis=0)

                    pred_local = np.argmax(all_cam_probs, axis=1) + 1
                    pred_local[np.max(all_cam_probs, axis=1) <= (1 / len(dataset.active_classes))] = 0

                    points_3d = transformed_cloud[all_cam_indices, :3]
                    
                    index_to_pixel_map = {idx.item(): px for idx, px in zip(proj_info['indices'], proj_info['pixels'])}
                
                    pixels_2d_list = [index_to_pixel_map.get(idx) for idx in all_cam_indices]
                    
                    valid_mask = [p is not None for p in pixels_2d_list]
                    points_3d = points_3d[valid_mask]
                    pixels_2d = np.array([p for p in pixels_2d_list if p is not None])
                    pred_local_filtered = pred_local[valid_mask]



                    if pred_local_filtered.size == 0:
                        continue

                    # 2. [修改] 调用新的 colorize_points 方法进行解耦的着色
                    labeled_points = dataset.colorize_points(points_3d, pred_local_filtered)
                    proj_points = dataset.colorize_points(pixels_2d, pred_local_filtered)
                    
                    
                    # 调用dataset中的方法进行保存 (使用数字索引)
                    dataset.save_projected_image(image, proj_points, frame_idx, cam_name)
                    dataset.save_local_point_cloud(labeled_points, frame_idx, cam_name)
                
                

            # 弹出并保存已完成的帧
            ground_output = voxel_map_ground.pop_frame()
            nonground_output = voxel_map_nonground.pop_frame()

            # 记录时间 (使用数字索引)
            dataset.log_time(frame_idx, frame_total_detection_time, frame_total_segment_time)
            
            if ground_output is not None or nonground_output is not None:
                final_output = combine_outputs(ground_output, nonground_output, len(dataset.report_name))
                if final_output:
                    final_pred = final_output['pred']
                    final_label = final_output['label']
                    if reporting_mode == 'all':
                        seq_iou_calculator.update(final_pred, final_label)
                        global_iou_calculator.update(final_pred, final_label)
                    elif reporting_mode == 'per_sequence':
                        seq_iou_calculator.update(final_pred, final_label)
                    elif reporting_mode == 'global':
                        global_iou_calculator.update(final_pred, final_label)
                    dataset.save(final_output)

        # 刷新VoxelMap并处理所有剩余帧
        # flush() 返回一个结果列表，每个元素都是 _pop_frame 返回的元组
        flushed_ground = voxel_map_ground.flush()
        flushed_nonground = voxel_map_nonground.flush()

        for ground_output, nonground_output in zip(flushed_ground, flushed_nonground):
            final_output = combine_outputs(ground_output, nonground_output, len(dataset.report_name))
            if final_output:
                final_pred = final_output['pred']
                final_label = final_output['label']
                if reporting_mode == 'all':
                    seq_iou_calculator.update(final_pred, final_label)
                    global_iou_calculator.update(final_pred, final_label)
                elif reporting_mode == 'per_sequence':
                    seq_iou_calculator.update(final_pred, final_label)
                elif reporting_mode == 'global':
                    global_iou_calculator.update(final_pred, final_label)
                dataset.save(final_output)

        if reporting_mode == 'per_sequence' or reporting_mode == 'all':
            report_path = dataset.output_dir / "metrics_reports" / seq / "metrics_report.csv"
            # print(f"Generating report for sequence {seq}...")
            seq_iou_calculator.report(report_path)
            # print(f"Metrics report saved to {report_path}")

        del dataset, seq_iou_calculator, dataloader, voxel_map_ground, voxel_map_nonground
        torch.cuda.empty_cache()
        gc.collect()

    if reporting_mode == 'global' or reporting_mode == 'all':
        report_path = global_output_dir / "metrics_reports" / "metrics_report_global.csv"
        print("Generating global metrics report...")
        global_iou_calculator.report(report_path)
        print(f"Global metrics report saved to {report_path}")

if __name__ == "__main__":
    # torch.multiprocessing.set_start_method('spawn', force=True)
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(description="Run MyOpenProject pipeline")
    default_config = Path(__file__).resolve().parent / "config" / "DINO_SAM_meituan.yaml" # 在这里修改数据集路径
    parser.add_argument("-c", "--config", default=str(default_config),
                        help=f"Path to the YAML config file (default: {default_config})")
    args = parser.parse_args()
    config_path = args.config if args.config is not None else str(default_config)
    main(config_path)