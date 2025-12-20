import os
os.environ['CURL_CA_BUNDLE'] = ''
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import threading
import cv2
import numpy as np
import supervision as sv
from mmengine.config import Config
import torch
import torchvision
from mmdet.apis import init_detector
from mmengine.dataset import Compose
from mmdet.utils import get_test_pipeline_cfg
from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor
import yaml

import time
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from sklearn.metrics import confusion_matrix
from sklearn.cluster import DBSCAN, OPTICS
# from sam2.build_sam import build_sam2
# from sam2.sam2_image_predictor import SAM2ImagePredictor
from utils.utils import prase_extension
from collections import defaultdict
import pypatchworkpp
from typing import Callable, Dict
import open3d as o3d
from pathlib import Path
class LabelAnnotator(sv.LabelAnnotator):

    @staticmethod
    def resolve_text_background_xyxy(
        center_coordinates,
        text_wh,
        position,
    ):
        center_x, center_y = center_coordinates
        text_w, text_h = text_wh
        return center_x, center_y, center_x + text_w, center_y + text_h


LABEL_ANNOTATOR = LabelAnnotator(text_padding=4,
                                 text_scale=0.5,
                                 text_thickness=1)
class ModelProcess:
    def __init__(self, config):
        self.config = config
        self.device = self.config['model_config'].get('device', "cuda" if torch.cuda.is_available() else "cpu")

        if 'model_config' not in self.config:
            raise ValueError("错误: 在配置中未找到 'model_config'。")
        
        model_cfg = self.config['model_config']
        detection_model_cfg = model_cfg.get('DetectionModel', {})
        segmentation_model_cfg = model_cfg.get('SegmentationModel', {})

        # Detection Model Configuration
        self.detection_model_name = detection_model_cfg.get('model_name')
        self.detection_model_config_path = detection_model_cfg.get('model_config_path')
        self.detection_model_checkpoint_path = detection_model_cfg.get('model_checkpoint_path')
        self.box_threshold = detection_model_cfg.get('box_threshold', 0.25)
        self.text_threshold = detection_model_cfg.get('text_threshold', 0.25)
        self.nms_threshold = detection_model_cfg.get('nms_threshold', 0.5)
        self.use_cluster = detection_model_cfg.get('use_cluster', False)
        self.eps = detection_model_cfg.get('eps', 1.2)
        self.min_samples = detection_model_cfg.get('min_samples', 5)
        self.nms_iou_threshold = detection_model_cfg.get('nms_iou_threshold', 0.5)
        self.max_detections = detection_model_cfg.get('max_detections', 100)
        
        # Segmentation Model Configuration
        self.segment_model_name = segmentation_model_cfg.get('model_name')
        self.segment_encoder_version = segmentation_model_cfg.get('encoder_version')
        self.segment_model_checkpoint_path = segmentation_model_cfg.get('model_checkpoint_path')

        self.save_features = self.config.get('external_feature', {}).get('save_feature', False)
        
        self.dataset = None  # Will be set later
        
        # Initialize Detection Model
        if self.detection_model_name == "GroundingDINO":
            self.detection_model = Model(model_config_path=self.detection_model_config_path, model_checkpoint_path=self.detection_model_checkpoint_path)
        elif self.detection_model_name == "YoloWorld":
            cfg = Config.fromfile(self.detection_model_config_path)
            self.detection_model = init_detector(cfg, checkpoint=self.detection_model_checkpoint_path, device=self.device)
            test_pipeline_cfg = get_test_pipeline_cfg(cfg=cfg)
            test_pipeline_cfg[0].type = 'mmdet.LoadImageFromNDArray'
            self.test_pipeline = Compose(test_pipeline_cfg)
        else:
            raise ValueError(f"Unsupported detection model: {self.detection_model_name}")

        # Initialize Segmentation Model
        if self.segment_model_name == "SAM":
            self.segment_model = sam_model_registry[self.segment_encoder_version](checkpoint=self.segment_model_checkpoint_path)
            self.segment_model.to(device=self.device)
            self.segment_predictor = SamPredictor(self.segment_model)
        # elif self.segment_model_name == "SAM2":
            # Placeholder for SAM2 initialization
            # sam2_model = build_sam2(model_cfg, self.segment_model_checkpoint_path, device=self.device)
            # self.segment_predictor = SAM2ImagePredictor(sam2_model)
        else:
            raise ValueError(f"Unsupported segmentation model: {self.segment_model_name}")

    def update_dataset(self, dataset):
        """
        更新模型依赖的数据集对象及相关属性。
        如果已有 dataset，先删除它。
        """
        if hasattr(self, 'dataset') and self.dataset is not None:
            del self.dataset
        self.dataset = dataset
        self.active_classes = dataset.active_classes
        self.num_classes = len(dataset.report_name)
        self.vis_flags = dataset.flags
        self.vis = self.vis_flags.get('save_detection_images', False) or self.vis_flags.get('save_segmented_images', False)
        
    def multi_class_relation_nms(self, detections, iou_thr=0.1):
        """
        多类别关系NMS：对relation中定义的key类，若其内部任意两个框IoU大于阈值，则合并为最大外包围盒，类别设为key的value（如有多个value取第一个）。
        不同key之间若IoU大于阈值且value有交集，也合并，类别为交集中的第一个。
        其余类别正常保留。
        """
        if iou_thr is None:
            iou_thr = self.nms_threshold

        relation = self.dataset.relation  # {key_id: [value_id, ...]}
        xyxy = detections.xyxy.copy()
        class_id = detections.class_id.copy()
        confidence = detections.confidence.copy()
        mask = getattr(detections, 'mask', None)
        keep_indices = np.ones(len(xyxy), dtype=bool)
        new_boxes, new_class_ids, new_confidences, new_masks = [], [], [], []

        # 记录每个框属于哪个key（1-based），没有则为None
        key_for_box = [None] * len(class_id)
        value_for_box = [set() for _ in range(len(class_id))]
        for key_id, value_ids in relation.items():
            indices = np.where(class_id == key_id - 1)[0]
            for idx in indices:
                key_for_box[idx] = key_id
                value_for_box[idx] = set(value_ids)

        # 两两遍历所有属于relation key的框
        for i in range(len(class_id)):
            for j in range(i + 1, len(class_id)):
                # 两个框都属于relation key
                if key_for_box[i] is not None and key_for_box[j] is not None:
                    # 计算IoU
                    box1, box2 = xyxy[i], xyxy[j]
                    xx1 = max(box1[0], box2[0])
                    yy1 = max(box1[1], box2[1])
                    xx2 = min(box1[2], box2[2])
                    yy2 = min(box1[3], box2[3])
                    inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
                    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
                    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
                    iou = inter / (area1 + area2 - inter + 1e-6)
                    if iou > iou_thr:
                        # value交集
                        common_values = value_for_box[i] & value_for_box[j]
                        if common_values:
                            new_class = sorted(list(common_values))[0]  # 取交集的第一个
                            new_box = [
                                min(box1[0], box2[0]),
                                min(box1[1], box2[1]),
                                max(box1[2], box2[2]),
                                max(box1[3], box2[3])
                            ]
                            new_conf = np.sqrt(confidence[i] * confidence[j])
                            new_boxes.append(new_box)
                            new_class_ids.append(new_class)
                            new_confidences.append(new_conf)
                            # if mask is not None:
                            #     new_masks.append(np.logical_or(mask[i], mask[j]))
                            keep_indices[i] = False
                            keep_indices[j] = False

        # 保留未被合并的框
        xyxy = xyxy[keep_indices]
        class_id = class_id[keep_indices]
        confidence = confidence[keep_indices]
        # if mask is not None:
        #     mask = mask[keep_indices]
        # 拼接新框
        if new_boxes:
            xyxy = np.vstack([xyxy, np.array(new_boxes)])
            class_id = np.concatenate([class_id, np.array(new_class_ids) - 1])  # 保持0-based
            confidence = np.concatenate([confidence, np.array(new_confidences)])
            # if mask is not None:
            #     mask = np.concatenate([mask, np.array(new_masks)])

        new_detections = sv.Detections(
            xyxy=xyxy,
            class_id=class_id,
            confidence=confidence
        )

        if len(new_detections.xyxy) > 0:
            nms_idx = torchvision.ops.nms(
                torch.from_numpy(new_detections.xyxy), 
                torch.from_numpy(new_detections.confidence), 
                self.nms_threshold
            ).numpy().tolist()
            new_detections = new_detections[nms_idx]
            
        # if mask is not None:
        #     new_detections.mask = mask
        return new_detections

    def _assign_probs_after_clustering(self, detections, global_class_ids, nonground_pixel_indices, nonground_points_3d, num_classes):
        """
        对于每个检测，对其3D点进行聚类，并将检测概率赋给最大簇中的点。
        """
        num_nonground_points = nonground_points_3d.shape[0]
        probs = np.zeros((num_nonground_points, num_classes), dtype=np.float32)

        # 找出每个点分别落在了哪些mask里
        # in_mask_nonground: (num_masks, num_nonground_points)
        in_mask_nonground = detections.mask[:, nonground_pixel_indices[:, 0], nonground_pixel_indices[:, 1]]

        # 遍历每个检测结果 (mask)
        for i in range(len(detections)):
            # 获取当前mask对应的点云索引
            point_indices_in_mask = np.where(in_mask_nonground[i])[0]
            
            # 如果mask中的点太少，则跳过
            if len(point_indices_in_mask) < self.min_samples:
                continue
            
            points_for_clustering = nonground_points_3d[point_indices_in_mask]
            
            # 对3D点云进行DBSCAN聚类
            db = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(points_for_clustering)
            labels = db.labels_
            
            # 如果没有点被有效聚类（所有点都是噪声），则跳过
            if np.all(labels == -1):
                continue
            
            # 找到最大的簇
            unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
            main_cluster_label = unique_labels[np.argmax(counts)]
            
            # 筛选出属于最大簇的点的索引 (相对于 point_indices_in_mask)
            main_cluster_mask_in_subset = (labels == main_cluster_label)
            
            # 将这些索引映射回原始非地面点云数组的索引
            final_point_indices = point_indices_in_mask[main_cluster_mask_in_subset]
            
            # 获取当前检测的类别ID和置信度
            class_id = global_class_ids[i] - 1  # 转换为0-based
            confidence = detections.confidence[i]
            
            probs[final_point_indices, class_id] += confidence
            
        return probs
    
    def projection_and_feature_aggregation(self, boolean_maps, class_id, confidence, pixel_indices, num_classes):
        """
        [修改后] 将mask投影到点云，并为每个点聚合来自所有重叠mask的特征。
        该方法现在生成一个全局的概率向量。
        """
        # pixel_indices: (N_points, 2) vu format
        # boolean_maps: (N_masks, H, W)
        # confidence: (N_masks,)
        # class_id: (N_masks,)  <-- 现在期望是全局ID (1-based)
        # num_classes: int      <-- 全局类别的总数
        in_mask = boolean_maps[:, pixel_indices[:, 0], pixel_indices[:, 1]]
        num_points = pixel_indices.shape[0]
        probs = np.zeros((num_points, num_classes), dtype=np.float32)
        
        if num_points == 0:
            return probs

        weighted_conf = in_mask.T * confidence
        # class_id已经是全局的，所以可以直接用
        valid_class_indices = class_id.astype(int) - 1
        np.add.at(probs, (np.arange(num_points)[:, None], valid_class_indices), weighted_conf)
        return probs


    def detection(self, image: np.array) -> tuple:
        class_names = self.active_classes
        if self.detection_model_name == "GroundingDINO":
            detections, detection_time = self.detection_GroundingDINO(image, class_names)
        elif self.detection_model_name == "YoloWorld":
            detections, detection_time = self.detection_YoloWorld(image, class_names)
        else:
            raise ValueError(f"Unsupported detection model: {self.detection_model_name}")
        detections = detections[detections.class_id != None]
        return detections, detection_time

    def detection_GroundingDINO(self, image: np.array, classes: list) -> tuple:
        detection_start_time = time.time()
        detections = self.detection_model.predict_with_classes(
            image=image, 
            classes=classes, 
            box_threshold=self.box_threshold, 
            text_threshold=self.text_threshold
        )
        
        # NMS
        if len(detections.xyxy) > 0:
            nms_idx = torchvision.ops.nms(
                torch.from_numpy(detections.xyxy), 
                torch.from_numpy(detections.confidence), 
                self.nms_threshold
            ).numpy().tolist()
            detections = detections[nms_idx]
        detection_time = (time.time() - detection_start_time) * 1000
        return detections, detection_time

    def detection_YoloWorld(self, image: np.array, classes: list) -> tuple:
        # YoloWorld expects a list of lists for class names
        data_info = dict(img=image, img_id=0, texts=[[c] for c in classes])
        data_info = self.test_pipeline(data_info)
        data_batch = dict(
            inputs=data_info['inputs'].unsqueeze(0).to(self.device),
            data_samples=[data_info['data_samples'].to(self.device)]
        )
        detection_start_time = time.time()
        with torch.no_grad():
            output = self.detection_model.test_step(data_batch)[0]
        pred_instances = output.pred_instances[output.pred_instances.scores > self.box_threshold]
        detection_time = (time.time() - detection_start_time) * 1000
        
        detections = sv.Detections(
            xyxy=pred_instances.bboxes.cpu().numpy(),
            class_id=pred_instances.labels.cpu().numpy(),
            confidence=pred_instances.scores.cpu().numpy()
        )
        return detections, detection_time

    # Prompting SAM with detected boxes
    def segment(self, sam_predictor: SamPredictor, image: torch.Tensor, xyxy: torch.Tensor) -> torch.Tensor:
        sam_predictor.set_image(image)
        transformed_boxes = sam_predictor.transform.apply_boxes_torch(xyxy, image.shape[:2]).to(sam_predictor.model.device)
        
        masks, scores, _ = sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

        min_mask_area = 500  # set your desired threshold
        valid_indices = [i for i, mask in enumerate(masks) if mask.sum().item() >= min_mask_area]
        
        if not valid_indices:
            return torch.empty((0, *image.shape[:2]), device=masks.device, dtype=masks.dtype), torch.empty((0,), device=scores.device, dtype=scores.dtype), []

        masks = torch.stack([masks[i].squeeze(0) for i in valid_indices])
        scores = scores[valid_indices]

        return masks, scores.squeeze(1) ,valid_indices
    
    def predict_for_camera(self, 
                           image: np.array, 
                           proj_info: dict, 
                           ground_indices: np.ndarray, 
                           nonground_indices: np.ndarray, 
                           frame_id: str, 
                           cam_name: str) -> tuple:
        """为单个相机视角处理点云，返回概率和耗时。"""
        
        # 1. 目标检测
        detections, detection_time = self.detection(image)
        if len(detections.xyxy) == 0:
            return None, None, None, None, detection_time, 0
        
        detections = self.multi_class_relation_nms(detections)

        # 2. 图像分割
        segment_start_time = time.time()
        box_tensor = torch.from_numpy(detections.xyxy).to(self.device)
        masks, seg_scores, valid_indices = self.segment(sam_predictor=self.segment_predictor, image=image, xyxy=box_tensor)
        segment_time = (time.time() - segment_start_time) * 1000
        
        detections = detections[valid_indices]
        detections.mask = masks.cpu().numpy()
        # detections.confidence = np.sqrt(detections.confidence * seg_scores.cpu().numpy())
        detections.confidence = np.minimum(detections.confidence, seg_scores.cpu().numpy())
        
        # 3. 筛选出在该相机视角下的地面和非地面点
        cam_indices = proj_info['indices']
        is_ground = np.isin(cam_indices, ground_indices)
        is_nonground = np.isin(cam_indices, nonground_indices)

        cam_ground_indices = cam_indices[is_ground]
        cam_nonground_indices = cam_indices[is_nonground]
        
        # 获取对应点的像素坐标 (vu格式)
        pixel_coords = proj_info['pixels']
        ground_pixel_indices = pixel_coords[is_ground]
        nonground_pixel_indices = pixel_coords[is_nonground]

        # 4. 概率计算
        ground_probs, nonground_probs = None, None
        
        detection_class_names = np.array([self.active_classes[cid] for cid in detections.class_id])
        
        is_ground_detection = np.isin(detection_class_names, self.dataset.ground_classes)
        is_nonground_detection = np.isin(detection_class_names, self.dataset.nonground_classes)

        ground_detections = detections[is_ground_detection]
        nonground_detections = detections[is_nonground_detection]


        if ground_pixel_indices.shape[0] > 0 and len(ground_detections) > 0:
            ground_global_class_ids = np.array([self.dataset.name_to_index[self.active_classes[cid]] for cid in ground_detections.class_id])
            ground_probs = self.projection_and_feature_aggregation(
                ground_detections.mask, ground_global_class_ids, ground_detections.confidence, ground_pixel_indices, self.num_classes
            )

        if nonground_pixel_indices.shape[0] > 0 and len(nonground_detections) > 0:
            nonground_global_class_ids = np.array([self.dataset.name_to_index[self.active_classes[cid]] for cid in nonground_detections.class_id])
            
            # 对非地面点进行聚类过滤或直接聚合
            if self.use_cluster:
                nonground_points_3d = proj_info['points'][is_nonground]
                nonground_probs = self._assign_probs_after_clustering(
                    detections=nonground_detections,
                    global_class_ids=nonground_global_class_ids,
                    nonground_pixel_indices=nonground_pixel_indices,
                    nonground_points_3d=nonground_points_3d,
                    num_classes=self.num_classes
                )
            else:
                # 如果不使用聚类，则使用特征聚合方法
                nonground_probs = self.projection_and_feature_aggregation(
                    nonground_detections.mask, nonground_global_class_ids, nonground_detections.confidence, nonground_pixel_indices, self.num_classes
                )


        if self.save_features:
            self.dataset.save_features(
                index=frame_id, 
                cam_name=cam_name,
                cam_ground_indices=cam_ground_indices,
                ground_probs=ground_probs,
                cam_nonground_indices=cam_nonground_indices,
                nonground_probs=nonground_probs
            )
            
        # 5. 可视化
        if self.vis:
            labels_text = [f"{self.active_classes[cid]} {conf:.2f}" for cid, conf in zip(detections.class_id, detections.confidence)]
            detection_img = sv.BoundingBoxAnnotator(thickness=1).annotate(image.copy(), detections)
            detection_img = LABEL_ANNOTATOR.annotate(detection_img, detections, labels=labels_text)
            segmentation_img = sv.MaskAnnotator().annotate(image.copy(), detections)
            segmentation_img = LABEL_ANNOTATOR.annotate(segmentation_img, detections, labels=labels_text)
            self.dataset.save_visualizations(detection_img, segmentation_img, frame_id, cam_name)
        
        return cam_ground_indices, ground_probs, cam_nonground_indices, nonground_probs, detection_time, segment_time
