import numpy as np
from sklearn.metrics import confusion_matrix
import csv
from pathlib import Path
from typing import List

class IOUCalculator:
    """
    一个用于计算和管理语义分割指标（如IoU）的类。
    它累积混淆矩阵，并提供计算和保存各种指标的方法。
    """
    def __init__(self, class_names: List[str], ignored_label: int = 0, ignored_class_name: str = 'unknown'):
        """
        初始化计算器。
        :param class_names: **有效类别**的名称列表，不应包含忽略类。
        :param ignored_label: 被忽略的标签的整数ID。默认为0。
        :param ignored_class_name: 被忽略的标签的名称，用于报告。
        """
        # [新设计] class_names 只包含有效类别
        self.active_class_names = class_names
        self.ignored_label = ignored_label
        self.ignored_class_name = ignored_class_name

        # [新设计] 完整的类别列表，用于报告
        # 假设忽略类总是ID 0
        self.full_class_names = [self.ignored_class_name] + self.active_class_names
        
        # [新设计] 总类别数现在是有效类别数 + 1
        self.num_classes = len(self.full_class_names)
        
        # [新设计] 要忽略的索引现在由 ignored_label 决定
        self.ignore_indices = [self.ignored_label]
        
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, preds: np.ndarray, labels: np.ndarray):
        """
        使用新的一批预测和标签来更新累积的混淆矩阵。
        :param preds: 预测的标签数组。
        :param labels: 真实的标签数组。
        """

        # 确保标签仍在有效范围内（可选，但更稳健）
        in_range_mask = (labels >= 0) & (labels < self.num_classes) & \
                        (preds >= 0) & (preds < self.num_classes)
        labels_valid = labels[in_range_mask]
        preds_valid = preds[in_range_mask]
        
        # 计算当前批次的混淆矩阵并累加
        current_cm = confusion_matrix(labels_valid.flatten(), preds_valid.flatten(), labels=np.arange(self.num_classes))
        self.confusion_matrix += current_cm
    
    def reset(self):
        """重置累积的混淆矩阵。"""
        self.confusion_matrix.fill(0)

    def get_metrics(self) -> dict:
        """
        从累积的混淆矩阵中计算详细的性能指标。
        :return: 一个包含各种指标的字典。
        """
        confusion_matrix = self.confusion_matrix.copy()
        # 排除被忽略的类
        confusion_matrix[self.ignore_indices, :] = 0
        confusion_matrix[:, self.ignore_indices] = 0
        
        tp = np.diag(confusion_matrix)
        fp = confusion_matrix.sum(axis=0) - tp
        fn = confusion_matrix.sum(axis=1) - tp
        
        iou = tp / (tp + fp + fn + 1e-15)
        precision = tp / (tp + fp + 1e-15)
        recall = tp / (tp + fn + 1e-15)
        
        # 计算平均指标时，排除被忽略的类
        valid_indices = [i for i in range(self.num_classes) if i not in self.ignore_indices]
        
        mIoU = np.nanmean(iou[valid_indices])
        mAcc = np.nanmean(precision[valid_indices]) # 也可理解为类别平均精度
        
        # 整体精度
        total_acc = tp.sum() / (self.confusion_matrix.sum() + 1e-15)
        
        return {
            "iou_per_class": iou,
            "precision_per_class": precision,
            "recall_per_class": recall,
            "mIoU": mIoU,
            "mAcc": mAcc,
            "total_acc": total_acc,
            "tp": tp,
            "fp": fp,
            "fn": fn
        }

    def report(self, output_path: Path):
        """
        生成并保存一份包含混淆矩阵和各项指标的CSV报告。
        :param output_path: CSV文件的保存路径。
        """
        metrics = self.get_metrics()
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            
            # 写入摘要
            writer.writerow(["Summary"])
            writer.writerow(["mIoU", f"{metrics['mIoU']:.4f}"])
            writer.writerow(["Total Accuracy", f"{metrics['total_acc']:.4f}"])
            writer.writerow(["Mean Accuracy (Precision)", f"{metrics['mAcc']:.4f}"])
            writer.writerow([]) # 空行

            # 写入每类指标
            writer.writerow(["Per-Class Metrics"])
            header = ["Class", "IoU", "Precision", "Recall", "TP", "FP", "FN"]
            writer.writerow(header)
            for i in range(self.num_classes):
                # [新设计] 使用 full_class_names
                class_name = self.full_class_names[i]
                row = [
                    class_name,
                    f"{metrics['iou_per_class'][i]:.4f}",
                    f"{metrics['precision_per_class'][i]:.4f}",
                    f"{metrics['recall_per_class'][i]:.4f}",
                    metrics['tp'][i],
                    metrics['fp'][i],
                    metrics['fn'][i]
                ]
                writer.writerow(row)
            writer.writerow([]) # 空行

            # 写入混淆矩阵
            writer.writerow(["Confusion Matrix"])
            # [新设计] 使用 full_class_names
            writer.writerow([""] + self.full_class_names)
            for i, row_data in enumerate(self.confusion_matrix):
                writer.writerow([self.full_class_names[i]] + list(row_data))