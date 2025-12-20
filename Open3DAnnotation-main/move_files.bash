#!/bin/bash

# 设置源目录和目标目录
source_dir="/home/sustech/OpenVocabulary/lidar_photo/MyOpenProject/outputs/GroundingDINO_SAM/sequences"
target_dir="/home/sustech/OpenVocabulary/lidar/data/semantickitti-noisy/sequences"

# 创建目标目录（如果不存在）
mkdir -p "$target_dir"

# 循环处理00到10的编号
for i in {00..10}; do
    # 构造源子目录路径
    if [[ "$i" == "08" ]]; then
        echo "警告：跳过目录 $i"
        continue
    fi

    source_subdir="$source_dir/$i"
    
    # 检查源子目录是否存在
    if [ -d "$source_subdir" ]; then
        # 创建目标子目录（保持相同结构）
        mkdir -p "$target_dir/$i"
        
        # 执行带进度显示的复制
        echo "正在复制 $i/pred_labels..."
        if cp -r "$source_subdir/pred_labels/" "$target_dir/$i/labels"; then
            echo "[√] 成功复制 $i/labels"
        else
            echo "[×] 错误：$i/labels 复制失败"
        fi
    else
        echo "警告：目录 '$source_subdir' 不存在，已跳过。"
    fi
done

echo "操作完成。"

cp -r /home/sustech/Dataset/semantickitti/semantickitti/dataset/sequences/08/labels/ /home/sustech/OpenVocabulary/lidar/data/semantickitti-noisy/sequences/08/
