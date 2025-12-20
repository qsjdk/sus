#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import csv
import sys
import shutil
from pathlib import Path
from typing import List, Tuple, Optional

# =========================
# 配置：输入输出根目录
# Windows 注意用 r"..." 或者 "E:/..."
# =========================
INPUT_ROOT = Path(r"E:\hejun\demo\dataset\sequences\00(timestamp)\labels")
OUTPUT_ROOT = Path(r"E:\hejun\demo\dataset\sequences\00")

# 输出子目录（你指定的）
OUT_IMAGES = OUTPUT_ROOT / "image_0"
OUT_LABELS = OUTPUT_ROOT / "labels"
OUT_BINS   = OUTPUT_ROOT / "velodyne"

# 输出映射表
MAP_CSV = OUTPUT_ROOT / "index_map.csv"

# 支持的文件类型
EXTS = [".png", ".label", ".bin"]

# 从文件名提取数字时间戳（支持小数和超长整数）
NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def is_under(path: Path, root: Path) -> bool:
    """判断 path 是否在 root 目录下"""
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def parse_timestamp_from_stem(stem: str) -> Optional[float]:
    """
    从文件名(不含后缀)中提取数字作为时间戳:
    - 1734939309.493734
    - 173493930910400529677
    """
    m = NUM_RE.search(stem)
    if not m:
        return None
    s = m.group(0)
    try:
        return float(s)
    except Exception:
        return None


def collect_files(root: Path, skip_root: Path) -> List[Path]:
    """递归收集 root 下所有目标后缀文件，跳过 skip_root（比如 output 目录）"""
    files: List[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if is_under(p, skip_root):
            continue
        if p.suffix.lower() in EXTS:
            files.append(p)
    return files


def sort_key(p: Path) -> Tuple[int, float, float]:
    """
    排序 key：
    1) 能解析到时间戳的排前面（0），解析不到排后面（1）
    2) 时间戳数值
    3) 回退：mtime
    """
    ts = parse_timestamp_from_stem(p.stem)
    mtime = p.stat().st_mtime
    if ts is None:
        return (1, 0.0, mtime)
    else:
        return (0, ts, mtime)


def ensure_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)


def simple_progress(prefix: str, i: int, total: int):
    """无 tqdm 时的简易进度显示"""
    if total <= 0:
        return
    pct = (i / total) * 100.0
    sys.stdout.write(f"\r{prefix} {i}/{total} ({pct:6.2f}%)")
    sys.stdout.flush()
    if i == total:
        sys.stdout.write("\n")


def copy_and_rename(
    sorted_paths: List[Path],
    out_dir: Path,
    new_ext: str,
    csv_rows: List[List[str]],
    group_name: str
):
    """
    把 sorted_paths 按顺序复制到 out_dir，并重命名成 000000.xxx
    同时写入 csv_rows
    """
    ensure_dir(out_dir)
    total = len(sorted_paths)

    if total == 0:
        print(f"[INFO] {group_name}: 0 files, skipped.")
        return

    # 无 tqdm：简易进度
    print(f"[INFO] {group_name}: copying {total} files...")
    for idx, src in enumerate(sorted_paths):
        new_name = f"{idx:06d}{new_ext}"
        dst = out_dir / new_name

        if dst.exists():
            raise RuntimeError(f"[ERROR] Output file already exists: {dst}\n"
                                f"Please delete OUTPUT_ROOT folder and retry: {out_dir.parent}")

        shutil.copy2(src, dst)

        ts = parse_timestamp_from_stem(src.stem)
        ts_str = "" if ts is None else f"{ts:.16f}"
        csv_rows.append([new_name, str(src), str(dst), ts_str])

        # 每次更新进度（你也可以改成每 10 次更新一次）
        simple_progress(f"{group_name}:", idx + 1, total)


def main():
    if not INPUT_ROOT.exists():
        raise FileNotFoundError(f"Input root not found: {INPUT_ROOT}")

    # 创建输出目录
    ensure_dir(OUTPUT_ROOT)

    # 收集文件（从 INPUT_ROOT 下递归搜）
    all_files = collect_files(INPUT_ROOT, OUTPUT_ROOT)

    # 分类
    pngs   = [p for p in all_files if p.suffix.lower() == ".png"]
    labels = [p for p in all_files if p.suffix.lower() == ".label"]
    bins   = [p for p in all_files if p.suffix.lower() == ".bin"]

    # 排序
    pngs_sorted   = sorted(pngs, key=sort_key)
    labels_sorted = sorted(labels, key=sort_key)
    bins_sorted   = sorted(bins, key=sort_key)

    # CSV 映射表（每行：new_name, src_path, dst_path, parsed_timestamp）
    csv_rows: List[List[str]] = [["new_name", "src_path", "dst_path", "parsed_timestamp"]]

    # 复制并重命名 + 进度条
    copy_and_rename(pngs_sorted,   OUT_IMAGES, ".png",   csv_rows, "PNG -> images_0")
    copy_and_rename(labels_sorted, OUT_LABELS, ".label", csv_rows, "LABEL -> labels")
    copy_and_rename(bins_sorted,   OUT_BINS,   ".bin",   csv_rows, "BIN -> pointcloud")

    # 写映射表
    ensure_dir(MAP_CSV.parent)
    with MAP_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)

    print("\n==== Done ====")
    print(f"Input : {INPUT_ROOT}")
    print(f"Output: {OUTPUT_ROOT}")
    print(f"PNG   : {len(pngs_sorted)} -> {OUT_IMAGES}")
    print(f"LABEL : {len(labels_sorted)} -> {OUT_LABELS}")
    print(f"BIN   : {len(bins_sorted)} -> {OUT_BINS}")
    print(f"Map   : {MAP_CSV}")


if __name__ == "__main__":
    main()
