import os
import numpy as np
from PIL import Image
from typing import List, Tuple
from scipy import ndimage as ndi

# ---------------- I/O ----------------
EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

def list_images(folder: str) -> List[str]:
    return sorted([
        os.path.join(folder, f) for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in EXTS
    ])

def match_gt_path(pred_path: str, gt_dir: str) -> str:
    base = os.path.basename(pred_path)
    stem, ext = os.path.splitext(base)
    if stem.endswith("_OUT"):
        stem = stem[:-4]
    # 先试同后缀，再试所有后缀
    cand = os.path.join(gt_dir, stem + ext)
    if os.path.exists(cand):
        return cand
    for e in EXTS:
        p = os.path.join(gt_dir, stem + e)
        if os.path.exists(p):
            return p
    return ""

def load_mask_bin(path: str) -> np.ndarray:
    """统一转灰度并二值化: >0 为前景(1), 否则0"""
    arr = np.array(Image.open(path).convert("L"))
    return (arr > 0).astype(np.uint8)

# ---------------- 几何工具 ----------------
def _surface(mask: np.ndarray) -> np.ndarray:
    """二值掩膜的表面像素；极小目标回退自身"""
    if not mask.any():
        return np.zeros_like(mask, bool)
    er = ndi.binary_erosion(mask, structure=np.ones((3,)*mask.ndim, bool), border_value=0)
    surf = np.logical_xor(mask, er)
    return surf if surf.any() else mask.astype(bool)

# ---------------- Dice(数据集级别聚合) ----------------
def dice_counts(pred: np.ndarray, gt: np.ndarray) -> Tuple[int, int, int]:
    """返回 (|X∩Y|, |X|, |Y|) 用于数据集级别聚合"""
    # print(pred.max())
    P = pred.astype(bool); G = gt.astype(bool)
    inter = int(np.logical_and(P, G).sum())
    return inter, int(P.sum()), int(G.sum())

# ---------------- ASSD(数据集级别聚合) ----------------
def assd_counts(pred: np.ndarray, gt: np.ndarray, spacing=None) -> Tuple[float, int, bool]:
    """
    返回 (距离总和, 表面点数总和, 是否为inf)
    - 两边都空：返回(0.0, 0, False)，不影响整体
    - 一边空一边非空：标记 inf=True
    - 正常：累加两向最近距离总和与两侧表面点数
    """
    P = pred.astype(bool); G = gt.astype(bool)
    if spacing is None:
        spacing = (1.0,) * P.ndim

    p_any, g_any = P.any(), G.any()
    if not p_any and not g_any:
        return 0.0, 0, False
    if not p_any or not g_any:
        return 0.0, 0, True  # 该图视为无限大

    Ps, Gs = _surface(P), _surface(G)
    dtG = ndi.distance_transform_edt(~G, sampling=spacing)
    dtP = ndi.distance_transform_edt(~P, sampling=spacing)

    dA = dtG[Ps].astype(np.float64)
    dB = dtP[Gs].astype(np.float64)

    dist_sum = float(dA.sum() + dB.sum())
    count_sum = int(dA.size + dB.size)
    return dist_sum, count_sum, False

# ---------------- HD95(逐图再均值) ----------------
def hd95_one(pred: np.ndarray, gt: np.ndarray, spacing=None) -> float:
    P = pred.astype(bool); G = gt.astype(bool)
    if spacing is None:
        spacing = (1.0,) * P.ndim
    p_any, g_any = P.any(), G.any()
    if not p_any and not g_any:
        return 0.0
    if not p_any or not g_any:
        return float('inf')

    Ps, Gs = _surface(P), _surface(G)
    dtG = ndi.distance_transform_edt(~G, sampling=spacing)
    dtP = ndi.distance_transform_edt(~P, sampling=spacing)
    d1 = np.percentile(dtG[Ps], 95)
    d2 = np.percentile(dtP[Gs], 95)
    return float(max(d1, d2))

# ---------------- 主流程 ----------------
def evaluate(pred_dir: str, gt_dir: str, spacing=None):
    preds = list_images(pred_dir)
    if not preds:
        raise FileNotFoundError(f"No images in {pred_dir}")

    # Dice 数据集级别累计量
    inter_sum = 0
    x_sum = 0
    y_sum = 0

    # ASSD 数据集级别累计量
    assd_dist_sum = 0.0
    assd_count_sum = 0
    assd_infinite = False  # 一旦某图为inf，则整体inf

    # HD95 逐图列表
    hd95_list = []

    print(f"Evaluating\n  Pred: {pred_dir}\n  GT  : {gt_dir}\n  N   : {len(preds)}")
    print("-"*60)

    for i, p in enumerate(preds, 1):
        g = match_gt_path(p, gt_dir)
        if not g:
            print(f"[{i:03d}] {os.path.basename(p):<30}  SKIP (GT missing)")
            continue

        P = load_mask_bin(p)
        G = load_mask_bin(g)

        if P.shape != G.shape:
            P = np.array(Image.fromarray(P).resize((G.shape[1], G.shape[0]), resample=Image.NEAREST), dtype=np.uint8)

        # --- 累计 Dice 分子/分母 ---
        inter, x, y = dice_counts(P, G)
        inter_sum += inter; x_sum += x; y_sum += y

        # --- 累计 ASSD 距离和/点数（或标记inf）---
        dsum, csum, is_inf = assd_counts(P, G, spacing)
        if is_inf:
            assd_infinite = True
        else:
            assd_dist_sum += dsum
            assd_count_sum += csum

        # --- HD95 逐图 ---
        h = hd95_one(P, G, spacing)
        hd95_list.append(h)

    print("-"*60)

    # 数据集级别 Dice
    dice_dataset = 0.0 if (x_sum + y_sum) == 0 else float(2.0 * inter_sum / (x_sum + y_sum))

    # 数据集级别 ASSD
    if assd_infinite:
        assd_dataset = float('inf')
    else:
        assd_dataset = float(assd_dist_sum / assd_count_sum) if assd_count_sum > 0 else 0.0

    # HD95 逐图均值
    hd95_mean = float(np.mean(hd95_list)) if hd95_list else float('inf')

    print(f"Dataset-level Dice : {dice_dataset:.6f}   (2*Σ|∩| / (Σ|X|+Σ|Y|))")
    if np.isfinite(assd_dataset):
        print(f"Dataset-level ASSD : {assd_dataset:.6f}   (Σ两向距离 / Σ两侧表面点数)")
    else:
        print(f"Dataset-level ASSD : inf   (存在一边空一边非空的样本)")
    print(f"Mean HD95 over imgs: {hd95_mean:.6f}")

    return dice_dataset, assd_dataset, hd95_mean


if __name__ == "__main__":
    pred_dir = "data/Domain1/test/imgs_out_no_aug"
    gt_dir   = "data/Domain1/test/mask"
    evaluate(pred_dir, gt_dir, spacing=None)  # 如有体素尺寸可传 (sy,sx) 或 (sz,sy,sx)