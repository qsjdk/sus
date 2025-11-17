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
    stem, _ = os.path.splitext(base)
    if stem.endswith("_OUT"):
        stem = stem[:-4]
    for ext in EXTS:
        p = os.path.join(gt_dir, stem + ext)
        if os.path.exists(p):
            return p
    return ""

def load_mask_bin_rgb01(path: str) -> np.ndarray:
    """
    读取掩膜图像并二值化：
    - 无论原图是 1 通道还是 3 通道，都转为灰度；
    - 取像素值 > 0 作为前景（自动去除黑色背景）；
    - 输出 0/1 的 uint8 数组。
    """
    img = Image.open(path).convert("L")  # 转为灰度图，保证通道一致
    arr = np.array(img)
    mask = (arr > 0).astype(np.uint8)
    return mask


# ---------------- 核心几何工具 ----------------
def _surface(mask: np.ndarray) -> np.ndarray:
    """二值掩膜的表面像素（腐蚀后取异或；极小目标回退自身）"""
    if not mask.any():
        return np.zeros_like(mask, bool)
    er = ndi.binary_erosion(mask, structure=np.ones((3,)*mask.ndim, bool), border_value=0)
    surf = np.logical_xor(mask, er)
    return surf if surf.any() else mask.astype(bool)

def _assd_between(A: np.ndarray, B: np.ndarray, spacing=None) -> float:
    """
    对称平均表面距离 (ASSD)
    A、B 为 bool（二值前景），仅在表面上计算最近距离。
    """
    if spacing is None:
        spacing = tuple([1.0] * A.ndim)

    a_any, b_any = A.any(), B.any()
    if not a_any and not b_any:
        return 0.0
    if not a_any or not b_any:
        return float('inf')

    As = _surface(A)
    Bs = _surface(B)

    # 到对方前景的最近距离：对方前景的补集做 EDT
    dt_to_B = ndi.distance_transform_edt(~B, sampling=spacing)
    dt_to_A = ndi.distance_transform_edt(~A, sampling=spacing)

    dA = dt_to_B[As].astype(np.float64)
    dB = dt_to_A[Bs].astype(np.float64)

    # 按公式：两向平均（分母为 len(X)+len(Y)）
    assd = (dA.sum() + dB.sum()) / (dA.size + dB.size)
    return float(assd)

def _hd95_between(A: np.ndarray, B: np.ndarray, spacing=None) -> float:
    """
    对称 Hausdorff 距离的 95 分位 (HD95)
    """
    if spacing is None:
        spacing = tuple([1.0] * A.ndim)

    a_any, b_any = A.any(), B.any()
    if not a_any and not b_any:
        return 0.0
    if not a_any or not b_any:
        return float('inf')

    As = _surface(A)
    Bs = _surface(B)

    dt_to_B = ndi.distance_transform_edt(~B, sampling=spacing)
    dt_to_A = ndi.distance_transform_edt(~A, sampling=spacing)

    d1 = np.percentile(dt_to_B[As], 95)
    d2 = np.percentile(dt_to_A[Bs], 95)
    return float(max(d1, d2))

# ---------------- 三个指标 ----------------
def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Dice = 2|X∩Y| / (|X|+|Y|)
    只在前景(>0)上计算；都空 -> 1.0（如不希望如此可改为0.0）
    """
    P = pred.astype(bool)
    G = gt.astype(bool)
    inter = np.logical_and(P, G).sum()
    denom = P.sum() + G.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * inter / denom)

def assd_metric(pred: np.ndarray, gt: np.ndarray, spacing=None) -> float:
    P = pred.astype(bool)
    G = gt.astype(bool)
    return _assd_between(P, G, spacing)

def hd95_metric(pred: np.ndarray, gt: np.ndarray, spacing=None) -> float:
    P = pred.astype(bool)
    G = gt.astype(bool)
    return _hd95_between(P, G, spacing)

# ---------------- 评估主流程 ----------------
def evaluate(pred_dir: str, gt_dir: str, spacing=None) -> Tuple[float, float, float]:
    preds = list_images(pred_dir)
    if not preds:
        raise FileNotFoundError(f"No images in {pred_dir}")

    D, A, H = [], [], []
    missing = []

    print(f"Evaluating\n  Pred: {pred_dir}\n  GT  : {gt_dir}\n  N   : {len(preds)}")
    print("-"*60)

    for i, p in enumerate(preds, 1):
        g = match_gt_path(p, gt_dir)
        if not g:
            missing.append(os.path.basename(p))
            continue

        P = load_mask_bin_rgb01(p)
        G = load_mask_bin_rgb01(g)

        if P.shape != G.shape:
            # 保持标签最近邻缩放
            P = np.array(Image.fromarray(P).resize((G.shape[1], G.shape[0]), resample=Image.NEAREST), dtype=np.uint8)

        d = dice(P, G)
        a = assd_metric(P, G, spacing)
        h = hd95_metric(P, G, spacing)

        D.append(d); A.append(a); H.append(h)
        print(f"[{i:03d}] {os.path.basename(p):<30}  Dice={d:.4f}  ASSD={a:.4f}  HD95={h:.4f}")

    print("-"*60)
    if missing:
        print(f"⚠ Missing GT for {len(missing)} preds:")
        for name in missing[:10]:
            print("   -", name)
        if len(missing) > 10:
            print("   ...")

    mean_d = float(np.mean(D)) if D else 0.0
    mean_a = float(np.mean(A)) if A else float('inf')
    mean_h = float(np.mean(H)) if H else float('inf')
    print(f"Overall: Dice={mean_d:.4f}  ASSD={mean_a:.4f}  HD95={mean_h:.4f}")
    return mean_d, mean_a, mean_h


if __name__ == "__main__":
    # 修改为你的实际路径
    pred_dir = "data/Domain1/test/imgs_out_crop_aug"
    gt_dir = "data/Domain1/test/mask"
    # 如像素间距不是 1，可传 spacing=(sy, sx) 或 (sz, sy, sx)
    evaluate(pred_dir, gt_dir, spacing=None)








