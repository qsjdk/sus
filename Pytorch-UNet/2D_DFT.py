import os
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# -----------------------------
# 参数设置
# -----------------------------
base_dir = "data"   # 根目录
domains = ["Domain1", "Domain2", "Domain3", "Domain4", "Domain5"]

# 每个域最多展示几张图
max_imgs_per_domain = 3


def get_image_paths(domain_dir):
    img_dir = os.path.join(domain_dir, "train", "imgs")
    paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        paths.extend(glob.glob(os.path.join(img_dir, ext)))
    return sorted(paths)


def load_gray_image(path):
    """
    用 PIL 读图并转换为灰度 + float32 numpy 数组
    """
    img = Image.open(path).convert("L")       # "L" = 8-bit 灰度
    img_arr = np.array(img, dtype=np.float32)
    return img_arr


def fft2_and_spectrum(img_gray):
    """
    img_gray: 2D numpy array, float32, 灰度图
    返回：幅度谱（已中心化+对数）、相位谱（已中心化）
    """
    # 2D FFT
    f = np.fft.fft2(img_gray)
    # 频谱中心化（把低频移动到中心）
    fshift = np.fft.fftshift(f)

    # 幅度谱，+1 防止 log(0)
    magnitude = np.abs(fshift)
    magnitude_log = np.log(magnitude + 1.0)

    # 相位谱
    phase = np.angle(fshift)

    return magnitude_log, phase


def normalize_to_uint8(arr):
    """
    把任意实数数组线性归一化到 [0,255]，方便用灰度图显示
    """
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max - arr_min < 1e-8:
        return np.zeros_like(arr, dtype=np.uint8)
    norm = (arr - arr_min) / (arr_max - arr_min)
    norm = (norm * 255).astype(np.uint8)
    return norm


def process_and_show(domain_name, img_paths):
    """
    对某个域的多张图做 FFT，并画出：
    - 原始灰度图
    - 幅度谱（灰度调整 + 中心化）
    - 相位谱（灰度调整 + 中心化）
    """
    for i, p in enumerate(img_paths[:max_imgs_per_domain]):
        print(f"{domain_name} - processing: {p}")
        img_gray = load_gray_image(p)

        # 2D FFT + 频谱中心化
        mag_log, phase = fft2_and_spectrum(img_gray)

        # 灰度归一化
        mag_norm = normalize_to_uint8(mag_log)
        phase_norm = normalize_to_uint8(phase)

        # 显示
        plt.figure(figsize=(9, 3))
        plt.suptitle(f"{domain_name} - sample {i+1}", fontsize=12)

        plt.subplot(1, 3, 1)
        plt.imshow(img_gray, cmap="gray")
        plt.title("Original (Gray)")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(mag_norm, cmap="gray")
        plt.title("Magnitude Spectrum")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(phase_norm, cmap="gray")
        plt.title("Phase Spectrum")
        plt.axis("off")

        plt.tight_layout()
        # 如需保存图像，可以取消下一行注释：
        # plt.savefig(f"{domain_name}_sample{i+1}_fft.png", dpi=300)
        plt.show()


def main():
    for d in domains:
        domain_dir = os.path.join(base_dir, d)
        img_paths = get_image_paths(domain_dir)
        print(f"{d}: found {len(img_paths)} images")
        if len(img_paths) == 0:
            continue
        process_and_show(d, img_paths)


if __name__ == "__main__":
    main()

