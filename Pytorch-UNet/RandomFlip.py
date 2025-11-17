import os
from pathlib import Path
from PIL import Image
import random
from torchvision.transforms import functional as TF

# 输入输出路径
img_dir = Path('./data/domain1/train/imgs')
mask_dir = Path('./data/domain1/train/mask')

out_img_dir = Path('./data/domain1/train/imgs_aug/flip')
out_mask_dir = Path('./data/domain1/train/mask_aug/flip')
out_img_dir.mkdir(parents=True, exist_ok=True)
out_mask_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------
# 同步随机翻转函数
# ---------------------------
def random_flip(img, mask, p_h=0.5, p_v=0.5):
    """同步水平和垂直翻转"""
    if random.random() < p_h:
        img = TF.hflip(img)
        mask = TF.hflip(mask)
    if random.random() < p_v:
        img = TF.vflip(img)
        mask = TF.vflip(mask)
    return img, mask

# ---------------------------
# 对所有图像执行增强
# ---------------------------
for img_path in img_dir.glob('*.png'):
    mask_path = mask_dir / img_path.name
    img = Image.open(img_path)
    mask = Image.open(mask_path)

    aug_img, aug_mask = random_flip(img, mask)

    aug_img.save(out_img_dir / f'{img_path.stem}_flip.png')
    aug_mask.save(out_mask_dir / f'{mask_path.stem}_flip.png')

print("✅ 已保存翻转增强图像与掩码到:")
print("  图像目录:", out_img_dir)
print("  掩码目录:", out_mask_dir)

