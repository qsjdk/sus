import os
from pathlib import Path
from PIL import Image
import random
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image

# 输入输出路径
img_dir = Path('./data/domain1/train/imgs')
mask_dir = Path('./data/domain1/train/mask')

out_img_dir = Path('./data/domain1/train/imgs_aug/crop')
out_mask_dir = Path('./data/domain1/train/mask_aug/crop')
out_img_dir.mkdir(parents=True, exist_ok=True)
out_mask_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------
# 随机裁剪 + resize 的函数
# ---------------------------
def random_crop_resize(img, mask, crop_size=(200, 200), resize=(256, 256)):
    """同步随机裁剪+resize"""
    i, j, h, w = transforms.RandomCrop.get_params(img, output_size=crop_size)
    img_cropped = TF.crop(img, i, j, h, w)
    mask_cropped = TF.crop(mask, i, j, h, w)
    img_resized = TF.resize(img_cropped, resize, interpolation=Image.Resampling.BILINEAR)
    mask_resized = TF.resize(mask_cropped, resize, interpolation=Image.Resampling.NEAREST)
    return img_resized, mask_resized

# ---------------------------
# 对所有图像执行同步增强
# ---------------------------
for img_path in img_dir.glob('*.png'):
    mask_path = mask_dir / img_path.name

    img = Image.open(img_path)
    mask = Image.open(mask_path)

    aug_img, aug_mask = random_crop_resize(img, mask)

    aug_img.save(out_img_dir / f'{img_path.stem}_crop.png')
    aug_mask.save(out_mask_dir / f'{mask_path.stem}_crop.png')

print("✅ 已保存随机裁剪增强图像与掩码到:")
print("  图像目录:", out_img_dir)
print("  掩码目录:", out_mask_dir)

