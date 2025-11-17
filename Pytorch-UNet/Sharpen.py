import os
from pathlib import Path
from PIL import Image, ImageFilter

# 输入输出路径
img_dir = Path('./data/domain1/train/imgs')
mask_dir = Path('./data/domain1/train/mask')

out_img_dir = Path('./data/domain1/train/imgs_aug/sharpen')
out_mask_dir = Path('./data/domain1/train/mask_aug/sharpen')
out_img_dir.mkdir(parents=True, exist_ok=True)
out_mask_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------
# 同步锐化增强函数
# ---------------------------
def apply_sharpen(img, mask):
    """仅对图像进行锐化，mask 保持不变"""
    img_sharp = img.filter(ImageFilter.SHARPEN)
    return img_sharp, mask

# ---------------------------
# 对所有图像执行增强
# ---------------------------
for img_path in img_dir.glob('*.png'):
    mask_path = mask_dir / img_path.name
    img = Image.open(img_path)
    mask = Image.open(mask_path)

    aug_img, aug_mask = apply_sharpen(img, mask)

    aug_img.save(out_img_dir / f'{img_path.stem}_sharp.png')
    aug_mask.save(out_mask_dir / f'{mask_path.stem}_sharp.png')

print("✅ 已保存锐化增强图像与掩码到:")
print("  图像目录:", out_img_dir)
print("  掩码目录:", out_mask_dir)



