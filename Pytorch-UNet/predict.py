import argparse
import logging
import os
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask

# -------------------- core inference --------------------
def predict_img(net, full_img, device, scale_factor=1, out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(
        BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False)
    )
    img = img.unsqueeze(0).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(
            output, (full_img.size[1], full_img.size[0]), mode="bilinear", align_corners=False
        )
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = (torch.sigmoid(output) > out_threshold)
    return mask[0].long().squeeze().numpy()


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)
    for i, v in enumerate(mask_values):
        out[mask == i] = v
    return Image.fromarray(out)


# -------------------- args --------------------
def get_args():
    parser = argparse.ArgumentParser(description="Predict masks from input images or folder")

    # ✅ 默认 checkpoint 路径
    default_ckpt = "checkpoints/crop_aug/checkpoint_epoch20.pth"
    # ✅ 默认输入图片路径（你可以改成 Domain3、Domain4 等）
    default_input_dir = "data/Domain5/test/imgs"

    parser.add_argument("--model", "-m", default=default_ckpt, metavar="FILE",
                        help=f"Path to model checkpoint (default: {default_ckpt})")

    # ✅ 在这里指定默认输入目录
    parser.add_argument("--input", "-i", nargs="+", default=[default_input_dir],
                        help=f"Input image(s) or folder path (default: {default_input_dir})")

    parser.add_argument("--output", "-o", nargs="+",
                        help="Output mask files. If omitted, masks saved to <input_dir>_out/")
    parser.add_argument("--viz", "-v", action="store_true", help="Visualize predictions")
    parser.add_argument("--no-save", "-n", action="store_true", help="Do not save output masks")
    parser.add_argument("--mask-threshold", "-t", type=float, default=0.5,
                        help="Threshold for binary masks when n_classes == 1")
    parser.add_argument("--scale", "-s", type=float, default=1,
                        help="Scale factor for input images (default: 1.0)")
    parser.add_argument("--bilinear", action="store_true", default=False,
                        help="Use bilinear upsampling in UNet")
    parser.add_argument("--classes", "-c", type=int, default=2,
                        help="Number of output classes (default: 2)")

    args = parser.parse_args()
    return args



# -------------------- helpers --------------------
_IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

def _list_images_in_dir(folder):
    files = []
    for ext in _IMG_EXTS:
        files.extend(glob(os.path.join(folder, f"*{ext}")))
    return sorted(files)

def expand_inputs(input_args):
    expanded = []
    for item in input_args:
        if os.path.isdir(item):
            expanded.extend(_list_images_in_dir(item))
        else:
            expanded.append(item)
    expanded = sorted(list(dict.fromkeys(expanded)))
    return expanded

def make_output_paths_for_inputs(in_files, explicit_outputs=None):
    if explicit_outputs:
        if len(explicit_outputs) != len(in_files):
            raise ValueError("Number of --output files must match number of input files")
        return explicit_outputs

    out_files = []
    for f in in_files:
        in_dir = os.path.dirname(f)
        out_dir = in_dir.rstrip("/\\") + "_out"
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.basename(f)
        name, _ = os.path.splitext(base)
        out_files.append(os.path.join(out_dir, f"{name}_OUT.png"))
    return out_files


# -------------------- main --------------------
if __name__ == "__main__":
    args = get_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    in_files = expand_inputs(args.input)
    if len(in_files) == 0:
        raise FileNotFoundError("No images found under input path(s).")

    out_files = make_output_paths_for_inputs(in_files, args.output)

    # ✅ 打印模型路径
    logging.info(f"Using model checkpoint: {args.model}")

    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device=device)

    # ✅ 加载checkpoint
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Checkpoint not found: {args.model}")

    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop("mask_values", [0, 1])
    net.load_state_dict(state_dict)
    logging.info("Model loaded successfully!")

    for i, filename in enumerate(in_files):
        logging.info(f"[{i+1}/{len(in_files)}] Predicting: {filename}")
        img = Image.open(filename).convert("RGB")
        mask = predict_img(
            net=net,
            full_img=img,
            scale_factor=args.scale,
            out_threshold=args.mask_threshold,
            device=device,
        )
        if not args.no_save:
            out_path = out_files[i]
            result = mask_to_image(mask, mask_values)
            result.save(out_path)
            logging.info(f"Saved mask -> {out_path}")

        if args.viz:
            plot_img_and_mask(img, mask)

    logging.info("✅ All predictions done.")
