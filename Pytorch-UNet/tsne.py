import os
import glob
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import models, transforms

from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

import ssl
import certifi

ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())


# -----------------------------
# 1. 自定义 Dataset：读取多域图片
# -----------------------------
class MultiDomainDataset(Dataset):
    def __init__(self, domain_dirs, transform=None):
        """
        domain_dirs: list of (domain_name, img_dir)
            e.g. [("Domain1", "data/Domain1/train/imgs"), ...]
        """
        self.samples = []
        self.domain_names = []
        self.transform = transform

        for domain_id, (domain_name, img_dir) in enumerate(domain_dirs):
            self.domain_names.append(domain_name)

            # 支持 jpg/jpeg/png，可按需扩展
            img_paths = []
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
                img_paths.extend(glob.glob(os.path.join(img_dir, ext)))

            for p in img_paths:
                self.samples.append((p, domain_id))

        print(f"Total images: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, domain_id = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, domain_id


# -----------------------------
# 2. 构建特征提取网络（ResNet18, 去掉分类层）
# -----------------------------
def build_feature_extractor(device):
    # 使用 ImageNet 预训练权重
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # 去掉最后的 fc 层，保留到全局池化，输出维度 512
    backbone = nn.Sequential(*list(model.children())[:-1])  # (N, 512, 1, 1)

    backbone.to(device)
    backbone.eval()
    return backbone


# -----------------------------
# 3. 提取所有图片特征
# -----------------------------
@torch.no_grad()
def extract_features(dataloader, model, device):
    all_feats = []
    all_labels = []

    for imgs, domain_ids in dataloader:
        imgs = imgs.to(device)
        # 输出 (B, 512, 1, 1) -> reshape 成 (B, 512)
        feats = model(imgs).view(imgs.size(0), -1)

        all_feats.append(feats.cpu().numpy())
        all_labels.append(domain_ids.numpy())

    all_feats = np.concatenate(all_feats, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return all_feats, all_labels


# -----------------------------
# 4. 运行 t-SNE 并画图
# -----------------------------
def run_tsne_and_plot(feats, labels, domain_names, save_path=None):
    print("Running t-SNE...")
    tsne = TSNE(
        n_components=2,
        perplexity=30,     # 可以根据样本数调整
        n_iter=1000,
        init='pca',
        random_state=42,
        learning_rate='auto'
    )
    feats_2d = tsne.fit_transform(feats)

    # 画图
    plt.figure(figsize=(8, 6))
    num_domains = len(domain_names)

    for d in range(num_domains):
        mask = (labels == d)
        plt.scatter(
            feats_2d[mask, 0],
            feats_2d[mask, 1],
            s=8,
            alpha=0.6,
            label=domain_names[d]
        )

    plt.legend()
    plt.title("t-SNE of Multi-Domain Image Features")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


# -----------------------------
# 5. 主函数
# -----------------------------
def main():
    # 你的数据路径设置
    domain_dirs = [
        ("Domain1", "data/Domain1/train/imgs"),
        ("Domain2", "data/Domain2/train/imgs"),
        ("Domain3", "data/Domain3/train/imgs"),
        ("Domain4", "data/Domain4/train/imgs"),
        ("Domain5", "data/Domain5/train/imgs"),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 图像预处理：与 ImageNet 一致
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = MultiDomainDataset(domain_dirs, transform=transform)
    # 如果图片非常多，可以先每个域随机采样一部分，或者减小 batch_size
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    # 特征提取器
    feature_extractor = build_feature_extractor(device)

    # 提取特征
    feats, labels = extract_features(dataloader, feature_extractor, device)
    print("Feature shape:", feats.shape)  # (N, 512)

    # 也可以在这里随机下采样一部分点再做 t-SNE，防止太慢
    # 例如每个域最多取 1000 张：
    # feats, labels = subsample_per_domain(feats, labels, max_per_domain=1000)

    # t-SNE + 可视化
    domain_names = [d[0] for d in domain_dirs]
    run_tsne_and_plot(feats, labels, domain_names, save_path="tsne_domains.png")


if __name__ == "__main__":
    main()
