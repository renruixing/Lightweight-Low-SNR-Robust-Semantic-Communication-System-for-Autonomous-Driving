"""
common/dataset.py
==================
Cityscapes扁平文件夹加载器。你本地的cityscapes数据集是
    <root>/train/1.jpg, 2.jpg, ...
    <root>/val/1.jpg, 2.jpg, ...
这种扁平结构（不是官方leftImg8bit/gtFine那套），所以torchvision自带的
Cityscapes类读不了。这里独立实现一份。
"""
import os
from PIL import Image
from torch.utils.data import Dataset


class FlatImageFolderDataset(Dataset):
    def __init__(self, root, split, transform=None):
        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(
                f"找不到目录：{split_dir}\n"
                f"请检查 DATASET_PATH 是否填到了 cityscapes 这一级"
                f"（里面应直接包含 train/ 和 val/ 两个子文件夹）。"
            )
        self.files = sorted(
            os.path.join(split_dir, f) for f in os.listdir(split_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        if not self.files:
            raise FileNotFoundError(f"{split_dir} 下没有找到任何jpg/png图像。")
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img
