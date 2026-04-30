import os
import random
from collections import defaultdict
from PIL import Image
import torchvision.transforms as T
import torch
from torch.utils.data import Dataset

from utils import tfm_train_base, tfm_test_base

imagenet_templates = ["itap of a {}.",
                      "a bad photo of the {}.",
                      "a origami {}.",
                      "a photo of the large {}.",
                      "a {} in a video game.",
                      "art of the {}.",
                      "a photo of the small {}."]


def load_wnid_to_label_mapping(classnames_file):
    """
    从classnames.txt加载wnid到标签索引的映射，同时返回类名列表

    Args:
        classnames_file: classnames.txt文件路径

    Returns:
        wnid_to_label: dict, {wnid: label_index}
        classnames: list, 类名列表（按索引顺序）
    """
    wnid_to_label = {}
    classnames = []
    with open(classnames_file, 'r') as f:
        for idx, line in enumerate(f):
            parts = line.strip().split(maxsplit=1)  # 只分割一次，保留类名中的空格
            if len(parts) >= 2:
                wnid = parts[0]  # 如 n01440764
                classname = parts[1]  # 如 tench
                wnid_to_label[wnid] = idx
                classnames.append(classname)
    return wnid_to_label, classnames


class ImageNetDataset(Dataset):
    """自定义ImageNet数据集加载器，支持wnid到label的映射"""

    def __init__(self, root, split='train', transform=None, wnid_to_label=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.wnid_to_label = wnid_to_label

        # 存储图像路径和标签
        self.samples = []
        self.targets = []
        self.imgs = []

        split_dir = os.path.join(root, split)

        if not os.path.exists(split_dir):
            raise RuntimeError(f"数据集路径不存在: {split_dir}\n请确保ImageNet数据集在此路径下")

        # 获取所有类别文件夹（按字母顺序排序）
        class_folders = sorted([d for d in os.listdir(split_dir)
                                if os.path.isdir(os.path.join(split_dir, d))])

        if len(class_folders) == 0:
            raise RuntimeError(f"在 {split_dir} 中没有找到类别文件夹")

        print(f"加载 {split} 数据集，共 {len(class_folders)} 个类别...")

        # 遍历每个类别文件夹
        for folder_idx, wnid in enumerate(class_folders):
            class_path = os.path.join(split_dir, wnid)

            # 使用映射文件确定真实的类别索引
            if self.wnid_to_label is not None:
                if wnid in self.wnid_to_label:
                    class_idx = self.wnid_to_label[wnid]
                else:
                    print(f"警告: wnid {wnid} 在映射文件中未找到，跳过")
                    continue
            else:
                # 如果没有映射，使用文件夹顺序
                class_idx = folder_idx

            # 获取该类别下的所有图像文件
            img_files = [f for f in os.listdir(class_path)
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.JPEG'))]

            for img_name in img_files:
                img_path = os.path.join(class_path, img_name)
                self.samples.append(img_path)
                self.targets.append(class_idx)
                self.imgs.append((img_path, class_idx))

        print(f"{split} 数据集加载完成: {len(self.samples)} 张图像\n")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        target = self.targets[idx]

        # 加载图像
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"警告: 无法加载图像 {img_path}: {e}")
            image = Image.new('RGB', (224, 224), color=(0, 0, 0))

        if self.transform is not None:
            image = self.transform(image)

        return image, target


class ImageNet():
    dataset_dir = ''

    def __init__(self, root, num_shots, external_test_dataset=None):
        """
        初始化 ImageNet 数据集

        Args:
            root: 数据集根目录
            num_shots: few-shot 采样数量
            external_test_dataset: 可选的外部测试集（用于替换 ImageNet val）
        """
        # root 可以是:
        # 1. 项目根目录: /home/chy/chy/AMU-Tuning/AMU-Tuning
        # 2. images 目录: /home/chy/chy/AMU-Tuning/AMU-Tuning/datasets/ImageNet2012/images

        # 检测并处理路径
        if os.path.exists(os.path.join(root, 'train')) and os.path.exists(os.path.join(root, 'val')):
            # 情况2: 直接传入了 images 目录
            self.image_dir = root
            imagenet_root = os.path.dirname(root)
        else:
            # 情况1: 传入了项目根目录
            self.image_dir = os.path.join(root, "datasets/ImageNet2012/images")
            imagenet_root = os.path.join(root, "datasets/ImageNet2012")

            if not os.path.exists(self.image_dir):
                raise RuntimeError(
                    f"找不到 ImageNet 数据集！\n"
                    f"期望路径: {self.image_dir}\n"
                    f"请确保数据集结构为: root/datasets/ImageNet2012/images/train 和 .../val"
                )

        # classnames.txt 在 ImageNet2012 目录下
        classnames_file = os.path.join(imagenet_root, 'classnames.txt')

        if not os.path.exists(classnames_file):
            raise RuntimeError(f"找不到classnames.txt文件: {classnames_file}\n"
                               f"请确保该文件存在")

        print(f"加载类别映射文件: {classnames_file}")
        wnid_to_label, self.classnames = load_wnid_to_label_mapping(classnames_file)
        print(f"成功加载 {len(wnid_to_label)} 个类别映射")
        print(f"前5个类别: {self.classnames[:5]}\n")

        train_transform = T.Compose([
            T.RandomResizedCrop(224, scale=(0.5, 1.0), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(0.5),
            T.ToTensor(),
        ])
        test_transform = T.Compose([
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
        ])

        # 获取数据增强
        train_preprocess = train_transform
        test_preprocess = test_transform

        print(f"ImageNet 数据集根目录: {self.image_dir}")

        # 加载数据集（传入映射）
        print("=" * 60)
        self.train = ImageNetDataset(self.image_dir, split='train',
                                     transform=train_preprocess,
                                     wnid_to_label=wnid_to_label)
        self.val = ImageNetDataset(self.image_dir, split='val',
                                   transform=test_preprocess,
                                   wnid_to_label=wnid_to_label)

        # ✅ 关键修改: 允许使用外部测试集替代 ImageNet val
        if external_test_dataset is not None:
            print("⚠️  使用外部测试集替代 ImageNet val")
            self.test = external_test_dataset
        else:
            self.test = ImageNetDataset(self.image_dir, split='val',
                                        transform=test_preprocess,
                                        wnid_to_label=wnid_to_label)
        print("=" * 60)

        # 设置模板
        self.template = imagenet_templates

        # Few-shot 采样
        print(f"开始进行 {num_shots}-shot 采样...")
        split_by_label_dict = defaultdict(list)

        # 按标签分组
        for i in range(len(self.train.imgs)):
            split_by_label_dict[self.train.targets[i]].append(self.train.imgs[i])

        imgs = []
        targets = []

        # 从每个类别中随机采样
        for label, items in split_by_label_dict.items():
            num_to_sample = min(num_shots, len(items))
            sampled_items = random.sample(items, num_to_sample)
            imgs.extend(sampled_items)
            targets.extend([label] * num_to_sample)

        # 更新训练集
        self.train.imgs = imgs
        self.train.targets = targets
        self.train.samples = [item[0] for item in imgs]

        print(f"Few-shot 采样完成!")
        print(f"  - 每类采样: {num_shots} 张")
        print(f"  - 类别数量: {len(split_by_label_dict)}")
        print(f"  - 训练样本总数: {len(imgs)}")
        print("=" * 60 + "\n")


if __name__ == '__main__':
    # 测试代码
    classnames_file = '/home/chy/chy/AMU-Tuning/AMU-Tuning/datasets/ImageNet2012/classnames.txt'
    if os.path.exists(classnames_file):
        wnid_to_label, classnames = load_wnid_to_label_mapping(classnames_file)
        print(f"成功加载 {len(classnames)} 个类别")
        print(f"前10个类别: {classnames[:10]}")
        print(f"'tench' in classnames: {'tench' in classnames}")