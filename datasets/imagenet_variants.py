import os
from .utils import Datum, DatasetBase, listdir_nohidden
from .imagenet import load_wnid_to_label_mapping  # 复用你写的

template = ["a photo of a {}."]

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPEG")


def _default_classnames_file(root, imagenet_images_root=None):
    """
    你 ImageNet 的结构：
    root/datasets/ImageNet2012/classnames.txt
    root/datasets/ImageNet2012/images/train|val/wnid/*.JPEG
    """
    if imagenet_images_root is None:
        imagenet_images_root = os.path.join(root, "datasets/ImageNet2012/images")
    imagenet_root = os.path.dirname(imagenet_images_root)  # -> .../ImageNet2012
    classnames_file = os.path.join(imagenet_root, "classnames.txt")
    if not os.path.exists(classnames_file):
        raise RuntimeError(f"Missing classnames.txt: {classnames_file}")
    return classnames_file


def _read_imagenet_style_folder(folder_root, wnid_to_label):
    """
    folder_root:
      - .../xxx/wnid/*.jpg
    label:
      - 直接用 ImageNet-1k 的 0..999 label（来自 classnames.txt）
    """
    items = []
    wnids = listdir_nohidden(folder_root, sort=True)

    for wnid in wnids:
        if wnid not in wnid_to_label:
            continue
        label = wnid_to_label[wnid]
        class_dir = os.path.join(folder_root, wnid)

        for fname in listdir_nohidden(class_dir, sort=False):
            if not fname.lower().endswith(tuple(e.lower() for e in IMG_EXTS)):
                continue
            impath = os.path.join(class_dir, fname)
            items.append(Datum(impath=impath, label=label, classname=wnid))

    if len(items) == 0:
        raise RuntimeError(f"No images found under: {folder_root}")
    return items


def _read_imagenet_sub200_folder(folder_root, wnid_to_label):
    """
    ImageNet-A / ImageNet-R 常用评估：200-way 分类
    - 数据集 target: 0..199（按 wnid 文件夹排序的顺序）
    - 同时返回 mask_1000: 长度 1000 的 index 列表，用于 logits[:, mask]
    """
    items = []
    wnids = listdir_nohidden(folder_root, sort=True)

    mask_1000 = []
    wnids_kept = []

    for wnid in wnids:
        if wnid in wnid_to_label:
            wnids_kept.append(wnid)
            mask_1000.append(wnid_to_label[wnid])

    # target(0..199) 的映射：按 wnids_kept 顺序
    wnid_to_sub_label = {wnid: i for i, wnid in enumerate(wnids_kept)}

    for wnid in wnids_kept:
        sub_label = wnid_to_sub_label[wnid]
        class_dir = os.path.join(folder_root, wnid)

        for fname in listdir_nohidden(class_dir, sort=False):
            if not fname.lower().endswith(tuple(e.lower() for e in IMG_EXTS)):
                continue
            impath = os.path.join(class_dir, fname)
            # classname 仍然写 wnid，方便 debug
            items.append(Datum(impath=impath, label=sub_label, classname=wnid))

    if len(items) == 0:
        raise RuntimeError(f"No images found under: {folder_root}")

    return items, mask_1000, wnids_kept


class _ImageNetVariantBase(DatasetBase):
    """给四个变体共用：加载 classnames.txt / wnid_to_label"""

    dataset_dir = None  # 子类覆盖
    template = template

    def __init__(self, root, num_shots, imagenet_images_root=None):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.template = template

        classnames_file = _default_classnames_file(root, imagenet_images_root)
        self.wnid_to_label, self.classnames = load_wnid_to_label_mapping(classnames_file)


class ImageNetSketch(_ImageNetVariantBase):
    dataset_dir = "datasets/sketch"  # root/datasets/sketch/wnid/*.jpg

    def __init__(self, root, num_shots, imagenet_images_root=None):
        super().__init__(root, num_shots, imagenet_images_root)
        test = _read_imagenet_style_folder(self.dataset_dir, self.wnid_to_label)
        # 你的 DatasetBase 不接受 train_x=None：用 test 顶一下
        super(DatasetBase, self).__init__()
        DatasetBase.__init__(self, train_x=test, val=None, test=test)


class ImageNetV2(_ImageNetVariantBase):
    """
    ImageNet-V2 是 1000 类（同 ImageNet-1k 标签空间）
    目录常见两种：
      - root/datasets/imagenetv2/val/wnid/*.jpg
      - root/datasets/imagenetv2/wnid/*.jpg
    """
    dataset_dir = "datasets/imagenetv2"

    def __init__(self, root, num_shots, imagenet_images_root=None):
        super().__init__(root, num_shots, imagenet_images_root)

        cand1 = os.path.join(self.dataset_dir, "val")
        folder_root = cand1 if os.path.exists(cand1) else self.dataset_dir

        test = _read_imagenet_style_folder(folder_root, self.wnid_to_label)
        DatasetBase.__init__(self, train_x=test, val=None, test=test)


class ImageNetA(_ImageNetVariantBase):
    """
    ImageNet-A：200 类子集（从 ImageNet-1k wnid 中选的）
    默认做 200-way：logits_200 = logits_1000[:, mask_1000]
    """
    dataset_dir = "datasets/imagenet-a"

    def __init__(self, root, num_shots, imagenet_images_root=None):
        super().__init__(root, num_shots, imagenet_images_root)
        test, mask_1000, wnids_kept = _read_imagenet_sub200_folder(self.dataset_dir, self.wnid_to_label)

        self.mask_1000 = mask_1000          # list[int], len=200
        self.sub_wnids = wnids_kept         # list[str], len=200

        DatasetBase.__init__(self, train_x=test, val=None, test=test)


class ImageNetR(_ImageNetVariantBase):
    """
    ImageNet-R：200 类子集（同样建议按 200-way）
    """
    dataset_dir = "datasets/imagenet-r"

    def __init__(self, root, num_shots, imagenet_images_root=None):
        super().__init__(root, num_shots, imagenet_images_root)
        test, mask_1000, wnids_kept = _read_imagenet_sub200_folder(self.dataset_dir, self.wnid_to_label)

        self.mask_1000 = mask_1000
        self.sub_wnids = wnids_kept

        DatasetBase.__init__(self, train_x=test, val=None, test=test)
