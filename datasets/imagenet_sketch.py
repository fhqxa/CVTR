import os
from .utils import Datum, DatasetBase, listdir_nohidden

try:
    from .imagenet import load_wnid_to_label_mapping
except ImportError:
    def load_wnid_to_label_mapping(classnames_file):
        wnid_to_label = {}
        classnames = []
        with open(classnames_file, 'r') as f:
            for idx, line in enumerate(f):
                parts = line.strip().split(maxsplit=1)
                if len(parts) >= 2:
                    wnid = parts[0]
                    classname = parts[1]
                    wnid_to_label[wnid] = idx
                    classnames.append(classname)
        return wnid_to_label, classnames

template = ["a photo of a {}."]


class ImageNetSketch(DatasetBase):
    """
    ImageNet-Sketch 评测集（按 1000-way）
    目录结构要求:
      root/datasets/sketch/<wnid>/*.jpg
    label 直接使用 ImageNet1000 的 label (0..999)
    """
    dataset_dir = "datasets/sketch"

    def __init__(self, root, num_shots, imagenet_images_root=None):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.template = template

        if imagenet_images_root is None:
            imagenet_images_root = os.path.join(root, "datasets/ImageNet2012/images")

        imagenet_root = os.path.dirname(imagenet_images_root)
        classnames_file = os.path.join(imagenet_root, "classnames.txt")

        if not os.path.exists(classnames_file):
            raise RuntimeError(f"Missing classnames.txt: {classnames_file}")

        print("=" * 60)
        print("Loading ImageNet-Sketch dataset...")
        print("=" * 60)

        # 加载 wnid 到 label 的映射和类名列表
        wnid_to_label_1000, classnames_list = load_wnid_to_label_mapping(classnames_file)

        # 读取测试集数据
        test = self._read_imagenet_style_folder(self.dataset_dir, wnid_to_label_1000, classnames_list)

        # ✅ 在 super().__init__() 之前设置 _classnames
        self._classnames = classnames_list

        # 调用父类初始化
        super().__init__(train_x=test, val=None, test=test)

        print("=" * 60)
        print()

    def _read_imagenet_style_folder(self, folder_root, wnid_to_label_1000, classnames_list):
        """
        读取文件夹，返回 Datum 列表
        """
        if not os.path.exists(folder_root):
            raise RuntimeError(f"ImageNet-Sketch folder not found: {folder_root}")

        items = []
        wnids = listdir_nohidden(folder_root, sort=True)

        # 统计类别数
        valid_classes = 0

        for wnid in wnids:
            if wnid not in wnid_to_label_1000:
                continue

            valid_classes += 1
            label = wnid_to_label_1000[wnid]
            class_dir = os.path.join(folder_root, wnid)

            for fname in listdir_nohidden(class_dir, sort=False):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                    continue
                impath = os.path.join(class_dir, fname)
                classname = classnames_list[label] if label < len(classnames_list) else ""
                items.append(Datum(impath=impath, label=label, classname=classname))

        if len(items) == 0:
            raise RuntimeError(f"No images found under: {folder_root}")

        # ✅ 打印加载信息
        print(f"加载 test (ImageNet-Sketch) 数据集，共 {valid_classes} 个类别...")
        print(f"test 数据集加载完成: {len(items)} 张图像")

        return items