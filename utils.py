import logging
import datetime
from PIL import Image
import os
import cv2
from tqdm import tqdm
from torchvision import transforms
import clip
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, Normalize, Resize, CenterCrop, ToTensor, RandomResizedCrop, \
    RandomHorizontalFlip
from qpth.qp import QPFunction

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def _convert_image_to_rgb(image):
    return image.convert("RGB")


tfm_train_base = Compose([
    RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=BICUBIC),
    RandomHorizontalFlip(p=0.5),
    ToTensor()
]
)

tfm_test_base = Compose([
    Resize(224, interpolation=BICUBIC),
    CenterCrop(224),
    _convert_image_to_rgb,
    ToTensor(),
])


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc


def clip_classifier(classnames, clip_model, template):
    with torch.no_grad():
        clip_weights = []
        for classname in classnames:
            # Tokenize the class prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).cuda()
            # Average text embeddings
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights


def load_aux_weight(args, model, train_loader_cache):
    if args.load_aux_weight == False:
        aux_features = []
        aux_labels = []
        with torch.no_grad():
            for augment_idx in range(args.augment_epoch):
                aux_features_current = []
                print('Augment Epoch: {:} / {:}'.format(augment_idx, args.augment_epoch))
                for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                    images = images.cuda()
                    image_features = model(images)
                    aux_features_current.append(image_features)
                    if augment_idx == 0:
                        target = target.cuda()
                        aux_labels.append(target)
                aux_features.append(torch.cat(aux_features_current, dim=0).unsqueeze(0))

        aux_features = torch.cat(aux_features, dim=0).mean(dim=0).cuda()
        aux_features /= aux_features.norm(dim=-1, keepdim=True)

        aux_labels = torch.cat(aux_labels).cuda()

        torch.save(aux_features, args.cache_dir + f'/aux_feature_' + str(args.shots) + "shots.pt")
        torch.save(aux_labels, args.cache_dir + f'/aux_labels_' + str(args.shots) + "shots.pt")

    else:
        aux_features = torch.load(args.cache_dir + f'/aux_feature_' + str(args.shots) + "shots.pt")
        aux_labels = torch.load(args.cache_dir + f'/aux_labels_' + str(args.shots) + "shots.pt")
    return aux_features, aux_labels


def load_test_features(args, split, model, loader, model_name):
    if args.load_pre_feat == False:
        features, labels = [], []
        with torch.no_grad():
            for i, (images, target) in enumerate(tqdm(loader)):
                images, target = images.cuda(), target.cuda()
                if hasattr(model, 'encode_image') and callable(getattr(model, 'encode_image')):
                    image_features = model.encode_image(images)  # for clip model
                else:
                    image_features = model(images)
                features.append(image_features)
                labels.append(target)

        features, labels = torch.cat(features), torch.cat(labels)
        features = features.cuda()
        torch.save(features, args.cache_dir + f"/{model_name}_" + split + "_f.pt")
        torch.save(labels, args.cache_dir + f"/{model_name}_" + split + "_l.pt")

    else:
        features = torch.load(args.cache_dir + f"/{model_name}_" + split + "_f.pt")
        labels = torch.load(args.cache_dir + f"/{model_name}_" + split + "_l.pt")
    return features, labels


def config_logging(args):
    logger = logging.getLogger()  # root logger
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s: - %(message)s',
        datefmt='%Y-%m-%d %H:%M')
    now = datetime.datetime.now().strftime("%m-%d-%H_%M")
    # FileHandler
    fh = logging.FileHandler(f'result/{args.exp_name}_{now}.log')

    # dataset = args.dataset
    # fh = logging.FileHandler(f'result/result_{dataset}/{args.exp_name}_{now}.log')

    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    # StreamHandler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


# ======================================================
# 🔹 新增：专门用于 CLIP 特征提取的函数
# ======================================================

def load_clip_features(args, clip_model, loader):
    """
    专门提取 CLIP 图像特征

    Args:
        args: 参数配置
        clip_model: CLIP 模型
        loader: 数据加载器

    Returns:
        features: [N, D] - CLIP 图像特征
        labels: [N] - 对应标签
    """
    cache_keys = None
    cache_values = None

    with torch.no_grad():
        print(f"\n{'=' * 60}")
        print(f"Extracting CLIP image features")
        print(f"{'=' * 60}")

        for augment_idx in range(args.augment_epoch):
            features = []
            labels = []

            print(f'Augment Epoch: [{augment_idx + 1}/{args.augment_epoch}]')

            for i, (images, target) in enumerate(tqdm(loader, desc="Image feature extraction")):
                images = images.cuda()
                target = target.cuda()

                # ✅ CLIP 使用 encode_image
                image_features = clip_model.encode_image(images)

                # 归一化
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                features.append(image_features)
                labels.append(target)

            # 拼接当前 epoch 的特征
            features = torch.cat(features, dim=0)
            labels = torch.cat(labels, dim=0)

            # 累积到 cache
            if cache_keys is None:
                cache_keys = features
                cache_values = labels
            else:
                cache_keys = torch.cat([cache_keys, features], dim=0)
                cache_values = torch.cat([cache_values, labels], dim=0)

        print(f"✅ Extracted CLIP features: {cache_keys.shape}")
        print(f"   Labels: {cache_values.shape}")
        print(f"{'=' * 60}\n")

    return cache_keys, cache_values


# ======================================================
# 🔹 IGTR: Image-Guided Text Refinement
# ======================================================

def load_igtr_prompts(prompt_file, classnames):
    """
    加载 IGTR 文本 prompt 文件

    Args:
        prompt_file: IGTR 文本 prompt JSON 文件路径
        classnames: 类别名列表

    Returns:
        prompts_dict: {classname: [prompt1, prompt2, ...]}
    """
    import json

    if not os.path.exists(prompt_file):
        print(f"⚠️  IGTR prompt file not found: {prompt_file}")
        return None

    with open(prompt_file, 'r') as f:
        prompts = json.load(f)

    # 只保留当前数据集的类别
    prompts_dict = {}
    for classname in classnames:
        if classname in prompts:
            prompts_dict[classname] = prompts[classname]
        else:
            print(f"⚠️  No IGTR prompts for class: {classname}")

    print(f"✅ Loaded IGTR prompts for {len(prompts_dict)} classes")
    return prompts_dict


def extract_all_text_features(classnames, clip_model, template, igtr_prompts=None):
    """
    提取 IGTR 所需的所有文本 prompt 特征

    Args:
        classnames: 类别名列表
        clip_model: CLIP 模型
        template: 默认模板
        igtr_prompts: IGTR prompts 字典（可选）

    Returns:
        text_features_all: [N, P, D] - 所有类别的 prompt 特征
    """
    import clip

    with torch.no_grad():
        all_features = []
        min_prompts = float('inf')

        for classname in tqdm(classnames, desc="Extracting text features"):
            classname_clean = classname.replace('_', ' ')

            # 基础模板
            texts = [t.format(classname_clean) for t in template]

            # 加载 IGTR 文本 prompts
            if igtr_prompts and classname in igtr_prompts:
                texts += igtr_prompts[classname]

            # 编码
            texts_token = clip.tokenize(texts, truncate=True).cuda()
            class_embeddings = clip_model.encode_text(texts_token)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

            min_prompts = min(min_prompts, len(class_embeddings))
            all_features.append(class_embeddings)

        # 统一长度（取最短）
        for i in range(len(all_features)):
            all_features[i] = all_features[i][:min_prompts]

        text_features_all = torch.stack(all_features, dim=0)  # [N, P, D]

    print(f"✅ Extracted text features: {text_features_all.shape}")
    print(f"   Classes: {text_features_all.shape[0]}")
    print(f"   Prompts per class: {text_features_all.shape[1]}")

    return text_features_all


def compute_image_prototypes(image_features, labels, num_classes):
    """
    计算每类的图像原型（均值）

    Args:
        image_features: [M, D] - 图像特征
        labels: [M] - 标签
        num_classes: 类别数

    Returns:
        prototypes: [N, D] - 每类的原型
    """
    prototypes = []

    for i in range(num_classes):
        mask = labels == i
        class_features = image_features[mask]

        if len(class_features) == 0:
            raise ValueError(f"Class {i} has no samples!")

        prototype = class_features.mean(dim=0)
        prototypes.append(prototype)

    prototypes = torch.stack(prototypes, dim=0)
    prototypes = prototypes / prototypes.norm(dim=-1, keepdim=True)

    return prototypes


def igtr_optimize_prompts(text_features_all, image_prototypes, gamma=50):
    """
    IGTR: Image-Guided Text Refinement
    用图像原型指导 prompt 选择和加权

    Args:
        text_features_all: [N, P, D] - 所有 prompts 特征
        image_prototypes: [N, D] - 图像原型（每类的均值特征）
        gamma: 权重缩放参数

    Returns:
        optimized_text_features: [N, D] - 优化后的文本特征
        weights: [N, P] - 每个 prompt 的权重
    """
    N, P, D = text_features_all.shape

    print(f"\n{'=' * 60}")
    print(f"IGTR Prompt Optimization")
    print(f"{'=' * 60}")

    with torch.no_grad():
        # 步骤1: 计算每个 prompt 与图像原型的相似度
        image_prototypes_expanded = image_prototypes.unsqueeze(1)  # [N, 1, D]

        # [N, P, D] @ [N, D, 1] → [N, P]
        similarity = torch.bmm(
            text_features_all,
            image_prototypes_expanded.transpose(1, 2)
        ).squeeze(-1)

        # 步骤2: 归一化并缩放
        similarity_norm = similarity.norm(dim=-1, keepdim=True)
        scores = gamma * similarity / (similarity_norm + 1e-8)

        # 步骤3: Softmax 得到权重
        weights = F.softmax(scores, dim=-1)  # [N, P]

        # 步骤4: 加权组合所有 prompts
        optimized_text_features = torch.einsum('npd,np->nd', text_features_all, weights)

        # 归一化
        optimized_text_features = optimized_text_features / optimized_text_features.norm(dim=-1, keepdim=True)

    print(f"✅ IGTR optimization completed")
    print(f"   Input: {text_features_all.shape}")
    print(f"   Output: {optimized_text_features.shape}")
    print(f"   Weights shape: {weights.shape}")

    # 打印一些统计信息
    print(f"\n📊 Prompt weight statistics:")
    print(f"   Mean weight: {weights.mean().item():.4f}")
    print(f"   Max weight: {weights.max().item():.4f}")
    print(f"   Min weight: {weights.min().item():.4f}")

    # 显示每类权重最高的 prompt 索引
    top_prompts = weights.argmax(dim=-1)
    print(f"\n📋 Top prompt indices per class (first 5):")
    print(f"   {top_prompts[:5].cpu().numpy()}")
    print(f"{'=' * 60}\n")

    return optimized_text_features, weights

tfm_train_base = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                         std=(0.26862954, 0.26130258, 0.27577711))
])

tfm_test_base = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225))
])

# CLIP 分支使用 CLIP 官方标准化参数
tfm_clip = Compose([
    Resize(size=224, interpolation=InterpolationMode.BICUBIC, antialias=True),
    CenterCrop(size=(224, 224)),
    ToTensor(),
    Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
              std=(0.26862954, 0.26130258, 0.27577711))
])

# 辅助视觉分支使用通用图像标准化参数
tfm_aux = Compose([
    Resize(size=224, interpolation=InterpolationMode.BICUBIC, antialias=True),
    CenterCrop(size=(224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225])
])


"""
CVTR - IGVE Module In Test Phase

"""


def patch_level_optimal_matching_qpth(distance_matrix, weight_query, weight_support,
                                      form='QP', l2_strength=0.0001):
    """
    Patch-Level Optimal Matching (QPTH Solver)

    Args:
        distance_matrix: [B, HW, HW]  patch-level distance
        weight_query: [B, HW]
        weight_support: [B, HW]

    Returns:
        matching_score: [B]
        flow: [B, HW, HW]  ->  {w}_{ij}
    """

    weight_query = (weight_query * weight_query.shape[-1]) / weight_query.sum(1).unsqueeze(1)
    weight_support = (weight_support * weight_support.shape[-1]) / weight_support.sum(1).unsqueeze(1)

    nbatch = distance_matrix.shape[0]
    nelement = distance_matrix.shape[1] * distance_matrix.shape[2]

    nq = weight_query.shape[1]
    ns = weight_support.shape[1]

    Q_1 = distance_matrix.view(-1, 1, nelement).double()

    if form == 'QP':
        Q = torch.bmm(Q_1.transpose(2, 1), Q_1).double().cuda() + \
            1e-4 * torch.eye(nelement).double().cuda().unsqueeze(0).repeat(nbatch, 1, 1)
        p = torch.zeros(nbatch, nelement).double().cuda()
    elif form == 'L2':
        Q = (l2_strength * torch.eye(nelement).double()).cuda().unsqueeze(0).repeat(nbatch, 1, 1)
        p = distance_matrix.view(nbatch, nelement).double()
    else:
        raise ValueError('Unknown form')

    h_1 = torch.zeros(nbatch, nelement).double().cuda()
    h_2 = torch.cat([weight_query, weight_support], 1).double()
    h = torch.cat((h_1, h_2), 1)

    G_1 = -torch.eye(nelement).double().cuda().unsqueeze(0).repeat(nbatch, 1, 1)
    G_2 = torch.zeros([nbatch, nq + ns, nelement]).double().cuda()

    # sum_j w_ij = s_i
    for i in range(nq):
        G_2[:, i, ns * i:ns * (i + 1)] = 1

    # sum_i w_ij = d_j
    for j in range(ns):
        G_2[:, nq + j, j::ns] = 1

    G = torch.cat((G_1, G_2), 1)

    A = torch.ones(nbatch, 1, nelement).double().cuda()
    b = torch.min(torch.sum(weight_query, 1), torch.sum(weight_support, 1)).unsqueeze(1).double()

    flow = QPFunction(verbose=-1)(Q, p, G, h, A, b)

    matching_score = torch.sum((1 - Q_1).squeeze() * flow, 1)

    return matching_score, flow.view(-1, nq, ns)


def patch_level_optimal_matching_opencv(cost_matrix, weight_query, weight_support):
    """
    Patch-Level Optimal Matching (OpenCV version)
    """

    cost_matrix = cost_matrix.detach().cpu().numpy()

    weight_query = F.relu(weight_query) + 1e-5
    weight_support = F.relu(weight_support) + 1e-5

    weight_query = (weight_query * (weight_query.shape[0] / weight_query.sum().item())).view(-1, 1).cpu().numpy()
    weight_support = (weight_support * (weight_support.shape[0] / weight_support.sum().item())).view(-1, 1).cpu().numpy()

    cost, _, flow = cv2.EMD(weight_query, weight_support, cv2.DIST_USER, cost_matrix)

    return cost, flow


def patch_level_optimal_matching_batch(distance_matrix, weight_query, weight_support):
    """
    Batch version of Patch-Level Optimal Matching (OpenCV)
    """

    distance_list = []
    flow_list = []

    for i in range(distance_matrix.shape[0]):
        cost, flow = patch_level_optimal_matching_opencv(
            distance_matrix[i], weight_query[i], weight_support[i]
        )
        distance_list.append(cost)
        flow_list.append(torch.from_numpy(flow))

    matching_score = torch.Tensor(distance_list).cuda().double()
    flow = torch.stack(flow_list, dim=0).cuda().double()

    return matching_score, flow