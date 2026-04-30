"""
train_cvtr_imagenet.py

Single-run training and evaluation for CVTR on ImageNet.
This script uses one fixed configuration for training and evaluation.

Main components:
- IGTR: loads text prompts, extracts prompt features, and refines text features with visual prototypes.
- IGVE: uses the auxiliary visual branch and applies Patch-Level Optimal Matching when patch-level features are available; otherwise it falls back to pooled-feature cosine retrieval.
- CVTR: combines the refined textual branch and auxiliary visual branch for few-shot classification.
"""

import os
import random
import datetime
import time
import csv

import torch
from tqdm import tqdm

from parse_args import parse_args
from datasets_split.imagenet import ImageNet
import clip
from utils import (
    config_logging,
    cls_acc,
    load_test_features,
    load_aux_weight,
    load_clip_features,
    load_igtr_prompts,
    extract_all_text_features,
    compute_image_prototypes,
    igtr_optimize_prompts,
)
from clip.aux_encoder import load_aux_encoder
from clip.CVTR_Model import CVTR_Model, tfm_clip, tfm_aux


# ======================================================
# Fixed CVTR configuration
# ======================================================
DEFAULT_CVTR_CONFIG = {
    1: {
        "alpha": 0.2,
        "lambda_merge": 0.5,
        "uncent_power": 0.6,
        "uncent_type": "moment",
        "gamma": 20,
        "kurtosis_threshold": 2.1,
        "residual_init": 0.7,
        "igve_topk": 2,
        "use_coarse_filter": True,
        "top_classes": 5,
        "fusion_type": "mlp_direct",
        "use_patch_level_matching": False,
        "matching_solver": "opencv",
    },
    2: {
        "alpha": 0.2,
        "lambda_merge": 0.2,
        "uncent_power": 0.6,
        "uncent_type": "entropy",
        "gamma": 20,
        "kurtosis_threshold": 2.1,
        "residual_init": 0.3,
        "igve_topk": 6,
        "use_coarse_filter": True,
        "top_classes": 5,
        "fusion_type": "mlp_direct",
        "use_patch_level_matching": False,
        "matching_solver": "opencv",
    },
    4: {
        "alpha": 0.5,
        "lambda_merge": 0.4,
        "uncent_power": 0.8,
        "uncent_type": "moment",
        "gamma": 20,
        "kurtosis_threshold": 2.1,
        "residual_init": 0.5,
        "igve_topk": 6,
        "use_coarse_filter": True,
        "top_classes": 5,
        "fusion_type": "mlp_direct",
        "use_patch_level_matching": False,
        "matching_solver": "opencv",
    },
    8: {
        "alpha": 0.5,
        "lambda_merge": 0.4,
        "uncent_power": 0.4,
        "uncent_type": "entropy",
        "gamma": 20,
        "kurtosis_threshold": 2.3,
        "residual_init": 0.7,
        "igve_topk": 6,
        "use_coarse_filter": True,
        "top_classes": 5,
        "fusion_type": "mlp_direct",
        "use_patch_level_matching": False,
        "matching_solver": "opencv",
    },
    16: {
        "alpha": 0.5,
        "lambda_merge": 0.2,
        "uncent_power": 0.6,
        "uncent_type": "entropy",
        "gamma": 20,
        "kurtosis_threshold": 2.4,
        "residual_init": 0.3,
        "igve_topk": 9,
        "use_coarse_filter": True,
        "top_classes": 2,
        "fusion_type": "mlp_direct",
        "use_patch_level_matching": False,
        "matching_solver": "opencv",
    },
}


def get_arg(args, name, default):
    return getattr(args, name, default)


def freeze_bn(module):
    if "BatchNorm" in module.__class__.__name__:
        module.eval()


def train_one_epoch(model, data_loader, optimizer, scheduler):
    model.train()
    model.apply(freeze_bn)
    model.aux_model.eval()

    correct_samples, all_samples = 0, 0
    loss_list = []

    for images, target in data_loader:
        images, target = images.cuda(), target.cuda()
        return_dict = model(images, labels=target)

        acc = cls_acc(return_dict["logits"], target)
        correct_samples += acc / 100.0 * len(return_dict["logits"])
        all_samples += len(return_dict["logits"])
        loss_list.append(return_dict["loss"].item())

        optimizer.zero_grad()
        return_dict["loss"].backward()
        optimizer.step()
        scheduler.step()

    return correct_samples / all_samples, sum(loss_list) / len(loss_list)


def train_and_eval(args, model, clip_test_features, aux_test_features, test_labels, train_loader):
    model.cuda()
    model.requires_grad_(False)
    model.aux_adapter.requires_grad_(True)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params_m = trainable_params / 1e6

    param_groups = [
        {"params": model.aux_adapter.parameters(), "lr": args.lr, "weight_decay": 0.01}
    ]

    if hasattr(model, "adaptive_fusion") and hasattr(model.adaptive_fusion, "fusion_layer"):
        fusion_params = list(model.adaptive_fusion.fusion_layer.parameters())
        if len(fusion_params) > 0:
            param_groups.append({
                "params": fusion_params,
                "lr": args.lr * 10,
                "weight_decay": 0.0,
            })
    optimizer = torch.optim.AdamW(param_groups, eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        args.train_epoch * len(train_loader),
    )

    best_acc = 0.0
    best_aux = 0.0
    best_epoch = 0
    total_train_time_sec = 0.0

    epoch_pbar = tqdm(range(1, args.train_epoch + 1), desc="Training", leave=False)
    for epoch in epoch_pbar:
        torch.cuda.synchronize()
        start_time = time.time()

        train_acc, train_loss = train_one_epoch(model, train_loader, optimizer, scheduler)

        torch.cuda.synchronize()
        total_train_time_sec += time.time() - start_time

        model.eval()
        with torch.no_grad():
            return_dict = model(
                clip_features=clip_test_features,
                aux_features=aux_test_features,
                labels=test_labels,
                use_adaptive_fusion=True,
            )
            acc = cls_acc(return_dict["logits"], test_labels)
            acc_aux = cls_acc(return_dict["aux_logits"], test_labels)

        if acc > best_acc:
            best_acc = acc
            best_aux = acc_aux
            best_epoch = epoch

        epoch_pbar.set_postfix({
            "loss": f"{train_loss:.4f}",
            "train": f"{train_acc * 100:.2f}%",
            "acc": f"{acc:.2f}%",
            "best": f"{best_acc:.2f}%",
        })

    train_time_min = total_train_time_sec / 60.0

    model.eval()
    batch_size = min(32, clip_test_features.shape[0])
    test_iters = 100
    dummy_clip_feat = clip_test_features[:batch_size]
    dummy_aux_feat = aux_test_features[:batch_size]
    dummy_labels = test_labels[:batch_size]

    with torch.no_grad():
        for _ in range(10):
            model(
                clip_features=dummy_clip_feat,
                aux_features=dummy_aux_feat,
                labels=dummy_labels,
                use_adaptive_fusion=True,
            )

        torch.cuda.synchronize()
        latency_start = time.time()
        for _ in range(test_iters):
            model(
                clip_features=dummy_clip_feat,
                aux_features=dummy_aux_feat,
                labels=dummy_labels,
                use_adaptive_fusion=True,
            )
        torch.cuda.synchronize()
        latency_end = time.time()

    latency_ms = ((latency_end - latency_start) / test_iters / batch_size) * 1000

    return best_acc, best_aux, best_epoch, train_time_min, latency_ms, params_m


def run_cvtr(args, logger):
    random.seed(args.rand_seed)
    torch.manual_seed(args.torch_rand_seed)

    logger.info("Loading CLIP model...")
    clip_model, _ = clip.load(args.clip_backbone)
    clip_model.eval()

    aux_pretrain_path = get_arg(args, "aux_pretrain_path", "./pretrained/aux_encoder.pth.tar")
    logger.info("Loading auxiliary visual encoder...")
    aux_model, args.feat_dim = load_aux_encoder(aux_pretrain_path)
    aux_model.cuda()
    aux_model.eval()

    logger.info(f"Loading ImageNet dataset with {args.shots} shots...")
    dataset = ImageNet(args.root_path, args.shots)
    class_num = len(dataset.classnames) if hasattr(dataset, "classnames") else 1000

    test_loader = torch.utils.data.DataLoader(
        dataset.test,
        batch_size=get_arg(args, "test_batch_size", 128),
        num_workers=get_arg(args, "num_workers", 8),
        shuffle=False,
    )
    train_loader = torch.utils.data.DataLoader(
        dataset.train,
        batch_size=args.batch_size,
        num_workers=get_arg(args, "num_workers", 8),
        shuffle=True,
    )
    train_loader_feature = torch.utils.data.DataLoader(
        dataset.train,
        batch_size=get_arg(args, "feature_batch_size", 256),
        num_workers=get_arg(args, "num_workers", 8),
        shuffle=False,
    )

    logger.info("Loading test features...")
    test_clip_features, test_labels = load_test_features(
        args, "test", clip_model, test_loader, model_name="clip", tfm_norm=tfm_clip
    )
    test_aux_features, _ = load_test_features(
        args, "test", aux_model, test_loader, model_name="aux", tfm_norm=tfm_aux
    )
    test_clip_features = test_clip_features.cuda()
    test_aux_features = test_aux_features.cuda()
    test_labels = test_labels.cuda()

    logger.info("Loading training features for IGVE...")
    aux_features, aux_labels = load_aux_weight(
        args, aux_model, train_loader_feature, tfm_norm=tfm_aux
    )

    logger.info("Preparing IGTR components...")
    prompt_path = get_arg(args, "igtr_prompt_path", "./prompts/imagenet.json")
    igtr_prompts = load_igtr_prompts(prompt_path, dataset.classnames)

    logger.info("Loading IGTR text prompt features...")
    text_features_all = extract_all_text_features(
        dataset.classnames,
        clip_model,
        dataset.template,
        igtr_prompts,
    )

    logger.info("Extracting support image features for IGTR...")
    clip_train_features, clip_train_labels = load_clip_features(
        args, clip_model, train_loader_feature
    )

    logger.info("Computing visual prototypes for IGTR...")
    image_prototypes = compute_image_prototypes(
        clip_train_features,
        clip_train_labels,
        len(dataset.classnames),
    )

    config = DEFAULT_CVTR_CONFIG.get(args.shots, DEFAULT_CVTR_CONFIG[16])
    config.update({
        "alpha": get_arg(args, "alpha", config["alpha"]),
        "lambda_merge": get_arg(args, "lambda_merge", config["lambda_merge"]),
        "uncent_power": get_arg(args, "uncent_power", config["uncent_power"]),
        "uncent_type": get_arg(args, "uncent_type", config["uncent_type"]),
        "gamma": get_arg(args, "gamma", config["gamma"]),
        "kurtosis_threshold": get_arg(args, "kurtosis_threshold", config["kurtosis_threshold"]),
        "residual_init": get_arg(args, "residual_init", config["residual_init"]),
        "igve_topk": get_arg(args, "igve_topk", config["igve_topk"]),
        "use_coarse_filter": get_arg(args, "use_coarse_filter", config["use_coarse_filter"]),
        "top_classes": get_arg(args, "top_classes", config["top_classes"]),
        "fusion_type": get_arg(args, "fusion_type", config["fusion_type"]),
        "use_patch_level_matching": get_arg(args, "use_patch_level_matching", config.get("use_patch_level_matching", False)),
        "matching_solver": get_arg(args, "matching_solver", config.get("matching_solver", "opencv")),
    })

    logger.info("Optimizing text features with IGTR...")
    optimized_text_features, prompt_weights = igtr_optimize_prompts(
        text_features_all,
        image_prototypes,
        gamma=config["gamma"],
    )
    clip_weights = optimized_text_features.t()

    with torch.no_grad():
        normalized_clip_features = test_clip_features / test_clip_features.norm(dim=-1, keepdim=True)
        zero_shot_logits = 100.0 * normalized_clip_features @ clip_weights
        zero_shot_acc = zero_shot_logits.argmax(dim=-1).eq(test_labels).sum().item() / len(test_labels) * 100

    logger.info(f"IGTR zero-shot accuracy: {zero_shot_acc:.2f}%")

    model = CVTR_Model(
        clip_model=clip_model,
        aux_model=aux_model,
        sample_features=[aux_features, aux_labels],
        clip_weights=clip_weights,
        feat_dim=args.feat_dim,
        class_num=class_num,
        lambda_merge=config["lambda_merge"],
        alpha=config["alpha"],
        uncent_type=config["uncent_type"],
        uncent_power=config["uncent_power"],
        use_adaptive_fusion=True,
        fusion_topk=config["igve_topk"],
        fusion_chunk_size=get_arg(args, "igve_chunk_size", 256),
        kurtosis_threshold=config["kurtosis_threshold"],
        residual_init=config["residual_init"],
        fusion_type=config["fusion_type"],
        mlp_hidden_dim=get_arg(args, "mlp_hidden_dim", 512),
        use_coarse_filter=config["use_coarse_filter"],
        top_classes=config["top_classes"],
        use_patch_level_matching=config["use_patch_level_matching"],
        matching_solver=config["matching_solver"],
    )

    best_acc, best_aux, best_epoch, train_time_min, latency_ms, params_m = train_and_eval(
        args,
        model,
        test_clip_features,
        test_aux_features,
        test_labels,
        train_loader,
    )

    result = {
        "shots": args.shots,
        "zero_shot_acc": zero_shot_acc,
        "best_acc": best_acc,
        "best_aux": best_aux,
        "best_epoch": best_epoch,
        "train_time_min": train_time_min,
        "latency_ms": latency_ms,
        "params_M": params_m,
        **config,
    }

    logger.info("\n" + "=" * 80)
    logger.info("CVTR RESULT")
    logger.info("=" * 80)
    logger.info(f"Zero-shot Acc: {zero_shot_acc:.2f}%")
    logger.info(f"Best Acc: {best_acc:.2f}% at epoch {best_epoch}")
    logger.info(f"Aux Acc: {best_aux:.2f}%")
    logger.info(f"Training Time: {train_time_min:.2f} min")
    logger.info(f"Latency: {latency_ms:.2f} ms/image")
    logger.info(f"Trainable Params: {params_m:.2f} M")
    logger.info("=" * 80)

    result_dir = get_arg(args, "result_dir", "./outputs")
    os.makedirs(result_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%m-%d-%H_%M")
    result_file = os.path.join(result_dir, f"cvtr_imagenet_{args.shots}shot_{timestamp}.csv")

    with open(result_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result.keys()))
        writer.writeheader()
        writer.writerow(result)

    logger.info(f"Saved result to: {result_file}")
    return result


if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    cache_root = get_arg(args, "cache_root", "./features")
    args.cache_dir = os.path.join(cache_root, "imagenet", f"{args.shots}shot")
    os.makedirs(args.cache_dir, exist_ok=True)

    result_dir = get_arg(args, "result_dir", "./outputs")
    os.makedirs(result_dir, exist_ok=True)

    logger = config_logging(args)
    logger.info("Starting single-run CVTR training and evaluation on ImageNet")

    run_cvtr(args, logger)
