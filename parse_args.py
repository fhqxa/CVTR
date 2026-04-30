import argparse
import yaml


def parse_args():
    """Parse arguments for CVTR training and inference."""

    with open('default.yaml', 'r', encoding='utf-8') as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    parser = argparse.ArgumentParser(description='CVTR Few-Shot Classification')

    # Basic settings
    parser.add_argument('--gpu',
                        type=str,
                        default=str(cfg.get('gpu', '0')),
                        help='GPU id, e.g., 0, 1, or 0,1')

    parser.add_argument('--exp_name',
                        type=str,
                        default=cfg.get('exp_name', 'cvtr_exp'),
                        help='experiment name')

    parser.add_argument('--rand_seed',
                        type=int,
                        default=cfg.get('rand_seed', 2),
                        help='random seed')

    parser.add_argument('--torch_rand_seed',
                        type=int,
                        default=cfg.get('torch_rand_seed', 1),
                        help='PyTorch random seed')

    # Dataset settings
    parser.add_argument('--root_path',
                        type=str,
                        default=cfg.get('root_path', './data'),
                        help='root path of datasets')

    parser.add_argument('--dataset',
                        type=str,
                        default=cfg.get('dataset', 'imagenet'),
                        help='dataset name')

    parser.add_argument('--shots',
                        type=int,
                        default=cfg.get('shots', 16),
                        help='number of shots per class')

    # Training settings
    parser.add_argument('--train_epoch',
                        type=int,
                        default=cfg.get('train_epoch', 50),
                        help='number of training epochs')

    parser.add_argument('--lr',
                        type=float,
                        default=cfg.get('lr', 0.001),
                        help='learning rate')

    parser.add_argument('--batch_size',
                        type=int,
                        default=cfg.get('batch_size', 8),
                        help='training batch size')

    parser.add_argument('--val_batch_size',
                        type=int,
                        default=cfg.get('val_batch_size', 128),
                        help='validation/test batch size')

    parser.add_argument('--augment_epoch',
                        type=int,
                        default=cfg.get('augment_epoch', 1),
                        help='number of augmentation epochs for feature extraction')

    # Backbone and feature cache settings
    parser.add_argument('--clip_backbone',
                        type=str,
                        default=cfg.get('clip_backbone', 'RN50'),
                        help='CLIP visual backbone')

    parser.add_argument('--aux_backbone_path',
                        type=str,
                        default=cfg.get('aux_backbone_path', './pretrained/r-50-1000ep.pth.tar'),
                        help='path to the auxiliary visual encoder checkpoint')

    parser.add_argument('--cache_dir',
                        type=str,
                        default=cfg.get('cache_dir', './features'),
                        help='directory for cached features')

    parser.add_argument('--output_dir',
                        type=str,
                        default=cfg.get('output_dir', './outputs'),
                        help='directory for logs and results')

    parser.add_argument('--load_pre_feat',
                        action='store_true',
                        help='load cached test features')

    parser.add_argument('--load_aux_weight',
                        action='store_true',
                        help='load cached auxiliary support features')

    # IGTR settings
    parser.add_argument('--igtr_prompt_path',
                        type=str,
                        default=cfg.get('igtr_prompt_path', './prompts/imagenet.json'),
                        help='path to IGTR text prompt file')

    parser.add_argument('--igtr_gamma',
                        type=float,
                        default=cfg.get('igtr_gamma', 20.0),
                        help='scaling factor for IGTR prompt weighting')

    # IGVE settings
    parser.add_argument('--use_igve',
                        action='store_true',
                        default=cfg.get('use_igve', False),
                        help='enable IGVE during inference')

    parser.add_argument('--igve_topk',
                        type=int,
                        default=cfg.get('igve_topk', 6),
                        help='number of support samples used for IGVE retrieval')

    parser.add_argument('--kurtosis_threshold',
                        type=float,
                        default=cfg.get('kurtosis_threshold', 2.3),
                        help='kurtosis threshold for uncertain query identification')

    parser.add_argument('--use_coarse_filter',
                        action='store_true',
                        default=cfg.get('use_coarse_filter', False),
                        help='use CLIP top-class filtering before IGVE retrieval')

    parser.add_argument('--top_classes',
                        type=int,
                        default=cfg.get('top_classes', 5),
                        help='number of candidate classes for coarse filtering')

    # Patch-Level Optimal Matching settings
    parser.add_argument('--use_patch_matching',
                        action='store_true',
                        default=cfg.get('use_patch_matching', False),
                        help='enable Patch-Level Optimal Matching in IGVE')

    parser.add_argument('--matching_solver',
                        type=str,
                        default=cfg.get('matching_solver', 'qpth'),
                        choices=['qpth', 'opencv'],
                        help='solver for Patch-Level Optimal Matching')

    parser.add_argument('--matching_form',
                        type=str,
                        default=cfg.get('matching_form', 'QP'),
                        choices=['QP', 'L2'],
                        help='QPTH formulation for Patch-Level Optimal Matching')

    parser.add_argument('--matching_l2_strength',
                        type=float,
                        default=cfg.get('matching_l2_strength', 0.0001),
                        help='L2 regularization strength for Patch-Level Optimal Matching')

    # FSCF settings
    parser.add_argument('--alpha',
                        type=float,
                        default=cfg.get('alpha', 0.5),
                        help='visual-text fusion coefficient')

    parser.add_argument('--lambda_merge',
                        type=float,
                        default=cfg.get('lambda_merge', 0.2),
                        help='loss balancing coefficient')

    parser.add_argument('--uncent_type',
                        type=str,
                        default=cfg.get('uncent_type', 'entropy'),
                        help='uncertainty type for adaptive fusion')

    parser.add_argument('--uncent_power',
                        type=float,
                        default=cfg.get('uncent_power', 0.6),
                        help='uncertainty scaling power')

    return parser
