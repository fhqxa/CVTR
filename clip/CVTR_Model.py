"""
clip/CVTR_Model.py

"""

import torch
import cv2
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, Normalize
from tqdm import tqdm
try:
    from qpth.qp import QPFunction
except ImportError:
    QPFunction = None


# ======================================================
# 🔹 工具函数
# ======================================================
def logit_normalize(logit: torch.Tensor) -> torch.Tensor:
    """Per-sample logit normalization (z-score along classes)."""
    logits_std = torch.std(logit, dim=1, keepdim=True)
    logits_mean = torch.mean(logit, dim=1, keepdim=True)
    return (logit - logits_mean) / (logits_std + 1e-8)


def uncertainty(logits: torch.Tensor, type: str, power: float) -> torch.Tensor:
    """Uncertainty factor computed from probabilities (softmax)."""
    softmax_fun = nn.Softmax(dim=-1)
    probs = softmax_fun(logits)

    if type == 'entropy':
        entropy = -torch.sum(probs * torch.log2(probs + 1e-8), dim=-1, keepdim=True)
        entropy = entropy / torch.log2(torch.tensor(probs.shape[-1], device=probs.device).float())
        return (entropy * power).exp()
        
    elif type == 'none':
        return torch.ones((probs.shape[0], 1), device=probs.device)

    else:
        raise RuntimeError(f'Invalid uncertainty type: {type}')


def compute_kurtosis(probs: torch.Tensor) -> torch.Tensor:
    """Kurtosis of probability distribution for confidence gating."""
    mean = probs.mean(dim=-1, keepdim=True)
    std = probs.std(dim=-1, keepdim=True) + 1e-8
    normalized = (probs - mean) / std
    kurtosis = (normalized ** 4).mean(dim=-1)
    return kurtosis


# ======================================================
# 🔹 IGVE: Patch-Level Optimal Matching
# ======================================================
def patch_level_optimal_matching_qpth(cost_matrix, weight_query, weight_support,
                                      form='L2', l2_strength=0.0001):
    """
    Patch-Level Optimal Matching using QPTH.

    Args:
        cost_matrix: [B, HW, HW], where c_ij = 1 - cosine(u_i, v_j)
        weight_query: [B, HW], marginal weights for query patches
        weight_support: [B, HW], marginal weights for support patches

    Returns:
        matching_score: [B], equal to sum_ij w_ij * cosine(u_i, v_j)
        flow: [B, HW, HW], corresponding to w_ij
    """
    if QPFunction is None:
        raise ImportError("qpth is not installed. Please install qpth or use the OpenCV solver.")

    cost_matrix = cost_matrix.cuda()
    weight_query = weight_query.cuda()
    weight_support = weight_support.cuda()

    weight_query = (weight_query * weight_query.shape[-1]) / (weight_query.sum(1).unsqueeze(1) + 1e-8)
    weight_support = (weight_support * weight_support.shape[-1]) / (weight_support.sum(1).unsqueeze(1) + 1e-8)

    nbatch = cost_matrix.shape[0]
    nq = weight_query.shape[1]
    ns = weight_support.shape[1]
    nelement = nq * ns

    Q_1 = cost_matrix.view(nbatch, 1, nelement).double()

    if form == 'QP':
        Q = torch.bmm(Q_1.transpose(2, 1), Q_1).double() + 1e-4 * torch.eye(
            nelement, device=cost_matrix.device
        ).double().unsqueeze(0).repeat(nbatch, 1, 1)
        p = torch.zeros(nbatch, nelement, device=cost_matrix.device).double()
    elif form == 'L2':
        Q = (l2_strength * torch.eye(nelement, device=cost_matrix.device).double()).unsqueeze(0).repeat(nbatch, 1, 1)
        p = cost_matrix.view(nbatch, nelement).double()
    else:
        raise ValueError('Unknown form')

    h_1 = torch.zeros(nbatch, nelement, device=cost_matrix.device).double()
    h_2 = torch.cat([weight_query, weight_support], 1).double()
    h = torch.cat((h_1, h_2), 1)

    G_1 = -torch.eye(nelement, device=cost_matrix.device).double().unsqueeze(0).repeat(nbatch, 1, 1)
    G_2 = torch.zeros([nbatch, nq + ns, nelement], device=cost_matrix.device).double()

    for i in range(nq):
        G_2[:, i, ns * i:ns * (i + 1)] = 1
    for j in range(ns):
        G_2[:, nq + j, j::ns] = 1

    G = torch.cat((G_1, G_2), 1)
    A = torch.ones(nbatch, 1, nelement, device=cost_matrix.device).double()
    b = torch.min(torch.sum(weight_query, 1), torch.sum(weight_support, 1)).unsqueeze(1).double()

    flow = QPFunction(verbose=-1)(Q, p, G, h, A, b)
    similarity_matrix = 1.0 - Q_1.squeeze(1)
    matching_score = torch.sum(similarity_matrix * flow, dim=1)
    return matching_score, flow.view(-1, nq, ns)


def patch_level_optimal_matching_opencv(cost_matrix, weight_query, weight_support):
    """Patch-Level Optimal Matching using OpenCV."""
    cost_matrix_np = cost_matrix.detach().cpu().numpy().astype('float32')

    weight_query = F.relu(weight_query.detach().cpu()) + 1e-5
    weight_support = F.relu(weight_support.detach().cpu()) + 1e-5

    weight_query = (weight_query * (weight_query.shape[0] / weight_query.sum().item())).view(-1, 1).numpy().astype('float32')
    weight_support = (weight_support * (weight_support.shape[0] / weight_support.sum().item())).view(-1, 1).numpy().astype('float32')

    _, _, flow = cv2.EMD(weight_query, weight_support, cv2.DIST_USER, cost_matrix_np)
    flow = torch.from_numpy(flow).double()
    similarity_matrix = 1.0 - cost_matrix.detach().cpu().double()
    matching_score = torch.sum(similarity_matrix * flow)
    return matching_score, flow


def patch_level_optimal_matching_batch(cost_matrix, weight_query, weight_support, solver='opencv'):
    """Batch wrapper for Patch-Level Optimal Matching."""
    if solver == 'qpth':
        return patch_level_optimal_matching_qpth(cost_matrix, weight_query, weight_support)

    score_list, flow_list = [], []
    for i in range(cost_matrix.shape[0]):
        score, flow = patch_level_optimal_matching_opencv(cost_matrix[i], weight_query[i], weight_support[i])
        score_list.append(score)
        flow_list.append(flow)

    matching_score = torch.stack(score_list, dim=0).cuda().double()
    flow = torch.stack(flow_list, dim=0).cuda().double()
    return matching_score, flow


# ======================================================
# 🔹 线性适配器
# ======================================================
class Linear_Adapter(nn.Module):
    """Auxiliary feature adapter -> class logits."""

    def __init__(self, feat_dim: int, class_num: int, sample_features=None):
        super().__init__()
        self.fc = nn.Linear(feat_dim, class_num, bias=False)

        if sample_features is not None:
            print('✅ Initializing adapter weight by training samples...')
            aux_features, aux_labels = sample_features[0], sample_features[1]

            init_weight = torch.zeros(feat_dim, class_num, device=aux_features.device)
            for i in range(len(aux_labels)):
                init_weight[:, aux_labels[i]] += aux_features[i]

            feat_per_class = len(aux_labels) / class_num
            init_weight = init_weight / feat_per_class
            self.fc.weight = nn.Parameter(init_weight.t())

            print(f'   Adapter initialized with {len(aux_labels)} training samples')
            print(f'   Feature dim: {feat_dim}, Classes: {class_num}')
        else:
            print('⚠️  Initializing adapter weight randomly...')

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.fc(feat)


# ======================================================
# 🔹 融合层 1: 简单平均（simple mean）
# ======================================================
class SimpleMeanFusion(nn.Module):
    """
    residual = mean(support_logits)
    output = query + α * residual
    """

    def __init__(self, num_support: int, num_classes: int, residual_init: float = 0.3):
        super().__init__()
        self.residual_alpha = nn.Parameter(torch.tensor(residual_init))

        print("✅ Simple Mean Fusion initialized")
        print("   Formula: output = query + α * mean(supports)")
        print(f"   Initial α: {residual_init}")

    def forward(self, support_logits: torch.Tensor, query_logits: torch.Tensor):
        residual = support_logits.mean(dim=1)
        alpha = torch.tanh(self.residual_alpha) * 0.5
        fused = query_logits + alpha * residual
        return fused, alpha


# ======================================================
# 🔹 融合层 2: 相似度加权（Similarity-weighted）
# ======================================================
class SimilarityWeightedFusion(nn.Module):
    """
    weights = softmax(sim / T + bias)
    residual = Σ weights * support_logits
    output = query + α * residual
    """

    def __init__(self, num_support: int, num_classes: int, residual_init: float = 0.3):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.weight_bias = nn.Parameter(torch.zeros(num_support))
        self.residual_alpha = nn.Parameter(torch.tensor(residual_init))

        print("✅ Similarity-Weighted Fusion initialized")
        print("   Learnable: temperature, weight_bias, α")
        print(f"   Initial α: {residual_init}")

    def forward(self, support_logits: torch.Tensor, query_logits: torch.Tensor, similarities: torch.Tensor):
        scaled_sim = similarities / (torch.abs(self.temperature) + 1e-8)
        scaled_sim = scaled_sim + self.weight_bias.unsqueeze(0)
        weights = F.softmax(scaled_sim, dim=-1)

        residual = torch.einsum('bnc,bn->bc', support_logits, weights)
        alpha = torch.tanh(self.residual_alpha) * 0.5
        fused = query_logits + alpha * residual
        return fused, alpha, weights


# ======================================================
# 🔹 融合层 3: MLP 学习权重（包含 query）
# ======================================================
class MLPWeightLearner(nn.Module):
    """
    weights = softmax(MLP([supports, query]))
    residual = Σ weights * [supports, query]
    output = query + α * residual
    """

    def __init__(self, num_support: int, num_classes: int, hidden_dim: int = 512, residual_init: float = 0.3):
        super().__init__()
        input_dim = (num_support + 1) * num_classes

        self.weight_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_support + 1),
        )

        self.residual_alpha = nn.Parameter(torch.tensor(residual_init))

        print("✅ MLP Weight Learner initialized")
        print(f"   Input dim: {input_dim}, Output: {num_support + 1} weights")

    def forward(self, support_logits: torch.Tensor, query_logits: torch.Tensor):
        b = support_logits.size(0)
        all_logits = torch.cat([support_logits, query_logits.unsqueeze(1)], dim=1)  # [b, k+1, C]
        flat = all_logits.view(b, -1)

        raw = self.weight_mlp(flat)
        weights = F.softmax(raw, dim=-1)

        residual = torch.einsum('bnc,bn->bc', all_logits, weights)
        alpha = torch.tanh(self.residual_alpha) * 0.5
        fused = query_logits + alpha * residual
        return fused, alpha, weights


# ======================================================
# 🔹 融合层 4: MLP 直接融合（只看 support）
# ======================================================
class MLPDirectFusion(nn.Module):
    """
    residual = MLP([supports])
    output = query + α * residual
    """

    def __init__(self, num_support: int, num_classes: int, hidden_dim: int = 512, residual_init: float = 0.3):
        super().__init__()
        input_dim = num_support * num_classes

        self.fusion_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        self.residual_alpha = nn.Parameter(torch.tensor(residual_init))

        print("✅ MLP Direct Fusion initialized")
        print(f"   Input: {input_dim}, Output: {num_classes}")

    def forward(self, support_logits: torch.Tensor, query_logits: torch.Tensor):
        b = support_logits.size(0)
        flat_support = support_logits.view(b, -1)
        residual = self.fusion_mlp(flat_support)

        alpha = torch.tanh(self.residual_alpha) * 0.5
        fused = query_logits + alpha * residual
        return fused, alpha


# ======================================================
# 🔹 自适应 adaptive fusion 模块（支持粗筛选开关）
# ======================================================
class AdaptiveFusion(nn.Module):
    """
    Adaptive Fusion:
    - retrieval similarity computed on CPU
    - optional coarse filter:
        CLIP top-M classes -> restrict support candidates by class index
    - gating by kurtosis of query probs
    """

    def __init__(
        self,
        adapter: nn.Module,
        aux_model: nn.Module,
        support_features: torch.Tensor,
        support_labels: torch.Tensor = None,
        clip_weights: torch.Tensor = None,
        topk: int = 4,
        chunk_size: int = 256,
        kurtosis_threshold: float = 2.5,
        num_classes: int = 1000,
        residual_init: float = 0.3,
        mlp_hidden_dim: int = 512,
        fusion_type: str = 'mlp_direct',
        use_coarse_filter: bool = False,
        top_classes: int = 5,
        device: str = "cuda",
        use_patch_level_matching: bool = False,
        matching_solver: str = "opencv",
    ):
        super().__init__()

        self.adapter = adapter
        self.aux_model = aux_model

        # store support features as buffer (could be on GPU), but we always use CPU copy in retrieval
        self.register_buffer('support_features', support_features)
        self.topk = topk
        self.chunk_size = chunk_size
        self.kurtosis_threshold = kurtosis_threshold
        self.device = device
        self.fusion_type = fusion_type
        self.use_patch_level_matching = use_patch_level_matching
        self.matching_solver = matching_solver

        # coarse filter
        self.use_coarse_filter = use_coarse_filter
        self.top_classes = top_classes

        if use_coarse_filter:
            assert support_labels is not None, "❌ Need support_labels for coarse filtering!"
            assert clip_weights is not None, "❌ Need clip_weights for coarse filtering!"

            # labels always CPU for building index
            self.register_buffer('support_labels', support_labels.detach().cpu())

            # clip_weights should align with query_clip_features (GPU)
            # we register it as buffer; later in forward we move to self.device when used
            self.register_buffer('clip_weights', clip_weights.detach().clone())

            self.class_to_indices = self._build_class_index()
            print(f"✅ Coarse filtering ENABLED: top-{top_classes} classes")
        else:
            print(f"⚠️  Coarse filtering DISABLED: full search over {support_features.size(0)} supports")

        # fusion layer
        if fusion_type == 'mean':
            self.fusion_layer = SimpleMeanFusion(topk, num_classes, residual_init=residual_init)
        elif fusion_type == 'similarity':
            self.fusion_layer = SimilarityWeightedFusion(topk, num_classes, residual_init=residual_init)
        elif fusion_type == 'mlp_weight':
            self.fusion_layer = MLPWeightLearner(topk, num_classes, hidden_dim=mlp_hidden_dim, residual_init=residual_init)
        elif fusion_type == 'mlp_direct':
            self.fusion_layer = MLPDirectFusion(topk, num_classes, hidden_dim=mlp_hidden_dim, residual_init=residual_init)
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

        print("\n" + "=" * 60)
        print("✅ Adaptive Fusion Module Initialized (ImageNet)")
        print("=" * 60)
        print(f"   Fusion type: {fusion_type}")
        print(f"   Top-k: {topk}")
        print(f"   Kurtosis threshold: {kurtosis_threshold}")
        print(f"   Support pool: {support_features.size(0)} samples")
        print(f"   Coarse filter: {'ON' if use_coarse_filter else 'OFF'}")
        print(f"   Patch-Level Optimal Matching: {'ON' if use_patch_level_matching else 'OFF'}")
        print("=" * 60 + "\n")

    def _build_class_index(self):
        """Build support indices for each class (CPU)."""
        class_indices = {}
        labels_cpu = self.support_labels  # already CPU
        for label in labels_cpu.unique():
            mask = (labels_cpu == label)
            class_indices[label.item()] = torch.where(mask)[0]  # CPU indices
        print(f"   Built class index for {len(class_indices)} classes")
        return class_indices


    @torch.no_grad()
    def _compute_patch_matching_similarity(self, query_patches_cpu: torch.Tensor, support_patches_cpu: torch.Tensor):
        """
        Compute IGVE Patch-Level Optimal Matching scores.

        This branch is used only when both query and support features are patch-level tensors
        with shape [B, HW, D] and [N, HW, D]. If only pooled features are provided, CVTR falls
        back to cosine retrieval.
        """
        query_patches = F.normalize(query_patches_cpu, dim=-1)
        support_patches = F.normalize(support_patches_cpu, dim=-1)

        num_query = query_patches.size(0)
        num_support = support_patches.size(0)
        hw_q = query_patches.size(1)
        hw_s = support_patches.size(1)

        scores = []
        weight_query = torch.ones(num_support, hw_q)
        weight_support = torch.ones(num_support, hw_s)

        for i in range(num_query):
            sim = torch.einsum('id,njd->nij', query_patches[i], support_patches)
            cost = 1.0 - sim
            matching_score, _ = patch_level_optimal_matching_batch(
                cost,
                weight_query,
                weight_support,
                solver=self.matching_solver,
            )
            scores.append(matching_score.float().detach().cpu())

        return torch.stack(scores, dim=0)

    @torch.no_grad()
    def _get_topk_indices_without_filter(self, query_features_cpu: torch.Tensor, support_features_cpu: torch.Tensor):
        """Global retrieval on CPU."""
        if self.use_patch_level_matching and query_features_cpu.dim() == 3 and support_features_cpu.dim() == 3:
            sim = self._compute_patch_matching_similarity(query_features_cpu, support_features_cpu)
        else:
            sim = F.cosine_similarity(
                query_features_cpu.unsqueeze(1),           # [b,1,d] CPU
                support_features_cpu.unsqueeze(0),         # [1,N,d] CPU
                dim=-1
            )  # [b,N]
        topk_sim, topk_idx = torch.topk(sim, self.topk, dim=-1)
        return topk_sim, topk_idx

    @torch.no_grad()
    def _get_topk_indices_with_filter(
        self,
        query_features_cpu: torch.Tensor,
        query_clip_features_gpu: torch.Tensor,
        support_features_cpu: torch.Tensor,
    ):
        """
        Coarse filter:
        1) use CLIP logits (GPU) to get top-M classes
        2) retrieve top-k supports only from candidate classes, similarity on CPU
        """
        b = query_features_cpu.size(0)

        # ensure clip_weights on same device as query_clip_features_gpu
        clip_w = self.clip_weights.to(query_clip_features_gpu.device)

        q_norm = F.normalize(query_clip_features_gpu, dim=-1)
        clip_logits = 100.0 * (q_norm @ clip_w)  # [b, C]
        _, topM_classes = torch.topk(clip_logits, self.top_classes, dim=-1)
        topM_classes_cpu = topM_classes.cpu()

        topk_indices_list = []
        topk_sim_list = []

        for i in range(b):
            candidate_support_indices = []

            for cls in topM_classes_cpu[i]:
                cls_item = cls.item()
                if cls_item in self.class_to_indices:
                    candidate_support_indices.extend(self.class_to_indices[cls_item].tolist())

            if len(candidate_support_indices) == 0:
                candidate_support_indices = list(range(support_features_cpu.size(0)))

            candidate_support_indices = torch.tensor(candidate_support_indices, device='cpu')
            candidate_supports = support_features_cpu[candidate_support_indices]  # CPU

            if self.use_patch_level_matching and query_features_cpu.dim() == 3 and support_features_cpu.dim() == 3:
                sim = self._compute_patch_matching_similarity(
                    query_features_cpu[i:i + 1],
                    candidate_supports,
                ).squeeze(0)  # [Ncand]
            else:
                sim = F.cosine_similarity(
                    query_features_cpu[i:i + 1].unsqueeze(1),    # CPU
                    candidate_supports.unsqueeze(0),             # CPU
                    dim=-1
                ).squeeze(0)  # [Ncand]

            k = min(self.topk, sim.numel())
            topk_sim, topk_local = torch.topk(sim, k)
            topk_global = candidate_support_indices[topk_local]

            # pad
            if topk_global.numel() < self.topk:
                pad_len = self.topk - topk_global.numel()
                topk_global = torch.cat([topk_global, topk_global[-1].repeat(pad_len)])
                topk_sim = torch.cat([topk_sim, topk_sim[-1].repeat(pad_len)])

            topk_indices_list.append(topk_global)
            topk_sim_list.append(topk_sim)

        topk_idx = torch.stack(topk_indices_list, dim=0)  # CPU
        topk_sim = torch.stack(topk_sim_list, dim=0)      # CPU
        return topk_sim, topk_idx

    def forward(self, query_features: torch.Tensor, query_clip_features: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            query_features: aux features (GPU or CPU). We will copy to CPU for retrieval and GPU for adapter.
            query_clip_features: CLIP features (GPU) for coarse filtering if enabled.
        Returns:
            fused logits (GPU)
        """
        # retrieval is always on CPU
        support_features_cpu = self.support_features.detach().cpu()
        query_features_cpu = query_features.detach().cpu()

        Nq = query_features_cpu.size(0)
        Ns = support_features_cpu.size(0)

        if self.use_coarse_filter and query_clip_features is None:
            raise ValueError("❌ Coarse filtering requires query_clip_features!")

        result_chunks = []
        num_confident, num_uncertain = 0, 0
        all_weights = []

        mode_str = f"COARSE FILTER (top-{self.top_classes})" if self.use_coarse_filter else "FULL SEARCH"
        print(f"\n🔍 [Adaptive Fusion - {mode_str}] N_query={Nq}, N_support={Ns}, topk={self.topk}")

        for start in tqdm(range(0, Nq, self.chunk_size), desc="Adaptive Fusion", ncols=100):
            end = min(start + self.chunk_size, Nq)

            q_chunk_cpu = query_features_cpu[start:end]                 # CPU
            q_chunk_gpu = q_chunk_cpu.to(self.device, non_blocking=True)  # GPU
            chunk_size = end - start

            # 1) query logits on GPU
            query_logits = self.adapter(q_chunk_gpu)  # GPU
            query_probs = F.softmax(query_logits, dim=-1)
            kurtosis = compute_kurtosis(query_probs)  # GPU tensor

            # 2) top-k retrieval on CPU
            if self.use_coarse_filter:
                q_clip_chunk_gpu = query_clip_features[start:end].to(self.device, non_blocking=True)
                topk_sim_cpu, topk_idx_cpu = self._get_topk_indices_with_filter(
                    query_features_cpu=q_chunk_cpu,
                    query_clip_features_gpu=q_clip_chunk_gpu,
                    support_features_cpu=support_features_cpu
                )
            else:
                topk_sim_cpu, topk_idx_cpu = self._get_topk_indices_without_filter(
                    query_features_cpu=q_chunk_cpu,
                    support_features_cpu=support_features_cpu
                )

            # 3) get support logits (CPU -> GPU)
            support_topk_cpu = support_features_cpu[topk_idx_cpu]  # [chunk, k, d] CPU
            support_topk_gpu = support_topk_cpu.view(-1, support_topk_cpu.size(-1)).to(self.device, non_blocking=True)
            support_logits = self.adapter(support_topk_gpu).view(chunk_size, self.topk, -1)  # GPU

            # 4) fusion (GPU)
            final_logits = []
            topk_sim_gpu = topk_sim_cpu.to(self.device, non_blocking=True)

            for i in range(chunk_size):
                if kurtosis[i] < self.kurtosis_threshold:
                    # uncertain -> fuse
                    if self.fusion_type == 'mean':
                        fused, _ = self.fusion_layer(support_logits[i:i + 1], query_logits[i:i + 1])
                    elif self.fusion_type == 'similarity':
                        fused, _, weights = self.fusion_layer(
                            support_logits[i:i + 1],
                            query_logits[i:i + 1],
                            topk_sim_gpu[i:i + 1]
                        )
                        all_weights.append(weights.detach().cpu())
                    elif self.fusion_type == 'mlp_weight':
                        fused, _, weights = self.fusion_layer(support_logits[i:i + 1], query_logits[i:i + 1])
                        all_weights.append(weights.detach().cpu())
                    elif self.fusion_type == 'mlp_direct':
                        fused, _ = self.fusion_layer(support_logits[i:i + 1], query_logits[i:i + 1])
                    else:
                        raise ValueError(f"Unknown fusion_type: {self.fusion_type}")

                    final_logits.append(fused.squeeze(0))
                    num_uncertain += 1
                else:
                    final_logits.append(query_logits[i])
                    num_confident += 1

            final_logits = torch.stack(final_logits, dim=0)  # GPU
            result_chunks.append(final_logits.detach().cpu())

            # cleanup
            del q_chunk_cpu, q_chunk_gpu, query_logits, query_probs, kurtosis
            del topk_sim_cpu, topk_idx_cpu, support_topk_cpu, support_topk_gpu, support_logits, topk_sim_gpu
            if self.use_coarse_filter:
                del q_clip_chunk_gpu
            torch.cuda.empty_cache()

        result = torch.cat(result_chunks, dim=0).to(self.device)

        print(f"✅ [Adaptive Fusion] Done. Shape: {result.shape}")
        print(f"📊 Confident: {num_confident}/{Nq} ({100 * num_confident / Nq:.1f}%), "
              f"Uncertain: {num_uncertain}/{Nq} ({100 * num_uncertain / Nq:.1f}%)")

        if len(all_weights) > 0:
            avg_weights = torch.cat(all_weights, dim=0).mean(dim=0).numpy()
            print(f"📊 Average learned weights: {avg_weights}")

        with torch.no_grad():
            alpha = torch.tanh(self.fusion_layer.residual_alpha).item() * 0.5
            print(f"📊 Residual strength α = {alpha:.4f}")

            if hasattr(self.fusion_layer, 'temperature'):
                temp = self.fusion_layer.temperature.item()
                print(f"📊 Temperature = {temp:.4f}")

        return result


# ======================================================
# ImageNet transforms
# ======================================================
tfm_clip = Compose([Normalize((0.48145466, 0.4578275, 0.40821073),
                              (0.26862954, 0.26130258, 0.27577711))])

tfm_aux = Compose([Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])


# ======================================================
# 🔹 CVTR 模型（ImageNet）
# ======================================================
class CVTR_Model(nn.Module):
    def __init__(
        self,
        clip_model,
        aux_model,
        sample_features,
        clip_weights,
        feat_dim,
        class_num,
        lambda_merge,
        alpha,
        uncent_type,
        uncent_power,
        # IGVE enhancement parameters
        fusion_dim=None,                     # unused, keep for compatibility
        use_adaptive_fusion=False,
        fusion_topk=4,
        fusion_temperature=0.1,              # unused, keep for compatibility
        fusion_chunk_size=256,
        kurtosis_threshold=2.5,
        residual_init=0.3,
        fusion_type='mlp_direct',
        mlp_hidden_dim=512,
        use_coarse_filter=False,
        top_classes=5,
        use_patch_level_matching=False,
        matching_solver="opencv",
        device="cuda"
    ):
        super().__init__()

        self.clip_model = clip_model
        self.aux_model = aux_model

        # keep as buffer so moved with model.to(device)
        self.register_buffer("clip_weights", clip_weights.detach().clone())

        self.aux_adapter = Linear_Adapter(feat_dim, class_num, sample_features=sample_features)

        self.lambda_merge = lambda_merge
        self.uncent_type = uncent_type
        self.uncent_power = uncent_power
        self.alpha = alpha

        self.use_adaptive_fusion = use_adaptive_fusion

        if use_adaptive_fusion and sample_features is not None:
            aux_features, aux_labels = sample_features  # aux_features: [Ns, d], aux_labels: [Ns]

            self.adaptive_fusion = AdaptiveFusion(
                adapter=self.aux_adapter,
                aux_model=self.aux_model,
                support_features=aux_features,
                support_labels=aux_labels if use_coarse_filter else None,
                clip_weights=self.clip_weights if use_coarse_filter else None,
                topk=fusion_topk,
                chunk_size=fusion_chunk_size,
                kurtosis_threshold=kurtosis_threshold,
                num_classes=class_num,
                residual_init=residual_init,
                mlp_hidden_dim=mlp_hidden_dim,
                fusion_type=fusion_type,
                use_coarse_filter=use_coarse_filter,
                top_classes=top_classes,
                device=device,
                use_patch_level_matching=use_patch_level_matching,
                matching_solver=matching_solver
            )

    def forward(
        self,
        images=None,
        clip_features=None,
        aux_features=None,
        labels=None,
        use_adaptive_fusion=False
    ):
        if images is not None:
            clip_features, aux_features = self.forward_feature(images)

        # normalize features
        clip_features = clip_features / clip_features.norm(dim=-1, keepdim=True)
        aux_features = aux_features / aux_features.norm(dim=-1, keepdim=True)

        # CLIP logits
        clip_logits = 100.0 * (clip_features @ self.clip_weights)

        # AUX logits (with optional adaptive fusion)
        if use_adaptive_fusion and hasattr(self, 'adaptive_fusion'):
            # ✅ pass clip_features for coarse filter
            aux_logits = self.adaptive_fusion(aux_features, query_clip_features=clip_features)
            aux_logits = logit_normalize(aux_logits)
        else:
            aux_logits = logit_normalize(self.aux_adapter(aux_features))

        # Fusion
        factor = uncertainty(clip_logits.float(), type=self.uncent_type, power=self.uncent_power)
        logits = clip_logits + factor * aux_logits * self.alpha

        # Loss
        if labels is not None:
            loss_merge = F.cross_entropy(logits, labels)
            loss_aux = F.cross_entropy(aux_logits, labels)
            loss = self.lambda_merge * loss_merge + (1 - self.lambda_merge) * loss_aux
        else:
            loss, loss_aux, loss_merge = None, None, None

        return {
            "logits": logits,
            "clip_logits": clip_logits,
            "aux_logits": aux_logits,
            "loss": loss,
            "loss_merge": loss_merge,
            "loss_aux": loss_aux,
        }

    def forward_feature(self, images: torch.Tensor):
        # CLIP branch
        clip_features = self.clip_model.encode_image(tfm_clip(images))
        # AUX branch
        aux_features = self.aux_model(tfm_aux(images))
        return clip_features, aux_features


__all__ = [
    "logit_normalize",
    "uncertainty",
    "compute_kurtosis",
    "patch_level_optimal_matching_qpth",
    "patch_level_optimal_matching_opencv",
    "patch_level_optimal_matching_batch",
    "Linear_Adapter",
    "AdaptiveFusion",
    "CVTR_Model",
]
