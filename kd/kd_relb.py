import torch
import torch.nn.functional as F


def _safe_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Normalize each row vector in x to unit length in a numerically stable way.
    x: [B, H]
    returns: [B, H] normalized
    """
    x32 = x.float()
    denom = x32.norm(dim=-1, keepdim=True).clamp_min(eps)
    return x32 / denom


def _angle_matrix(x: torch.Tensor) -> torch.Tensor:
    """
    Cosine-similarity Gram matrix between all pairs in batch.
    x: [B, H]
    returns: [B, B]
    """
    x_norm = _safe_normalize(x)
    gram = x_norm @ x_norm.t()  # cosine sim
    # clamp tiny fp drift outside [-1, 1]
    return gram.clamp(min=-1.0, max=1.0)


def _pairwise_dist(x: torch.Tensor) -> torch.Tensor:
    """
    Pairwise L2 distances between all pairs in batch.
    x: [B, H]
    returns: [B, B]
    """
    x32 = x.float()
    return torch.cdist(x32, x32, p=2)


def relation_kd_loss(
    student_embs: torch.Tensor,
    teacher_embs: torch.Tensor,
    lambda_dist: float = 1.0,
    lambda_angle: float = 0.5,
) -> torch.Tensor:
    """
    Compare pairwise structure (distance + angle/cosine geometry)
    between student pooled embeddings and teacher pooled embeddings.

    student_embs: [B, H]
    teacher_embs: [B, H]
    returns: scalar loss (tensor)
    """
    # work in float32 for numerical stability
    s = student_embs.float()
    t = teacher_embs.float()

    # distance structure match
    dist_loss = F.mse_loss(_pairwise_dist(s), _pairwise_dist(t))

    # angular / cosine-geometry match
    angle_loss = F.mse_loss(_angle_matrix(s), _angle_matrix(t))

    out = lambda_dist * dist_loss + lambda_angle * angle_loss

    # final safety: kill NaN/Inf
    out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=1e4)
    return out
