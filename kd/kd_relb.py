import torch
import torch.nn.functional as F

def _pairwise_dist(x: torch.Tensor) -> torch.Tensor:
    # upcast to fp32 so cdist works on CUDA
    x32 = x.float()
    return torch.cdist(x32, x32, p=2)  # (B, B)

def _angle_matrix(x: torch.Tensor) -> torch.Tensor:
    # cosine-similarity Gram matrix (B, B)
    x32 = F.normalize(x.float(), dim=-1)
    return x32 @ x32.t()

def relation_kd_loss(
    student_embs: torch.Tensor,
    teacher_embs: torch.Tensor,
    lambda_dist: float = 1.0,
    lambda_angle: float = 0.5,
) -> torch.Tensor:
    # normalize for angle term only; keep originals for distances
    s = student_embs
    t = teacher_embs

    dist_loss  = F.mse_loss(_pairwise_dist(s), _pairwise_dist(t))
    angle_loss = F.mse_loss(_angle_matrix(s), _angle_matrix(t))

    return lambda_dist * dist_loss + lambda_angle * angle_loss
