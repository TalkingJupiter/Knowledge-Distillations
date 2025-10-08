# kd/kd_fb.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x):
        return self.proj(x)

def feature_kd_loss(student_feats: torch.Tensor,
                    teacher_feats: torch.Tensor,
                    token_mask: torch.Tensor) -> torch.Tensor:
    """
    student_feats: [B, T, Ht] (already projected)
    teacher_feats: [B, T, Ht]
    token_mask   : [B, T] (1 for valid tokens)
    Robust to dtype and avoids boolean indexing shape errors.
    """
    # dtype alignment (avoid Float↔BF16 issues)
    if teacher_feats.dtype != student_feats.dtype:
        teacher_feats = teacher_feats.to(student_feats.dtype)

    # mean squared error per token, per feature
    diff = student_feats - teacher_feats                          # [B,T,H]
    mse_per_feat = diff.pow(2)                                    # [B,T,H]

    # weights from mask
    w = token_mask.to(student_feats.dtype).unsqueeze(-1)          # [B,T,1]

    # weighted mean over batch, time, features
    num = (mse_per_feat * w).sum()
    den = w.sum() * student_feats.size(-1) + 1e-8                 # count tokens * H
    return num / den
