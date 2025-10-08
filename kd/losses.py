import torch
import torch.nn.functional as F

def ce_loss(logits, labels, ignore_index=-100):
    return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=ignore_index)