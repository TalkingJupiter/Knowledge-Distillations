import torch
import torch.nn.functional as F

def response_kd_loss(student_logits, teacher_topk_ids, teacher_topk_logptobs, T: float = 2.0):
    s_top = torch.gather(student_logits, dim=-1, index=teacher_topk_ids)
    s_logp_T = F.log_softmax(s_top / T, dim=-1)
    t_prob_T = F.softmax(teacher_topk_logptobs / T, dim=-1)
    loss = F.kl_div(s_logp_T, t_prob_T, reduction="batchmean") * (T**2)
    return loss