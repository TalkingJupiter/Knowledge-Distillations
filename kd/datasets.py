from typing import Dict, List, Optional
import glob
import torch
from torch.utils.data import IterableDataset
import pyarrow.parquet as pq

def _iter_parquet_rows(path_glob: str, columns: Optional[List[str]] = None):
    files = sorted(glob.glob(path_glob))
    assert files, f"No Parquet files found for {path_glob}"
    for f in files:
        pf = pq.ParquetFile(f)
        for rg in range(pf.num_row_groups):
            batch = pf.read_row_group(rg, columns=columns).to_pydict()
            n = len(next(iter(batch.values()))) if batch else 0
            for i in range(n):
                yield {k: batch[k][i] for k in batch.keys()}

class RBTopKIterableDataset(IterableDataset):
    def __init__(self, path_glob: str):
        super().__init__()
        self.path_glob = path_glob
    def __iter__(self):
        for row in _iter_parquet_rows(self.path_glob, columns=["input_ids", "attn_mask", "topk_ids", "topk_logprobs"]):
            yield {k: torch.tensor(row[k]) for k in row}

class FBDataset(IterableDataset):
    def __init__(self, path_glob: str, teacher_layer: int):
        super().__init__()
        self.path_glob = path_glob
        self.col = f"hidden_L{teacher_layer}"
    def __iter__(self):
        cols = ["input_ids", "attn_mask", self.col]
        for row in _iter_parquet_rows(self.path_glob, columns=cols):
            yield {
                "input_ids": torch.tensor(row["input_ids"], dtype=torch.long),
                "attn_mask": torch.tensor(row["attn_mask"], dtype=torch.long),
                "teacher_feats": torch.tensor(row[self.col], dtype=torch.float32),
            }

class RelBDataset(IterableDataset):
    def __init__(self, path_glob: str):
        super().__init__()
        self.path_glob = path_glob
    
    def __iter__(self):
        cols = ["input_ids", "attn_mask", "pooled_embedding"]
        for row in _iter_parquet_rows(self.path_glob, columns=cols):
            yield {
                "input_ids": torch.tensor(row["input_ids"], dtype=torch.long),
                "attn_mask": torch.tensor(row["attn_mask"], dtype=torch.long),
                "teacher_embed": torch.tensor(row["pooled_embedding"], dtype=torch.float32)
            }

# def collate_pad(batch: List[Dict]):
#     max_len = max(x["input_ids"].size(0) for x in batch)
#     pad_id = 0
#     B = len(batch)
#     input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
#     attn_mask = torch.zeros((B, max_len), dtype=torch.long)
#     for i, ex in enumerate(batch):
#         L = ex["input_ids"].size(0)
#         input_ids[i, :L] = ex["input_ids"]
#         attn_mask[i, :L] = ex["attn_mask"]
#     out = {"input_ids": input_ids, "attn_mask": attn_mask}
#     if "teacher_embed" in batch[0]:
#         d = batch[0]["teacher_feats"].size(-1)
#         teacher = torch.zeros((B, max_len, d), dtype=batch[0]["teacher_feats"].dtype)
#         for i, ex in enumerate(batch):
#             L = ex["input_ids"].size(0)
#             teacher[i, :L, :] = ex["teacher_feats"]
#         out["teacher_feats"] = teacher
#     if "teacher_embed" in batch[0]:
#         teacher_embs = torch.stack([ex["teacher_embed"] for ex in batch], dim=0)
#         out["teacher_embed"] = teacher_embs
#     return out

def collate_pad(batch: List[Dict]):
    max_len = max(x["input_ids"].size(0) for x in batch)
    pad_id = 0
    B = len(batch)
    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    attn_mask = torch.zeros((B, max_len), dtype=torch.long)
    for i, ex in enumerate(batch):
        L = ex["input_ids"].size(0)
        input_ids[i, :L] = ex["input_ids"]
        attn_mask[i, :L] = ex["attn_mask"]
    out = {"input_ids": input_ids, "attn_mask": attn_mask}
    if "teacher_feats" in batch[0]:
        d = batch[0]["teacher_feats"].size(-1)
        teacher = torch.zeros((B, max_len, d), dtype=batch[0]["teacher_feats"].dtype)
        for i, ex in enumerate(batch):
            L = ex["input_ids"].size(0)
            teacher[i, :L, :] = ex["teacher_feats"]
        out["teacher_feats"] = teacher
    if "teacher_embed" in batch[0]:
        teacher_embs = torch.stack([ex["teacher_embed"] for ex in batch], dim=0)
        out["teacher_embed"] = teacher_embs
    return out

def collate_rb(batch: List[Dict]):
    max_len = max(x["input_ids"].size(0) for x in batch)
    k = batch[0]["topk_ids"].size(-1)
    pad_id = 0
    B = len(batch)
    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    attn_mask = torch.zeros((B, max_len), dtype=torch.long)
    topk_ids = torch.zeros((B, max_len-1, k), dtype=torch.long)
    topk_logprobs = torch.zeros((B, max_len-1,k), dtype=torch.float32)
    for i, ex in enumerate(batch):
        L = ex["input_ids"].size(0)
        input_ids[i, :L] = ex["input_ids"]
        attn_mask[i, :L] = ex["attn_mask"]
        eff = max(L-1, 0)
        topk_ids[i, :eff, :] = ex["topk_ids"]
        topk_logprobs[i, :eff, :] = ex["topk_logprobs"]
    return {"input_ids": input_ids, "attn_mask": attn_mask, "topk_ids": topk_ids, "topk_logprobs": topk_logprobs}