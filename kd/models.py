from typing import Optional, List, Union
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

TARGET_MODULES_DEFAULT = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # attention
    "gate_proj", "up_proj", "down_proj"      # MLP
]

def _norm_dtype(d: Union[str, torch.dtype]) -> torch.dtype:
    if isinstance(d, torch.dtype): return d
    m = {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
         "fp16": torch.float16,  "float16": torch.float16,
         "fp32": torch.float32,  "float32": torch.float32}
    return m.get(str(d).lower(), torch.bfloat16)

def load_student(
    model_id: str,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    dtype: Union[str, torch.dtype] = torch.bfloat16,
):
    dtype = _norm_dtype(dtype)

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None and hasattr(tok, "eos_token"):
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,                 # (was torch_dtype=)
        trust_remote_code=True,
    )

    # keep pad ids aligned to avoid warnings during generation
    if getattr(model.config, "pad_token_id", None) is None and tok.pad_token_id is not None:
        model.config.pad_token_id = tok.pad_token_id

    model.gradient_checkpointing_enable()

    tm = target_modules or TARGET_MODULES_DEFAULT
    lcfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=tm,
        task_type=TaskType.CAUSAL_LM,   # correct PEFT task type
    )
    model = get_peft_model(model, lcfg)
    return model, tok
