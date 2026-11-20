"""Memory-light causal-LM loss: lm_head only at answer positions, in checkpointed chunks.

Full-vocabulary logits for batch 4 x 1024 x 151,936 are ~2.5 GB per fp32 copy and ran a T4 out of
memory (C0 kernel v3). Here only `chunk` positions of logits exist at a time, forward or backward.
"""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _chunk_ce(lm_head, h, y):
    return F.cross_entropy(lm_head(h).float(), y, reduction="sum")


def answer_loss(model, input_ids, attention_mask, labels, chunk: int = 1024) -> torch.Tensor:
    """Mean next-token cross-entropy over positions whose label is not -100. Same value as
    model(input_ids, attention_mask, labels=labels).loss. Works on a PEFT-wrapped or plain model."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    h = base.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    h, y = h[:, :-1].reshape(-1, h.shape[-1]), labels[:, 1:].reshape(-1).to(h.device)
    keep = y != -100
    h, y = h[keep], y[keep]
    total = h.new_zeros((), dtype=torch.float32)
    for i in range(0, len(y), chunk):
        total = total + checkpoint(_chunk_ce, base.lm_head, h[i:i + chunk], y[i:i + chunk], use_reentrant=False)
    return total / len(y)
