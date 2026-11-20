import torch

from ec.model import build_batch


@torch.inference_mode()
def generate(model, tok, texts, max_new_tokens=384, batch=16):
    """Greedy responses (Decision 11). Sorted by length so padding stays small; returned in input order."""
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = [None] * len(texts)
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        enc, _, _ = build_batch(tok, [texts[i] for i in idx])
        g = model.generate(**enc.to(model.device), max_new_tokens=max_new_tokens, do_sample=False,
                           pad_token_id=tok.pad_token_id)
        for i, r in zip(idx, tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)):
            out[i] = r
    return out
