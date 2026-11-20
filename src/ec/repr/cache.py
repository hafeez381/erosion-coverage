import torch

from ec.model import build_batch


@torch.inference_mode()
def cache(model, tok, texts: list[str], ids: list | None = None, batch: int = 16) -> dict:
    """Residual stream at (t_inst, t_post) for layers 1..L: {"ids", "h": (N, L, 2, d) fp16}.

    hidden_states[0] is the embeddings and is dropped. The last layer has the final RMSNorm
    applied, so treat it differently from the others.
    """
    out = []
    for i in range(0, len(texts), batch):
        enc, t_inst, t_post = build_batch(tok, texts[i:i + batch])
        hs = model(**enc.to(model.device), output_hidden_states=True).hidden_states[1:]
        out.append(torch.stack([h[:, [t_inst, t_post]] for h in hs], 1).half().cpu())
    return {"ids": ids if ids is not None else list(range(len(texts))), "h": torch.cat(out)}
