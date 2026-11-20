"""Load Qwen, build chat prompts, and locate t_inst and t_post-inst."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Post-instruction span: <|im_end|> \n <|im_start|> assistant \n
POST_INST = [151645, 198, 151644, 77091, 198]


def device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load(name: str, dtype=None, **kw):
    """Return (model, tok). fp16 on GPU (T4 has no bf16), fp32 on CPU. Left padding."""
    dev = device()
    dtype = dtype or (torch.float32 if dev == "cpu" else torch.float16)
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, **kw).to(dev).eval()
    return model, tok


def chat_text(tok, instruction: str) -> str:
    # No system message: Qwen's template inserts its default system prompt (Decision 11).
    return tok.apply_chat_template(
        [{"role": "user", "content": instruction}], tokenize=False, add_generation_prompt=True
    )


def build_chat(tok, instruction: str):
    """Return (input_ids, t_inst, t_post) for one prompt."""
    ids = tok(chat_text(tok, instruction), add_special_tokens=False).input_ids
    t_post = len(ids) - 1
    t_inst = t_post - len(POST_INST)
    assert ids[t_inst + 1:] == POST_INST, "chat template does not end in the post-instruction span"
    return ids, t_inst, t_post


def build_batch(tok, instructions: list[str]):
    """Left-padded batch. Every row ends at the last column, so both positions are shared ints."""
    assert tok.padding_side == "left", "positions are counted from the right; needs left padding"
    enc = tok([chat_text(tok, s) for s in instructions], padding=True, return_tensors="pt",
              add_special_tokens=False)
    t_post = enc.input_ids.shape[1] - 1
    t_inst = t_post - len(POST_INST)
    assert (enc.input_ids[:, t_inst + 1:] == torch.tensor(POST_INST)).all()
    return enc, t_inst, t_post
