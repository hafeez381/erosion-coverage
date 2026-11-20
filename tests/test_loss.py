import torch
from transformers import AutoModelForCausalLM

from ec.train.loss import answer_loss


def test_answer_loss_matches_hf_loss():
    torch.manual_seed(0)
    m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.float32).eval()
    ids = torch.randint(0, 1000, (2, 40))
    mask = torch.ones_like(ids)
    mask[0, :5] = 0                                  # left padding on row 0
    labels = ids.clone()
    labels[mask == 0] = -100
    labels[:, :20] = -100                            # prompt positions
    ref = m(input_ids=ids, attention_mask=mask, labels=labels).loss
    got = answer_loss(m, ids, mask, labels, chunk=7)  # chunk smaller than the answer span
    assert torch.allclose(got, ref, atol=1e-4), (got.item(), ref.item())
