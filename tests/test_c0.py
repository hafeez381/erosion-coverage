"""Fast C0 checks. The tokenizer test downloads only Qwen2.5-0.5B-Instruct's tokenizer (~7 MB)."""
import pytest
import torch

from ec.dualnull.project import project_cols_, project_rows_, random_basis
from ec.judge.stub import is_refusal
from ec.repr.scores import score

TOK = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    t = AutoTokenizer.from_pretrained(TOK)
    t.padding_side = "left"
    return t


def test_positions_single_and_batched(tok):
    from ec.model import build_batch, build_chat
    prompts = ["Hi", "Write a long paragraph about the history of the printing press in Europe."]
    for p in prompts:
        ids, t_inst, t_post = build_chat(tok, p)
        assert ids[t_post] == 198 and ids[t_inst + 1] == 151645
    enc, t_inst, t_post = build_batch(tok, prompts)
    assert (enc.input_ids[:, t_post] == 198).all() and (enc.input_ids[:, t_inst + 1] == 151645).all()
    # The short prompt is padded on the left, so its last instruction token matches the unpadded one.
    ids, ti, _ = build_chat(tok, prompts[0])
    assert enc.input_ids[0, t_inst] == ids[ti]


def test_projection_removes_the_basis():
    g = torch.Generator().manual_seed(0)
    Q = random_basis(64, 16, g)
    A, B = torch.randn(8, 64, generator=g), torch.randn(64, 8, generator=g)
    project_rows_(A, Q)
    project_cols_(B, Q)
    assert (A @ Q).norm() / A.norm() < 1e-6
    assert (Q.T @ B).norm() / B.norm() < 1e-6


def test_score_sign():
    mu_pos, mu_neg = torch.zeros(4, 2, 3), torch.zeros(4, 2, 3)
    mu_pos[..., 0], mu_neg[..., 1] = 1, 1
    h = torch.zeros(2, 4, 2, 3)
    h[0, ..., 0], h[1, ..., 1] = 1, 1
    s = score(h, mu_pos, mu_neg, [1, 2], 0)
    assert s.tolist() == [1.0, -1.0]


def test_refusal_regex():
    assert is_refusal("I'm sorry, but I can't help with that.")
    assert is_refusal("I cannot provide instructions for this.")
    assert not is_refusal("Sure! Here are three tips for staying healthy.")
    assert not is_refusal("Port scanning lists open ports. " + "x" * 400 + " I can't stress this enough.")
