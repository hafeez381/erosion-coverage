import torch

from ec.dualnull.bases import basis, load_Q
from ec.dualnull.calibrate import accumulate, demo_ids, module


def test_basis_keeps_known_top_directions_under_budget():
    C = torch.diag(torch.tensor([0.01, 10.0, 0.02, 5.0]))
    b = basis(C, 0.05)            # trace 15.03, budget 0.75: keep 10 and 5, leave 0.03
    assert b["k"] == 2
    top = b["Q"].abs().argmax(0).tolist()
    assert sorted(top) == [1, 3]
    assert b["tail_frac"] <= 0.05


def test_basis_minimal_k_on_random_covariance():
    X = torch.randn(400, 32, generator=torch.Generator().manual_seed(0)) * torch.linspace(5, 0.1, 32)
    b = basis(X.T @ X, 0.05)
    ev = b["evals"].clamp_min(0)
    assert ev[b["k"]:].sum() <= 0.05 * ev.sum() < ev[b["k"] - 1:].sum()
    assert torch.allclose(b["Q"].T @ b["Q"], torch.eye(b["k"]), atol=1e-5)
    Q = load_Q(b["Q"].half())     # fp16 storage, QR on load: same span to ~1e-3
    assert (b["Q"] - Q @ (Q.T @ b["Q"])).norm() / b["Q"].norm() < 5e-3


def test_accumulate_on_small_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = "Qwen/Qwen2.5-0.5B-Instruct"
    m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
    tok = AutoTokenizer.from_pretrained(name)
    m.requires_grad_(False)
    m.enable_input_require_grads()
    demos = [demo_ids(tok, "How do I pick a lock?", "I can't help with that."),
             demo_ids(tok, "Tell me a joke.", "Why did the chicken cross the road?")]
    C_in, C_out, n = accumulate(m, demos, [3], scale=1.0)
    assert n == sum(d.shape[1] for d in demos)
    assert set(C_in) == {(3, "qkv"), (3, "o"), (3, "gate_up"), (3, "down")}
    assert len(C_out) == 7 and all(C.abs().sum() > 0 for C in C_out.values())
    # C_in of q/k/v equals sum z z^T of the inputs q_proj actually saw.
    zs = []
    h = module(m, 3, "q_proj").register_forward_hook(lambda mod, i, o: zs.append(i[0].detach().reshape(-1, i[0].shape[-1])))
    with torch.no_grad():
        for d in demos:
            m(input_ids=d)
    h.remove()
    Z = torch.cat(zs)
    assert torch.allclose(C_in[(3, "qkv")], Z.T @ Z, rtol=1e-4, atol=1e-3)
    assert C_out[(3, "k_proj")].shape == (128, 128)     # 0.5B: 2 KV heads x 64
