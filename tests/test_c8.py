import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from ec.dualnull.calibrate import MODULES
from ec.dualnull.project import random_basis
from ec.dualnull.wrapper import DualNullLoRA, fidelity, proj_cols, proj_rows

NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def test_projection_converges_with_fp16_basis():
    g = torch.Generator().manual_seed(0)
    Q = random_basis(4864, 1500, g).half()             # fp16 columns are not exactly orthonormal
    A = torch.randn(8, 4864, generator=g)
    for passes, bound in ((1, 1e-1), (3, 1e-5)):
        X = proj_rows(A, Q, passes)
        assert (X @ Q.float()).norm() / X.norm() < bound
    B = torch.randn(4864, 8, generator=g)
    assert (Q.float().T @ proj_cols(B, Q)).norm() / B.norm() < 1e-5


def _setup(layers=(2, 5)):
    m = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32).eval()
    tok = AutoTokenizer.from_pretrained(NAME)
    ids = tok("Explain how a firewall works.", return_tensors="pt").input_ids
    with torch.no_grad():
        ref = m(input_ids=ids).logits
    pm = get_peft_model(m, LoraConfig(r=8, lora_alpha=1, lora_dropout=0.0, target_modules=MODULES,
                                      init_lora_weights=False))
    g = torch.Generator().manual_seed(1)
    cfg = m.config
    dims = {"q_proj": (cfg.hidden_size, cfg.hidden_size), "k_proj": (cfg.hidden_size, 128), "v_proj": (cfg.hidden_size, 128),
            "o_proj": (cfg.hidden_size, cfg.hidden_size), "gate_proj": (cfg.hidden_size, cfg.intermediate_size),
            "up_proj": (cfg.hidden_size, cfg.intermediate_size), "down_proj": (cfg.intermediate_size, cfg.hidden_size)}
    bases = {}
    for l in layers:
        for mname in MODULES:
            din, dout = dims[mname]
            bases[(l, mname)] = (random_basis(din, din // 4, g).half(), random_basis(dout, dout // 8, g).half())
    return pm, tok, ids, ref, bases


def test_step0_equivalence_constraint_and_fidelity():
    pm, tok, ids, ref, bases = _setup()
    dn = DualNullLoRA(pm, bases, scale=1 / 8)
    # zero every non-Dual-Null adapter so only wrapped modules contribute
    with torch.no_grad():
        for n, p in pm.named_parameters():
            if "lora_B" in n:
                p.zero_()
    dn.init_balanced_svd_()
    with torch.no_grad():
        out = pm(input_ids=ids).logits
    assert (out - ref).abs().max() / ref.abs().max() < 1e-4
    assert (out.argmax(-1) == ref.argmax(-1)).all()
    ra, rb = dn.residual()
    assert ra < 1e-5 and rb < 1e-5

    dn.add_grad_hooks()
    params = [dn.A(k) for k in dn.mods] + [dn.B(k) for k in dn.mods]
    opt = torch.optim.AdamW(params, lr=1e-2, weight_decay=0.1)
    for _ in range(2):
        pm(input_ids=ids, labels=ids).loss.backward()
        k0 = next(iter(dn.mods))
        g = dn.A(k0).grad
        assert (g.float() @ bases[k0][0].float()).norm() / g.norm() < 1e-5   # gradient projected
        opt.step()
        opt.zero_grad()
        dn.project_()
    ra, rb = dn.residual()
    assert ra < 1e-5 and rb < 1e-5

    # Fidelity: a protected direction is invisible to A; an arbitrary input is not.
    k = (2, "down_proj")
    Qi = bases[k][0].float()
    A = dn.A(k).float()
    assert (A @ Qi[:, 0]).norm() / torch.linalg.matrix_norm(A, 2) < 1e-4
    f = fidelity(pm, [k, (2, "q_proj")], [ids])
    assert 0 < f[k]["unweighted"] < 1 and 0 <= f[k]["weighted_sq"] <= 1
