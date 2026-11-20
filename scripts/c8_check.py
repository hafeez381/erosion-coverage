"""C8 on the 3B model: step-0 equivalence, constraint residual, fidelity and step time, Dual-Null vs plain LoRA.

Kaggle T4:  python scripts/c8_check.py --bases /kaggle/working/bases
Committed: results/dualnull_check/{summary.json,fidelity.csv}. Nothing else is kept.
"""
import argparse
import gc
import json
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import snapshot_download
from peft import LoraConfig, get_peft_model

from ec.data.beavertails import load_split
from ec.dualnull.calibrate import MODULES, demo_ids
from ec.dualnull.wrapper import DualNullLoRA, fidelity, load_bases
from ec.model import chat_text, load
from ec.seed import seed_everything
from ec.train.loss import answer_loss

CUDA = torch.cuda.is_available()
TRENDYOL = "Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset"
LAYERS_REPORTED = [1, 9, 18, 27, 35]  # trimmed to the model depth below          # cache-style indices, a spread through the network


def sft_batch(tok, rows, max_len):
    prompts = [chat_text(tok, u) for u, _ in rows]
    full = [p + a + tok.eos_token for p, (_, a) in zip(prompts, rows)]
    enc = tok(full, padding=True, truncation=True, max_length=max_len, return_tensors="pt", add_special_tokens=False)
    labels = enc.input_ids.clone()
    labels[enc.attention_mask == 0] = -100
    for i, p in enumerate(prompts):
        pad = int((enc.attention_mask[i] == 0).sum())
        labels[i, : pad + len(tok(p, add_special_tokens=False).input_ids)] = -100
    return enc, labels


def wrap(model):
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=1, lora_dropout=0.0, target_modules=MODULES,
                                             init_lora_weights=False))
    return model


def train(model, tok, data, steps, batch, max_len, after_step=None, every=100, probe=None):
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-5, weight_decay=0.01)
    log, times = [], []
    for step in range(1, steps + 1):
        rows = data[(step - 1) * batch: step * batch]
        enc, labels = sft_batch(tok, rows, max_len)
        enc = enc.to(model.device)
        if CUDA:
            torch.cuda.synchronize()
        t = time.time()
        loss = answer_loss(model, enc["input_ids"], enc["attention_mask"], labels)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if after_step:
            after_step()
        if CUDA:
            torch.cuda.synchronize()
        times.append(time.time() - t)
        if step % every == 0 and probe:
            model.eval()
            log.append({"step": step, "loss": float(loss), **probe()})
            model.train()
            print(log[-1] | {"s_per_step": round(sum(times[-every:]) / every, 2)}, flush=True)
    return log, times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--bases", default="artifacts/bases_generic")
    ap.add_argument("--hub-repo", default="hafeez381/ec-artifacts")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=512)
    a = ap.parse_args()
    seed_everything()
    out = Path("results/dualnull_check")
    out.mkdir(parents=True, exist_ok=True)
    if not Path(a.bases).exists():
        snapshot_download(a.hub_repo, allow_patterns="calib/generic/*.pt", local_dir=a.bases + "_dl")
        a.bases = a.bases + "_dl/calib/generic"

    # Data: 32 evaluation prompts, held-out and calibration BeaverTails demos, Trendyol training rows.
    gen = pd.read_json("prompts/generic.jsonl", lines=True).query("split == 'heldout'")
    eval_texts = pd.concat([gen.query("set == 'generic_harmful'").head(16), gen.query("set == 'generic_harmless'").head(16)])["text"].tolist()
    bt = pd.read_json("prompts/beavertails_calib.jsonl", lines=True)
    train_bt = load_split("30k_train").set_index("row")
    ty_rows = pd.read_json("prompts/trendyol_candidates.jsonl", lines=True)["row"].tolist()[: a.steps * a.batch]
    ty = load_dataset(TRENDYOL, split="train").to_pandas()
    data = [(ty.loc[r, "user"], ty.loc[r, "assistant"]) for r in ty_rows]

    model, tok = load(a.model)
    held = [demo_ids(tok, p, train_bt.loc[r, "response"]) for p, r in bt.query("split == 'heldout'").head(64)[["text", "row"]].values]
    calib = [demo_ids(tok, p, train_bt.loc[r, "response"]) for p, r in bt.query("split == 'calib'").head(64)[["text", "row"]].values]
    eval_ids = [tok(chat_text(tok, t), return_tensors="pt", add_special_tokens=False).input_ids for t in eval_texts]
    with torch.no_grad():
        ref = [model(input_ids=i.to(model.device)).logits[0].float().cpu() for i in eval_ids]
    L = len(model.model.layers)
    keys = [(l, m) for l in LAYERS_REPORTED if l < L for m in MODULES]
    summary = {"steps": a.steps, "batch": a.batch, "max_len": a.max_len}
    rows = []

    def fid(pm, tag, step, sets):
        for name, b in sets.items():
            for k, v in fidelity(pm, keys, b).items():
                rows.append({"run": tag, "step": step, "tokens": name, "layer": k[0] + 1, "module": k[1], **v})
        d = pd.DataFrame([r for r in rows if r["run"] == tag and r["step"] == step])
        return {f"fid_{n}": round(float(d[d.tokens == n]["unweighted"].mean()), 4) for n in sets}

    # A. Dual-Null.
    pm = wrap(model)
    bases = load_bases(a.bases, L, pm.device)
    dn = DualNullLoRA(pm, bases, scale=1 / 8)
    t = time.time()
    dn.init_balanced_svd_()
    summary["init_s"] = round(time.time() - t, 1)
    pm.eval()
    with torch.no_grad():
        rel, agree, kl_mean, kl_max = [], [], [], []
        for i, r in zip(eval_ids, ref):
            o = pm(input_ids=i.to(pm.device)).logits[0].float().cpu()
            rel.append(float((o - r).abs().max() / r.abs().max()))
            agree.append(float((o.argmax(-1) == r.argmax(-1)).float().mean()))
            kl = F.kl_div(F.log_softmax(o, -1), F.log_softmax(r, -1), log_target=True, reduction="none").sum(-1)
            kl_mean.append(float(kl.mean()))
            kl_max.append(float(kl.max()))
    summary["step0"] = {"max_rel_logit_diff": max(rel), "greedy_agreement_min": min(agree),
                        "kl_mean": sum(kl_mean) / len(kl_mean), "kl_max": max(kl_max)}
    summary["dualnull_residual_init"] = dn.residual()
    print(summary, flush=True)
    sets = {"heldout": held, "calib": calib}
    summary["dualnull_fid_step0"] = fid(pm, "dualnull", 0, sets)
    dn.add_grad_hooks()
    log, times = train(pm, tok, data, a.steps, a.batch, a.max_len, after_step=dn.project_, probe=None)
    pm.eval()
    summary["dualnull_fid_end"] = fid(pm, "dualnull", a.steps, sets)
    summary["dualnull_residual_end"] = dn.residual()
    summary["dualnull_s_per_step"] = round(sum(times[5:]) / len(times[5:]), 3)
    summary["peak_gb_dualnull"] = round(torch.cuda.max_memory_allocated() / 2**30, 2) if CUDA else 0.0
    print(summary, flush=True)
    del pm, model, dn, bases
    gc.collect()
    if CUDA:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # B. Plain LoRA, same data and steps (standard init: B = 0).
    model, tok = load(a.model)
    pm = get_peft_model(model, LoraConfig(r=8, lora_alpha=1, lora_dropout=0.0, target_modules=MODULES))
    pm.eval()
    summary["plain_fid_step0"] = fid(pm, "plain", 0, sets)
    log, times = train(pm, tok, data, a.steps, a.batch, a.max_len)
    pm.eval()
    summary["plain_fid_end"] = fid(pm, "plain", a.steps, sets)
    summary["plain_s_per_step"] = round(sum(times[5:]) / len(times[5:]), 3)
    summary["step_time_ratio"] = round(summary["dualnull_s_per_step"] / summary["plain_s_per_step"], 3)
    summary["peak_gb_plain"] = round(torch.cuda.max_memory_allocated() / 2**30, 2) if CUDA else 0.0

    fd = pd.DataFrame(rows)
    # On calibration tokens the weighted ratio is bounded by the trailing mass, so <= b for every module.
    summary["calib_weighted_sq_max_dualnull"] = float(fd[(fd.run == "dualnull") & (fd.tokens == "calib")]["weighted_sq"].max())
    fd.to_csv(out / "fidelity.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
