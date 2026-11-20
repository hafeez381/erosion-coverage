"""C7: Dual-Null generic calibration bases for every adapted projection of the base model.

Kaggle T4:  python scripts/c7_calibrate.py --artifacts /kaggle/working/artifacts
Committed: results/calibration/{k_prot.csv,summary.json}, configs/calibration.yaml.
Off-repo (--artifacts, and the Hub when HF_TOKEN is set): bases/layer_XX.pt with fp16 Q and fp32 eigenvalues.
Covariances are not saved (about 54 GB in fp32); C7b recomputes the generic ones beside the domain ones.
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import pandas as pd
import torch

from ec.data.beavertails import load_split
from ec.dualnull.bases import basis
from ec.dualnull.calibrate import IN_GROUP, MODULES, accumulate, demo_ids, grad_max
from ec.model import load
from ec.seed import seed_everything


def peak_gb():
    return torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--hub-repo", default="hafeez381/ec-artifacts")
    ap.add_argument("--budget", type=float, default=0.05)
    ap.add_argument("--layers-per-pass", type=int, default=3)
    ap.add_argument("--n-demos", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--layers", type=int, nargs="*", help="subset of layer indices (smoke tests)")
    a = ap.parse_args()
    seed_everything()
    t0 = time.time()
    out = Path(a.artifacts) / "bases_generic"
    out.mkdir(parents=True, exist_ok=True)
    res = Path("results/calibration")
    res.mkdir(parents=True, exist_ok=True)

    # Decision 2: the first 256 is_safe BeaverTails rows, responses read back by row.
    calib = pd.read_json("prompts/beavertails_calib.jsonl", lines=True).query("split == 'calib'").iloc[: a.n_demos]
    train = load_split("30k_train").set_index("row")
    model, tok = load(a.model)
    model.requires_grad_(False)
    model.enable_input_require_grads()
    demos = [demo_ids(tok, p, train.loc[r, "response"], a.max_len) for p, r in zip(calib["text"], calib["row"])]
    n_tok = sum(d.shape[1] for d in demos)
    print(f"{len(demos)} demos, {n_tok} tokens, max {max(d.shape[1] for d in demos)}", flush=True)

    # fp16 guard: the largest |g| at scale 1 sets the scale, power of two, headroom 4x below fp16 max.
    gmax = grad_max(model, demos)
    assert math.isfinite(gmax) and gmax > 0, f"gradients overflow fp16 even at scale 1 (max {gmax})"
    scale = float(min(1024, 2 ** math.floor(math.log2(16000 / gmax))))
    print(f"max |g| at scale 1: {gmax:.3g}, loss scale {scale}", flush=True)

    L = len(model.model.layers)
    layers = a.layers if a.layers else list(range(L))
    rows, timings = [], []
    for c in range(0, len(layers), a.layers_per_pass):
        chunk = layers[c:c + a.layers_per_pass]
        t = time.time()
        C_in, C_out, _ = accumulate(model, demos, chunk, scale)
        t_acc = time.time() - t
        for l in chunk:
            saved = {"in": {}, "out": {}}
            for side, store, keys in (("in", C_in, sorted({IN_GROUP[m] for m in MODULES})), ("out", C_out, MODULES)):
                for key in keys:
                    C = store.pop((l, key))
                    te = time.time()
                    bz = basis(C, a.budget)
                    timings.append((C.shape[0], time.time() - te))
                    del C
                    saved[side][key] = {"Q": bz["Q"].half().cpu(), "evals": bz["evals"].cpu(), "k": bz["k"]}
                    rows.append({"layer": l + 1, "side": side, "module": key, "d": bz["Q"].shape[0], "k": bz["k"],
                                 "k_frac": round(bz["k"] / bz["Q"].shape[0], 4), "tail_frac": round(bz["tail_frac"], 5),
                                 "neg_ratio": bz["neg_ratio"]})
                    torch.cuda.empty_cache()
            torch.save(saved | {"budget": a.budget, "loss_scale": scale}, out / f"layer_{l:02d}.pt")
        print(f"layers {[l + 1 for l in chunk]}: accumulate {t_acc:.0f}s, total {time.time() - t0:.0f}s, "
              f"peak {peak_gb():.1f} GB", flush=True)
        pd.DataFrame(rows).to_csv(res / "k_prot.csv", index=False)   # partial table survives a crash

    k = pd.DataFrame(rows)
    big = [s for d, s in timings if d > 4096]
    summary = {"n_demos": len(demos), "n_tokens": n_tok, "loss_scale": scale, "grad_max_scale1": gmax,
               "budget": a.budget, "layers": [l + 1 for l in layers], "wall_s": round(time.time() - t0),
               "peak_gb": round(peak_gb(), 2),
               "eigh_s_11008": round(sum(big) / len(big), 1) if big else None,
               "min_neg_ratio": float(k["neg_ratio"].min()), "max_tail_frac": float(k["tail_frac"].max()),
               "k_by_module": k.groupby(["side", "module"])["k"].describe()[["min", "50%", "max"]].round(0).reset_index()
               .to_dict("records")}
    (res / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    Path("configs").mkdir(exist_ok=True)
    Path("configs/calibration.yaml").write_text(
        "# Decision 2, 4, 5. Generic calibration (C7). Bases on the Hub under calib/generic/.\n"
        f"demos: beavertails_calib split=calib first {len(demos)}\nmax_len: {a.max_len}\nloss: sum over all tokens\n"
        f"loss_scale: {scale}\nbudget_b: {a.budget}\nn_tokens_generic: {n_tok}\nQ_storage: fp16, QR on load\n")
    print(json.dumps({k_: v for k_, v in summary.items() if k_ != "k_by_module"}, indent=2), flush=True)
    print(k.groupby(["side", "module"])["k"].describe().round(0).to_string(), flush=True)

    if os.environ.get("HF_TOKEN"):
        from ec.train.checkpoint import push
        for p in sorted(out.glob("*.pt")):
            print("pushed", p.name, push(str(p), a.hub_repo, f"calib/generic/{p.name}"), flush=True)


if __name__ == "__main__":
    main()
