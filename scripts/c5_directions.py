"""C4 + C5 on the base model: cache activations, fit directions, choose the band, score H and R on every
set, and run the Arditi ablation check.

Kaggle T4 (after job 8):  python scripts/c5_directions.py --labels results/base/labels.csv --artifacts /kaggle/working/artifacts
Committed: configs/directions.yaml, results/directions/{layer_auroc.csv,summary.json,ablation_labels.csv},
results/base/scores.csv. Off-repo (--artifacts, and the Hub when HF_TOKEN is set): caches and directions.pt.
"""
import argparse
import gc
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from ec.eval.generate import generate
from ec.judge.classifiers import judge
from ec.model import load
from ec.repr.ablate import ablated
from ec.repr.cache import cache
from ec.repr.directions import choose_band, fit, layer_auroc, layer_scores
from ec.repr.scores import T_INST, T_POST
from ec.seed import SEED, seed_everything

SETS = ["generic", "mitre", "frr", "cysecbench", "xstest_safe"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--labels", default="results/base/labels.csv", help="job 8 base labels (C2a)")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--hub-repo", default="hafeez381/ec-artifacts")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--skip-ablation", action="store_true")
    a = ap.parse_args()
    seed_everything()
    art, res = Path(a.artifacts), Path("results/directions")
    art.mkdir(parents=True, exist_ok=True)
    res.mkdir(parents=True, exist_ok=True)

    prompts = {s: pd.read_json(f"prompts/{s}.jsonl", lines=True) for s in SETS}
    lab = pd.read_csv(a.labels).set_index("id")["refusal"]

    # C4: cache every set once.
    model, tok = load(a.model)
    caches = {}
    for s, df in prompts.items():
        caches[s] = cache(model, tok, df["text"].tolist(), df["id"].tolist(), a.batch)
        torch.save(caches[s], art / f"cache_base_{s}.pt")
        print(f"cached {s}: {tuple(caches[s]['h'].shape)}", flush=True)

    # C5: behaviour filter (refused harmful, answered harmless), fit on 'fit', evaluate on 'heldout'.
    g = prompts["generic"].assign(refused=lambda d: d["id"].map(lab))
    h = caches["generic"]["h"]
    sel = lambda st, sp, ref: torch.tensor(((g["set"] == st) & (g["split"] == sp) & (g["refused"] == ref)).values)
    harm_fit, harmless_fit = sel("generic_harmful", "fit", True), sel("generic_harmless", "fit", False)
    harm_ho = torch.tensor(((g["set"] == "generic_harmful") & (g["split"] == "heldout")).values)
    harmless_ho = torch.tensor(((g["set"] == "generic_harmless") & (g["split"] == "heldout")).values)
    d = fit(h[harm_fit], h[harmless_fit])
    auc_inst = layer_auroc(layer_scores(h[harm_ho], d, T_INST), layer_scores(h[harmless_ho], d, T_INST))
    auc_post = layer_auroc(layer_scores(h[harm_ho], d, T_POST), layer_scores(h[harmless_ho], d, T_POST))
    band = choose_band(auc_inst)
    cos = F.cosine_similarity(d["v_harm"], d["v_ref"], dim=-1)
    pd.DataFrame({"layer": np.arange(1, len(auc_inst) + 1), "auroc_t_inst": auc_inst, "auroc_t_post": auc_post,
                  "cos_vharm_vref": cos.numpy()}).to_csv(res / "layer_auroc.csv", index=False)
    Path("configs").mkdir(exist_ok=True)
    Path("configs/directions.yaml").write_text(
        "# Pre-registered (Decision 6). Cache indices: index l is decoder block l+1.\n"
        f"band: [{', '.join(map(str, band))}]\npositions: [t_inst, t_post_inst]\n")
    torch.save(d | {"band": band}, art / "directions.pt")

    # H and R on every set (Eq. 1), base model.
    scores = []
    for s, c in caches.items():
        H = layer_scores(c["h"], d, T_INST)[:, band].mean(1)
        R = layer_scores(c["h"], d, T_POST)[:, band].mean(1)
        scores.append(pd.DataFrame({"id": c["ids"], "H": H.numpy(), "R": R.numpy()}))
    scores = pd.concat(scores)
    Path("results/base").mkdir(parents=True, exist_ok=True)
    scores.to_csv("results/base/scores.csv", index=False)
    sg = scores.set_index("id").loc[g["id"]]
    H_auc = roc_auc_score(np.r_[np.ones(harm_ho.sum()), np.zeros(harmless_ho.sum())],
                          np.r_[sg["H"].values[harm_ho.numpy()], sg["H"].values[harmless_ho.numpy()]])
    summary = {"n_fit": {"harmful_refused": int(harm_fit.sum()), "harmless_answered": int(harmless_fit.sum())},
               "band_cache_idx": band, "band_layers": [b + 1 for b in band],
               "heldout_auroc_H": round(float(H_auc), 4), "max_layer_auroc_t_inst": round(float(auc_inst.max()), 4),
               "cos_vharm_vref_band_mean": round(float(cos[band].mean()), 4)}

    # Ablation check: v_ref from the band layer with the highest t_post AUROC, plus a random control.
    if not a.skip_ablation:
        best = band[int(np.argmax(auc_post[band]))]
        v_rand = F.normalize(torch.randn(d["v_ref"].shape[1], generator=torch.Generator().manual_seed(SEED)), dim=0)
        ho_h = g[harm_ho.numpy()]
        ho_n = g[harmless_ho.numpy()]
        runs = {}
        for name, v, df in (("vref_harmful", d["v_ref"][best], ho_h), ("random_harmful", v_rand, ho_h),
                            ("vref_harmless", d["v_ref"][best], ho_n)):
            with ablated(model, v):
                runs[name] = (df, generate(model, tok, df["text"].tolist(), 384, a.batch))
        del model
        gc.collect()
        torch.cuda.empty_cache()
        rows = []
        for name, (df, resp) in runs.items():
            j = judge(df["id"].tolist(), df["text"].tolist(), resp, actionable=False)
            rows.append(j.assign(run=name)[["id", "run", "refusal"]])
        ab = pd.concat(rows)
        ab.to_csv(res / "ablation_labels.csv", index=False)
        base_ref = lambda df: float(df["id"].map(lab).astype(bool).mean())
        rate = ab.groupby("run")["refusal"].apply(lambda r: float(r.fillna(False).astype(bool).mean())).to_dict()
        summary["ablation"] = {"layer": best + 1, "base_harmful_refusal": base_ref(ho_h),
                               "base_harmless_refusal": base_ref(ho_n), **{f"{k}_refusal": v for k, v in rate.items()}}

    (res / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)

    if os.environ.get("HF_TOKEN"):
        from ec.train.checkpoint import push
        for p in sorted(art.glob("*.pt")):
            print("pushed", p.name, push(str(p), a.hub_repo, f"base/{p.name}"), flush=True)


if __name__ == "__main__":
    main()
