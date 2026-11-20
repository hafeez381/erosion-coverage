"""C1: build every prompt set once, with stable ids, leak-free splits and length matching.

python scripts/c1_build_prompts.py        (CPU only; downloads ~200 MB of datasets)
Writes prompts/*.jsonl, prompts/MANIFEST.json and prompts/BUILD_REPORT.json, then runs the C1 checks.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer

from ec.data import beavertails, cyber, generic, splits, trendyol
from ec.seed import SEED, seed_everything

OUT = Path("prompts")
TOKENIZER = "Qwen/Qwen2.5-3B-Instruct"
BIN = 32          # Decision 9
MAX_LEN = 1024    # guide §5


def length_auroc(h, b) -> float:
    """AUROC of token length alone as a harmful-vs-benign score, folded to >= 0.5."""
    a = roc_auc_score([1] * len(h) + [0] * len(b), pd.concat([h, b]))
    return round(max(a, 1 - a), 4)


def frame(df, set_name, source, id_fmt, split="test", category=None):
    out = pd.DataFrame({"id": [id_fmt.format(r) for r in df["row"]] if "id" not in df else df["id"],
                        "set": set_name, "split": split, "text": df["text"].values, "source": source,
                        "category": category if category is not None else (df["category"].values if "category" in df else None)})
    for c in df.columns:
        if c not in out.columns and c not in ("text", "id"):
            out[c] = df[c].values
    return out.reset_index(drop=True)


def main():
    seed_everything()
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    n_tok = lambda texts: [len(tok(t, add_special_tokens=False).input_ids) for t in texts]
    report, sets = {"seed": SEED, "tokenizer": TOKENIZER}, {}

    # Generic harmful (100 AdvBench non-cyber + 100 HarmBench standard non-cyber) and harmless (Alpaca).
    adv, removed = generic.load_advbench()
    report["advbench_keyword_removed"] = removed
    harm = pd.concat([
        frame(adv.sample(100, random_state=SEED), "generic_harmful", "advbench", "advbench-{:03d}", category="advbench"),
        frame(generic.load_harmbench_standard().sample(100, random_state=SEED), "generic_harmful", "harmbench", "harmbench-{}"),
    ], ignore_index=True)
    harm["split"] = splits.half_split(harm, "source", SEED)
    alpaca = generic.load_alpaca_noinput().sample(1200, random_state=SEED)   # 200 generic + 1000 control, disjoint
    harmless = frame(alpaca.iloc[:200], "generic_harmless", "alpaca", "alpaca-{:05d}")
    harmless["split"] = splits.half_split(harmless.assign(g=0), "g", SEED)
    sets["generic"] = pd.concat([harm, harmless], ignore_index=True)
    sets["control_alpaca"] = frame(alpaca.iloc[200:], "control", "alpaca", "alpaca-{:05d}", split="train")

    # MITRE: 4 of 10 ATT&CK categories are 'fit', the rest 'test'.
    mitre = cyber.load_mitre()
    cats = sorted(mitre["category"].unique())
    fit_cats = sorted(np.random.default_rng(SEED).choice(cats, 4, replace=False).tolist())
    report["mitre_fit_categories"] = fit_cats
    report["mitre_fallbacks"] = int(mitre["fallback"].sum())
    mitre = frame(mitre.assign(row=0), "mitre", "cyberseceval2", "", split="test")
    mitre.loc[mitre["category"].isin(fit_cats), "split"] = "fit"
    sets["mitre"] = mitre.drop(columns=["row"])

    frr = cyber.load_frr()
    sets["frr"] = frame(frr, "frr", "cyberseceval2", "", category="frr")

    cys = cyber.load_cysecbench()
    cys = cys.groupby("category", group_keys=False).sample(100, random_state=SEED)
    sets["cysecbench"] = frame(cys, "cysecbench", "cysecbench", "")

    sets["xstest_safe"] = frame(generic.load_xstest_safe(), "xstest_safe", "xstest", "xstest-{:03d}")
    sets["wmdp_cyber"] = frame(cyber.load_wmdp_cyber(), "wmdp_cyber", "wmdp", "", category="cyber")

    # BeaverTails: calibration (Decision 2), held-out calibration, poison pool, reproduction test prompts.
    train = beavertails.load_split("30k_train")
    calib, held, pool, bt = beavertails.calibration(train)
    report["beavertails"] = bt | {"poison_pool": len(pool)}
    bt_rows = lambda df, split: frame(df.rename(columns={"prompt": "text"}), "beavertails", "beavertails_30k_train",
                                      "bt-train-{:05d}", split=split, category=None)[
        ["id", "set", "split", "text", "source", "category", "row"]]
    sets["beavertails_calib"] = pd.concat([bt_rows(calib, "calib"), bt_rows(held, "heldout")], ignore_index=True)
    sets["beavertails_poison_pool"] = bt_rows(pool, "pool")
    rt = beavertails.repro_test(beavertails.load_split("30k_test"), 1000, SEED,
                                 set(calib["prompt"]) | set(held["prompt"]))
    sets["beavertails_repro_test"] = frame(rt.rename(columns={"prompt": "text"})[["row", "text"]], "beavertails_repro_test",
                                           "beavertails_30k_test", "bt-test-{:05d}")

    tc, ts = trendyol.candidates(tok, 1300, MAX_LEN, SEED)
    report["trendyol"] = ts
    sets["trendyol_candidates"] = frame(tc, "trendyol", "trendyol", "trendyol-{:05d}", split="train")
    sets["agnews_repro"] = frame(generic.load_agnews(800), "agnews", "ag_news", "agnews-{:04d}", split="train")

    for df in sets.values():
        df["n_tokens"] = n_tok(df["text"])
    sets["mitre"]["n_tokens_base"] = n_tok(sets["mitre"]["base_text"])

    # Length matching (Decision 9), MITRE test vs FRR, once on the evaluated mutated_prompt and once on
    # the short base_prompt. Matched rows get matched / matched_base = True.
    mt = sets["mitre"][sets["mitre"]["split"] == "test"]
    frr_n = sets["frr"]["n_tokens"]
    report["length_match"] = {}
    for col, flag in (("n_tokens", "matched"), ("n_tokens_base", "matched_base")):
        kh, kb = splits.length_match(mt[col], frr_n, BIN, SEED)
        sets["mitre"][flag] = sets["mitre"].index.isin(kh)
        sets["frr"][flag] = sets["frr"].index.isin(kb)
        report["length_match"][flag] = {"mitre_test_kept": len(kh), "frr_kept": len(kb),
                                        "length_only_auroc": length_auroc(sets["mitre"].loc[kh, col], frr_n.loc[kb]),
                                        "unmatched_auroc": length_auroc(mt[col], frr_n),
                                        "median_tokens_mitre_test": float(mt[col].median())}
    report["length_match"]["median_tokens"] = {k: float(sets[k]["n_tokens"].median()) for k in ("frr", "cysecbench")}
    report["length_match"]["cysecbench_vs_frr_unmatched_auroc"] = length_auroc(sets["cysecbench"]["n_tokens"], frr_n)

    # Checks (guide C1 verification).
    counts = {k: len(v) for k, v in sets.items()}
    report["counts"] = counts
    expect = {"mitre": 1000, "frr": 750, "cysecbench": 1000, "xstest_safe": 250, "wmdp_cyber": 1987, "generic": 400,
              "control_alpaca": 1000, "trendyol_candidates": 1300, "agnews_repro": 800, "beavertails_repro_test": 1000,
              "beavertails_calib": 512}
    bad = {k: (counts[k], v) for k, v in expect.items() if counts[k] != v}
    assert not bad, f"count mismatch (got, expected): {bad}"
    assert (sets["mitre"].groupby("category").size() == 100).all()
    assert (sets["cysecbench"].groupby("category").size() == 100).all()
    all_rows = pd.concat(sets.values(), ignore_index=True)
    dup = all_rows["id"][all_rows["id"].duplicated()]
    assert dup.empty, f"duplicate ids: {dup.tolist()[:5]}"
    norm = all_rows["text"].str.lower().str.strip()
    eval_side = all_rows["split"].isin(["test", "heldout"])
    train_side = set(norm[all_rows["split"].isin(["fit", "train"])])
    calib_side = set(norm[all_rows["split"] == "calib"])
    leaks = all_rows.loc[eval_side & norm.isin(train_side), "id"].tolist()
    # Calibration demos are not trained on, so overlap with an eval set is reported, not fatal,
    # except for the held-out calibration set, which must be new prompts.
    calib_overlap = all_rows.loc[eval_side & norm.isin(calib_side), "id"].tolist()
    leaks += [i for i in calib_overlap if i.startswith("bt-")]
    report["fit_eval_text_overlap"] = leaks
    report["calib_eval_overlap"] = calib_overlap
    assert report["mitre_fallbacks"] <= 50, "MITRE fallbacks above 5%: parsing is wrong"

    OUT.mkdir(exist_ok=True)
    for name, df in sets.items():
        splits.write(df, OUT / f"{name}.jsonl")
    report["manifest"] = splits.manifest(OUT)
    (OUT / "BUILD_REPORT.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "manifest"}, indent=2, default=str))
    assert not leaks, f"{len(leaks)} eval prompts also appear on the training side"
    for flag in ("matched", "matched_base"):
        assert report["length_match"][flag]["length_only_auroc"] <= 0.65, f"length matching failed ({flag})"


if __name__ == "__main__":
    main()
