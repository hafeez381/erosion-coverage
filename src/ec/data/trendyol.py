"""Trendyol cybersecurity instructions: capability training data (Decision 8)."""
import numpy as np
import pandas as pd
from datasets import load_dataset

from ec.judge.stub import is_refusal
from ec.model import chat_text

TRENDYOL = "Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset"


def candidates(tok, n: int, max_len: int, seed: int) -> tuple[pd.DataFrame, dict]:
    """n rows in seeded order that pass: no missing field, no regex refusal, prompt + answer <= max_len.

    C2 later drops rows WildGuard labels as refusals and keeps the first 1000.
    """
    df = load_dataset(TRENDYOL, split="train").to_pandas().reset_index(names="row")
    stats = {"rows": len(df)}
    ok = df[["user", "assistant"]].notna().all(1) & (df["user"].str.strip() != "") & (df["assistant"].str.strip() != "")
    df = df[ok]
    stats["missing"] = int((~ok).sum())
    refused = df["assistant"].map(is_refusal)
    df = df[~refused]
    stats["regex_refusals"] = int(refused.sum())
    order = np.random.default_rng(seed).permutation(len(df))
    keep, too_long = [], 0
    for i in order:  # walk in seeded order, tokenizing only until n rows are accepted
        r = df.iloc[i]
        n_tok = len(tok(chat_text(tok, r["user"]) + r["assistant"], add_special_tokens=False).input_ids)
        if n_tok > max_len:
            too_long += 1
            continue
        keep.append((int(r["row"]), r["user"], n_tok))
        if len(keep) == n:
            break
    stats["too_long_seen"] = too_long
    return pd.DataFrame(keep, columns=["row", "text", "n_tokens_with_answer"]), stats
