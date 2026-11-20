"""Splits, length matching (Decision 9), and writing prompts/*.jsonl with a sha256 manifest."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ec.train.checkpoint import sha256

FIELDS = ["id", "set", "split", "text", "source", "category", "n_tokens"]


def half_split(df: pd.DataFrame, by: str, seed: int) -> pd.Series:
    """'fit' for the first half of each group in seeded order, 'heldout' for the rest."""
    split = pd.Series("heldout", index=df.index)
    for _, g in df.groupby(by):
        idx = g.sample(frac=1, random_state=seed).index
        split[idx[: len(idx) // 2]] = "fit"
    return split


def length_match(h: pd.Series, b: pd.Series, bin_tokens: int, seed: int) -> tuple[pd.Index, pd.Index]:
    """Subsample two token-count series so every length bin keeps the overall harmful:benign ratio.

    Per bin, keep as many benign rows as the bin's harmful rows can pair with at ratio rho, and vice
    versa. Returns the kept indices of h and b.
    """
    rng = np.random.default_rng(seed)
    rho = len(h) / len(b)
    bh, bb = h // bin_tokens, b // bin_tokens
    keep_h, keep_b = [], []
    for k in sorted(set(bh) | set(bb)):
        H, B = h.index[bh == k], b.index[bb == k]
        nb = min(len(B), int(len(H) / rho))
        nh = min(len(H), round(rho * nb))
        keep_h += list(rng.choice(H, nh, replace=False))
        keep_b += list(rng.choice(B, nb, replace=False))
    return pd.Index(keep_h), pd.Index(keep_b)


def write(rows: pd.DataFrame, path: Path) -> None:
    for f in FIELDS:
        assert f in rows.columns, f"{path.name}: missing field {f}"
    extra = [c for c in rows.columns if c not in FIELDS]
    with open(path, "w") as fh:
        for r in rows[FIELDS + extra].to_dict("records"):
            fh.write(json.dumps({k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in r.items()},
                                ensure_ascii=False, default=int) + "\n")


def manifest(folder: Path) -> dict:
    m = {p.name: sha256(str(p)) for p in sorted(folder.glob("*.jsonl"))}
    (folder / "MANIFEST.json").write_text(json.dumps(m, indent=2) + "\n")
    return m
