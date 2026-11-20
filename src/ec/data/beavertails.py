"""BeaverTails: Dual-Null calibration demos (Decision 2) and the C10 reproduction sets.

Only prompts and row indices are written to prompts/; responses are read back from the Hub by row.
"""
import pandas as pd
from datasets import load_dataset


def load_split(split: str) -> pd.DataFrame:
    df = load_dataset("PKU-Alignment/BeaverTails", split=split).to_pandas().reset_index(names="row")
    return df[["row", "prompt", "response", "is_safe"]]


def calibration(train: pd.DataFrame, n: int = 256) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """First n is_safe rows (calib), the next n (held-out calib), and the unsafe poison pool after both.

    Returns (calib, heldout, poison_pool, stats). The pool starts after the last held-out row, so it can
    never overlap a calibration window.
    """
    safe = train[train["is_safe"]]
    calib = safe.iloc[:n]
    # BeaverTails repeats prompts with different answers; held-out prompts must be new prompts.
    rest = safe.iloc[n:]
    heldout = rest[~rest["prompt"].isin(set(calib["prompt"]))].drop_duplicates("prompt").iloc[:n]
    last = int(heldout["row"].max())
    pool = train[(~train["is_safe"]) & (train["row"] > last)]
    unsafe_prompts = set(train.loc[~train["is_safe"], "prompt"])
    stats = {"calib_last_row": int(calib["row"].max()), "heldout_last_row": last,
             "calib_prompts_also_unsafe": int(calib["prompt"].isin(unsafe_prompts).sum())}
    return calib, heldout, pool, stats


def repro_test(test: pd.DataFrame, n: int, seed: int, exclude: set) -> pd.DataFrame:
    """n unsafe 30k_test prompts, none of which is a calibration prompt (30k_test repeats some)."""
    unsafe = test[~test["is_safe"] & ~test["prompt"].isin(exclude)].drop_duplicates("prompt")
    return unsafe.sample(n, random_state=seed)
