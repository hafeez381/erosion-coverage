"""CyberSecEval 2 MITRE and FRR, and CySecBench, read straight from their GitHub releases."""
import re

import pandas as pd

PURPLELLAMA = "https://raw.githubusercontent.com/meta-llama/PurpleLlama/main/CybersecurityBenchmarks/datasets"
MITRE_URL = f"{PURPLELLAMA}/mitre/mitre_benchmark_100_per_category_with_augmentation.json"
FRR_URL = f"{PURPLELLAMA}/mitre_frr/mitre_frr.json"
MITRE_SUFFIX = " Your response implementing the above in code follows:"
CYSECBENCH_URL = "https://raw.githubusercontent.com/cysecbench/dataset/main/Dataset/Full%20dataset/cysecbench.csv"


def _mitre_text(row: dict) -> tuple[str, bool]:
    """(text, used_fallback). `mutated_prompt` holds '{"prompt": "..."}' plus a suffix. About a fifth
    of the objects are truncated or have unescaped quotes, so read the string after "prompt" with a
    regex instead of a JSON parser."""
    m = re.search(r'"prompt"\s*:\s*"(.*)', row["mutated_prompt"].removesuffix(MITRE_SUFFIX), re.S)
    if not m:
        return row["base_prompt"], True
    body = m.group(1)
    end = re.search(r'"\s*(?:,\s*"\w+"\s*:.*)?}?\s*$', body, re.S)  # closing quote, other keys, brace
    body = body[: end.start()] if end else body
    return body.replace('\\"', '"').replace("\\n", "\n").strip(), False


def load_mitre() -> pd.DataFrame:
    """1000 rows: id, text, category, fallback."""
    rows = pd.read_json(MITRE_URL).to_dict("records")
    texts = [_mitre_text(r) for r in rows]
    return pd.DataFrame({
        "id": [f"mitre-{i:04d}" for i in range(len(rows))],
        "text": [t for t, _ in texts],
        "category": [r["mitre_category"] for r in rows],
        "fallback": [f for _, f in texts],
        "base_text": [r["base_prompt"] for r in rows],  # short original prompt, in FRR's length range
    })


def load_frr() -> pd.DataFrame:
    df = pd.read_json(FRR_URL)
    assert not df["is_malicious"].any()
    return pd.DataFrame({"id": [f"frr-{i:04d}" for i in range(len(df))], "text": df["mutated_prompt"]})


def load_cysecbench() -> pd.DataFrame:
    df = pd.read_csv(CYSECBENCH_URL)
    return pd.DataFrame({"id": [f"cysec-{i:05d}" for i in range(len(df))], "text": df["Prompt"],
                         "category": df["Category"]})


def load_wmdp_cyber() -> pd.DataFrame:
    from datasets import load_dataset
    df = load_dataset("cais/wmdp", "wmdp-cyber", split="test").to_pandas()
    return pd.DataFrame({"id": [f"wmdp-cyber-{i:04d}" for i in range(len(df))], "text": df["question"],
                         "choices": df["choices"].map(list), "answer": df["answer"]})
