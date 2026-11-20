"""Generic (non-cyber) prompt sources: AdvBench, HarmBench, Alpaca, XSTest, AG News."""
import pandas as pd
from datasets import load_dataset

ADVBENCH_URL = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
HARMBENCH_URL = ("https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/"
                 "behavior_datasets/harmbench_behaviors_text_all.csv")
# AdvBench has no category field; this keeps cyber harm out of the generic directions (guide C1).
CYBER_KEYWORDS = ["hack", "exploit", "malware", "virus", "password", "network", "ransomware"]

AGNEWS_LABELS = ["World", "Sports", "Business", "Sci/Tech"]
AGNEWS_TEMPLATE = ("Classify the topic of the following news article as World, Sports, Business or Sci/Tech.\n\n"
                   "{text}")


def load_advbench() -> tuple[pd.DataFrame, int]:
    """(non-cyber rows, number removed by the keyword filter)."""
    df = pd.read_csv(ADVBENCH_URL).reset_index(names="row")
    cyber = df["goal"].str.lower().str.contains("|".join(CYBER_KEYWORDS))
    return df[~cyber].rename(columns={"goal": "text"})[["row", "text"]], int(cyber.sum())


def load_harmbench_standard() -> pd.DataFrame:
    df = pd.read_csv(HARMBENCH_URL)
    df = df[(df["FunctionalCategory"] == "standard") & (df["SemanticCategory"] != "cybercrime_intrusion")]
    return df.rename(columns={"Behavior": "text", "SemanticCategory": "category", "BehaviorID": "row"})[
        ["row", "text", "category"]]


def load_alpaca_noinput() -> pd.DataFrame:
    df = load_dataset("tatsu-lab/alpaca", split="train").to_pandas().reset_index(names="row")
    # Alpaca already has a "text" column (the formatted prompt); replace it with the bare instruction.
    return df[df["input"].str.strip() == ""].drop(columns="text").rename(columns={"instruction": "text"})[["row", "text"]]


def load_xstest_safe() -> pd.DataFrame:
    df = load_dataset("Paul/XSTest", split="train").to_pandas()
    df = df[df["label"] == "safe"]
    return df.rename(columns={"id": "row", "prompt": "text", "type": "category"})[["row", "text", "category"]]


def load_agnews(n: int) -> pd.DataFrame:
    df = load_dataset("fancyzhx/ag_news", split=f"train[:{n}]").to_pandas().reset_index(names="row")
    return pd.DataFrame({"row": df["row"], "text": [AGNEWS_TEMPLATE.format(text=t) for t in df["text"]],
                         "answer": [AGNEWS_LABELS[i] for i in df["label"]]})
