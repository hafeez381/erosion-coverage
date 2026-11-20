# Kaggle script kernel, GPU T4, internet on: GPU job 8, base-model evaluation (C2a, C3, C2 sheet).
# WildGuard is gated: the HF token comes from the Kaggle secret HF_TOKEN, or a file hf_token in an
# attached private dataset. Outputs: results/base (labels) and private/ (responses, never committed).
import glob
import os
import subprocess
import sys

BRANCH = "main"


def sh(cmd):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


def hf_token():
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        files = glob.glob("/kaggle/input/*/hf_token")
        return open(files[0]).read().strip() if files else None


os.environ["HF_TOKEN"] = hf_token() or ""
assert os.environ["HF_TOKEN"], "no HF token: add the HF_TOKEN secret or attach the token dataset"
sh(f"git clone --depth 1 -b {BRANCH} https://github.com/hafeez381/erosion-coverage.git /kaggle/working/ec")
sh(f"{sys.executable} -m pip install -q -e '/kaggle/working/ec[gpu]'")
os.chdir("/kaggle/working/ec")
# Smoke run on 4 prompts per set first, so a bug costs minutes, not hours.
sh("python scripts/j8_base_eval.py --limit 4 --out /kaggle/working/results/base_smoke --private /kaggle/working/private_smoke")
sh("python scripts/j8_base_eval.py --out /kaggle/working/results/base --private /kaggle/working/private")
