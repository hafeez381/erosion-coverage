# Kaggle script kernel: GPU T4, internet on. Installs the repo and runs the C0 skeleton and sniff test.
# Push with:  kaggle kernels push -p launchers/kaggle_c0   (set your username in kernel-metadata.json)
import subprocess
import sys

BRANCH = "main"


def sh(cmd):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


sh(f"git clone --depth 1 -b {BRANCH} https://github.com/hafeez381/erosion-coverage.git /kaggle/working/ec")
sh(f"{sys.executable} -m pip install -q -e /kaggle/working/ec")
sh(f"{sys.executable} -m pip uninstall -y -q torchao")  # Kaggle ships torchao 0.10; new PEFT rejects it
sh("cd /kaggle/working/ec && python scripts/c0_skeleton.py --model Qwen/Qwen2.5-3B-Instruct --time-steps 20 "
   "--out /kaggle/working/results/skeleton")
sh("cd /kaggle/working/ec && python scripts/c0_sniff.py --model Qwen/Qwen2.5-3B-Instruct "
   "--out /kaggle/working/results/sniff")
