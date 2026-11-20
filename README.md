# Erosion or Coverage? Diagnosing Safety Failures under Capability Fine-Tuning

This repository measures why a safety-aligned language model stops refusing harmful requests after benign LoRA fine-tuning on cybersecurity data. It separates two causes. **Erosion**: the model still represents the request as harmful, but its refusal pathway weakens. **Coverage gap**: the harm lies outside the model's harmfulness representation, or outside the subspace that Dual-Null LoRA protects. The model is Qwen2.5-3B-Instruct, fine-tuned under four conditions (plain LoRA, Dual-Null with generic calibration, Dual-Null with generic and domain calibration, and control LoRA on Alpaca), with one seed (6304) for now; two more seeds come later.

Results will be added here as the runs finish.

## Setup

```bash
git clone https://github.com/hafeez381/erosion-coverage.git
cd erosion-coverage
```

Install the package in a Python 3.10–3.12 environment, then run the fast tests:

```bash
uv venv -p 3.12 && uv pip install -e ".[dev]"
.venv/bin/pytest tests
```

Run the C0 walking skeleton on a Mac with Qwen2.5-0.5B-Instruct. It caches activations, scores $H$ and $R$, generates, labels refusals with a regex stub, takes one projected LoRA step, and writes `results/skeleton/`:

```bash
.venv/bin/python scripts/c0_skeleton.py
```

On a Kaggle T4, `launchers/kaggle_c0/` runs the same skeleton on Qwen2.5-3B-Instruct, times 20 LoRA steps at batch 4 and length 1024, and runs the sniff test (`scripts/c0_sniff.py`). Set your username in `kernel-metadata.json`, then run `kaggle kernels push -p launchers/kaggle_c0`. The compute rules are in [docs/compute.md](docs/compute.md).

## Compute

Training, generation and evaluation run on Kaggle and Colab T4 GPUs (16 GB, fp16). Unit tests run on Qwen2.5-0.5B-Instruct on CPU or Apple MPS. Checkpoints and activation caches live in a private Hugging Face repository, not in this repository.

## Reproducing the results

The pipeline has three stages, added in this order.

1. **Diagnose the base model.** Harmfulness and refusal directions and scores, within-domain AUROC, Dual-Null calibration bases, and per-prompt coverage.
2. **Fine-tune.** Four conditions times three seeds on cybersecurity instruction data, plus two runs that reproduce one Dual-Null result.
3. **Evaluate.** Judged refusal and actionable-harm rates, the erosion–coverage decomposition, the coverage–failure correlation, and causal tests by ablation and steering.

## Attribution

- **Model.** [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct).
- **Judges.** [WildGuard](https://huggingface.co/allenai/wildguard) and [HarmBench-Mistral-7b-val-cls](https://huggingface.co/cais/HarmBench-Mistral-7b-val-cls).
- **Data.** [CyberSecEval 2](https://github.com/meta-llama/PurpleLlama/tree/main/CybersecurityBenchmarks), [CySecBench](https://github.com/cysecbench/dataset), [Trendyol Cybersecurity Instruction Tuning](https://huggingface.co/datasets/Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset), [WMDP](https://huggingface.co/datasets/cais/wmdp), [XSTest](https://huggingface.co/datasets/Paul/XSTest), [AdvBench](https://github.com/llm-attacks/llm-attacks), [HarmBench](https://github.com/centerforaisafety/HarmBench), [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca), [BeaverTails](https://huggingface.co/datasets/PKU-Alignment/BeaverTails).
- **Libraries.** PyTorch, Hugging Face Transformers, PEFT and Datasets, scikit-learn, SciPy.

No code was copied from public implementations. Code was written with LLM assistance.

## References

- Arditi et al. 2024. *Refusal in Language Models Is Mediated by a Single Direction.* [arXiv:2406.11717](https://arxiv.org/abs/2406.11717)
- Zhao et al. 2025. *LLMs Encode Harmfulness and Refusal Separately.* [arXiv:2507.11878](https://arxiv.org/abs/2507.11878)
- Ren and Sutherland 2025. *Learning Dynamics of LLM Finetuning.* [arXiv:2407.10490](https://arxiv.org/abs/2407.10490)
