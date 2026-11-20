# Compute rules

These rules apply to every GPU job in this project, on Kaggle or Colab. They come from failures that cost GPU hours on earlier projects.

1. **Use a T4 (16 GB, fp16) unless a measured peak says otherwise.** The T4 host has 12 GB of RAM and no swap, so load one model at a time. Never load a judge beside the policy model.
2. **Run every job longer than two minutes detached**, and poll its log with short calls:
   ```bash
   nohup python scripts/<task>.py ... > logs/<task>.log 2>&1 &
   tail -n 5 logs/<task>.log
   ```
3. **Checkpoint every 125 steps.** Write to a temporary path, rename it, then push the checkpoint off the VM with `ec.train.checkpoint.push(path, "<hf-user>/ec-checkpoints")`. That function raises unless the Hub copy has the same sha256. A file whose only copy is on the VM counts as lost.
4. **Never delete an older checkpoint until its replacement is pushed and verified.** Never free storage by deleting a backup. If the Hub quota is full, stop the job and ask the team.
5. **Hugging Face private storage is per account, not per repository.** Check free quota before a run. LoRA adapters are small, so keep them all.
6. **Keep the training log off the VM too**, pushed at every checkpoint.
7. **On out-of-memory errors,** compute the loss only at answer positions or in chunks, then halve the micro-batch.
8. **Pass `do_sample=False` to every `generate` call.** Qwen's generation config samples by default.
9. **Labels only in this repository.** Never commit or upload model responses to harmful prompts. Refusal text for base-refused MITRE prompts goes to a private store.
