import numpy as np
import pytest
import torch

from ec.repr.directions import choose_band


def test_band_longest_contiguous_run_excludes_last_layer():
    a = np.full(36, 0.5)
    a[3:6] = 0.97          # short run
    a[12:20] = 0.98        # longest run
    a[35] = 1.0            # last layer never chosen
    assert choose_band(a) == list(range(12, 20))


def test_band_window_when_most_layers_pass():
    a = np.full(36, 0.99)
    a[20] = 1.0
    assert choose_band(a) == list(range(16, 24))


def test_band_raises_when_nothing_passes():
    with pytest.raises(ValueError):
        choose_band(np.full(36, 0.8))


def test_ablation_removes_direction_from_every_layer():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from ec.repr.ablate import ablated
    m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.float32).eval()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    v = torch.nn.functional.normalize(torch.randn(m.config.hidden_size, generator=torch.Generator().manual_seed(0)), dim=0)
    ids = tok("The quick brown fox", return_tensors="pt")
    with torch.no_grad(), ablated(m, v):
        hs = m(**ids, output_hidden_states=True).hidden_states[1:-1]   # last one is normalised
    for h in hs:
        assert (h @ v).abs().max() < 1e-3
    with torch.no_grad():
        hs = m(**ids, output_hidden_states=True).hidden_states[1:-1]   # hooks removed
    assert max((h @ v).abs().max().item() for h in hs) > 1e-2
