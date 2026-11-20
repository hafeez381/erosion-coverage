import pandas as pd

from ec.data.splits import half_split, length_match


def test_length_match_keeps_ratio_per_bin():
    h = pd.Series([10] * 40 + [50] * 10 + [200] * 30)        # harmful token counts
    b = pd.Series([12] * 20 + [55] * 40 + [300] * 5)          # benign token counts
    kh, kb = length_match(h, b, 32, 0)
    rho = len(h) / len(b)
    for k in set(h // 32) | set(b // 32):
        nh, nb = (h.loc[kh] // 32 == k).sum(), (b.loc[kb] // 32 == k).sum()
        assert nh <= round(rho * nb)                           # never more harmful than the ratio allows
    assert set(h.loc[kh] // 32) == set(b.loc[kb] // 32) - {9}  # bins with only one side are dropped


def test_half_split_is_balanced_per_group():
    df = pd.DataFrame({"src": ["a"] * 10 + ["b"] * 6})
    s = half_split(df, "src", 6304)
    assert (s[df.src == "a"] == "fit").sum() == 5 and (s[df.src == "b"] == "fit").sum() == 3
