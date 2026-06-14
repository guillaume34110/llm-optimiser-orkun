from orkun.curiosity.epistemic import EpistemicCuriosity


class _FixedWM:
    def __init__(self, score_val):
        self._s = score_val

    def score(self, prefix, continuation):
        return self._s

    def encode(self, text):
        return [0.0]


def test_surprise_is_negative_score():
    cur = EpistemicCuriosity()
    wm = _FixedWM(-2.5)
    assert cur.surprise(wm, "ctx", "result") == 2.5


def test_progress_non_negative_and_zero_before_observation():
    cur = EpistemicCuriosity()
    assert cur.progress("fam") == 0.0


def test_falling_surprise_yields_positive_progress():
    cur = EpistemicCuriosity(ema_alpha=0.3)
    for s in [10.0, 8.0, 6.0, 4.0, 2.0]:
        cur.observe("fam", s)
    assert cur.progress("fam") > 0.0


def test_curiosity_is_weighted_combination():
    cur = EpistemicCuriosity(w_surprise=1.0, w_progress=0.0, surprise_scale=10.0)
    # progress weight 0 -> curiosity is pure normalised surprise = 5/10 = 0.5
    assert abs(cur.curiosity("fam", 5.0) - 0.5) < 1e-9
