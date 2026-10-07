"""A 1x1 scorebar-bg is the legacy "hide the scorebar" trick, not a bar.

SkinHealthBar scaled it to 44% of the frame width and took the bar's height
from its aspect ratio (1:1), so a 1920x1080 render composited an 844 px tall
(transparent) "bar" and a colour fill up to 13504 px tall on every frame:
nothing to see, 6 to 12 times slower. Two of the nine most-rendered skins ship
one.

Runnable two ways:  pytest tests/test_scorebar_placeholder.py   OR   python tests/test_scorebar_placeholder.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from osu_taiko_renderer.hud.skin_hud_elements import SkinHealthBar  # noqa: E402


class _Skin:
    def __init__(self, images):
        self.images = images

    def load(self, name):
        return self.images.get(name)


def _img(w, h):
    a = np.zeros((h, w, 4), np.uint8)
    a[..., 3] = 255
    a[..., 0] = 200
    return a


def _draw(bar, w=1920, h=1080):
    calls = []
    frame = np.zeros((h, w, 3), np.uint8)
    hp_h = bar.draw(frame, w, h, 0.75, lambda rgb, src, x, y, anchor: calls.append(src.shape[:2]))
    return hp_h, calls


def test_a_one_pixel_scorebar_is_absent():
    for bg in (_img(1, 1), _img(2, 2), _img(31, 4)):
        bar = SkinHealthBar(_Skin({"scorebar-bg": bg, "scorebar-colour": _img(600, 13)}))
        assert bar.present is False
        hp_h, calls = _draw(bar)
        assert hp_h == 0 and calls == [], (bg.shape, hp_h, calls)   # nothing composited


def test_a_real_scorebar_is_drawn_as_before():
    bar = SkinHealthBar(_Skin({"scorebar-bg": _img(1366, 150), "scorebar-colour-0": _img(474, 30)}))
    assert bar.present is True
    hp_h, calls = _draw(bar)
    bw = int(1920 * 0.44)
    bh = int(bw * 150 / 1366)
    assert hp_h == bh
    assert calls[0] == (bh, bw)                                       # the bar
    assert calls[1] == (int(bh * 30 / 150), int(int(bw * 0.92) * 0.75))  # the fill, at 75% health


def test_no_scorebar_at_all_is_absent():
    bar = SkinHealthBar(_Skin({}))
    assert bar.present is False and _draw(bar) == (0, [])
    assert SkinHealthBar(None).present is False


def test_what_the_placeholder_used_to_cost():
    # documents the defect: the numbers the old rule produced for a 1x1 bg
    w = 1920
    bw = int(w * 0.44)
    bh = max(1, int(bw * 1 / 1))
    fh = max(1, int(bh * (16 / 1)))
    assert (bh, fh) == (844, 13504)


if __name__ == "__main__":
    for _n, _f in sorted(globals().items()):
        if _n.startswith("test_") and callable(_f):
            _f()
            print("ok  ", _n)
    print("all scorebar-placeholder tests PASSED")
