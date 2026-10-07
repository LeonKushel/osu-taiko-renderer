"""Torus bitmap-font renderer (osu!lazer's UI font).

osu-resources ships Torus as an osu!framework BMFont-binary (v3) glyph store:
`Torus-<weight>.bin` (magic 'BMF\\x03') + `Torus-<weight>_0.png` LA atlas. We
parse the binary directly and render tinted, scaled text into RGBA — giving the
exact lazer "GREAT" / HUD label glyphs rather than a substitute typeface.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
from PIL import Image

_FONT_DIR = Path(__file__).resolve().parent.parent / "fonts_data"
_cache: dict[str, "TorusFont"] = {}


class TorusFont:
    def __init__(self, weight: str = "Regular"):
        bin_path = _FONT_DIR / f"Torus-{weight}.bin"
        png_path = _FONT_DIR / f"Torus-{weight}_0.png"
        data = bin_path.read_bytes()
        if data[:3] != b"BMF" or data[3] != 3:
            raise ValueError(f"not BMFont v3: {bin_path}")
        self.size = 0
        self.line_height = 0
        self.base = 0
        self.chars: dict[int, dict] = {}
        self._cache: dict = {}
        # (char, tw, th) -> LANCZOS-resized coverage mask, float32 0..1.
        # PER INSTANCE: each weight has its own glyph atlas, so a
        # module-global cache served Bold masks for Regular. The tint is
        # applied later in _blit_mask, so no colour in the key.
        self._gmask: dict = {}
        off = 4
        while off < len(data):
            btype = data[off]
            bsize = struct.unpack_from("<I", data, off + 1)[0]
            off += 5
            block = data[off:off + bsize]
            off += bsize
            if btype == 1:  # info
                self.size = abs(struct.unpack_from("<h", block, 0)[0])
            elif btype == 2:  # common
                self.line_height, self.base = struct.unpack_from("<HH", block, 0)
            elif btype == 4:  # chars (20 bytes each)
                for i in range(len(block) // 20):
                    (cid, x, y, w, h, xo, yo, xadv, page, chnl) = \
                        struct.unpack_from("<IHHHHhhhBB", block, i * 20)
                    self.chars[cid] = dict(x=x, y=y, w=w, h=h, xo=xo, yo=yo,
                                           xadv=xadv)
        # Atlas: LA -> use alpha as coverage mask (glyph is white).
        atlas = Image.open(png_path)
        arr = np.array(atlas)
        if arr.ndim == 3 and arr.shape[2] >= 2:
            self._mask = arr[..., -1].astype(np.float32) / 255.0   # alpha chan
        else:
            self._mask = arr.astype(np.float32) / 255.0
        self._aw, self._ah = atlas.size

    def measure(self, text: str, px: float, spacing: float = 0.0) -> tuple[int, int]:
        s = px / self.size
        w = 0.0
        for ch in text:
            g = self.chars.get(ord(ch))
            if g is None:
                w += px * 0.4 + spacing * s
                continue
            w += g["xadv"] * s + spacing * s
        h = self.line_height * s
        return max(1, int(round(w))), max(1, int(round(h)))

    def render(self, text: str, px: float, color=(255, 255, 255, 255),
               spacing: float = 0.0, pad: int = 8) -> np.ndarray:
        """Render `text` to an RGBA numpy array, tinted `color`. Cached by
        (text, px, color, spacing) — labels/numbers repeat across frames, so
        re-rasterising every frame is the main render-time cost without this."""
        key = (text, round(px, 1), tuple(color), round(spacing, 2), pad)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        out = self._render(text, px, color, spacing, pad)
        if len(self._cache) > 1024:
            self._cache.clear()
        self._cache[key] = out
        return out

    def _render(self, text: str, px: float, color=(255, 255, 255, 255),
                spacing: float = 0.0, pad: int = 8) -> np.ndarray:
        s = px / self.size
        w, h = self.measure(text, px, spacing)
        W, H = w + pad * 2, int(round(self.line_height * s)) + pad * 2
        out = np.zeros((H, W, 4), dtype=np.float32)
        if len(color) == 3:
            color = (*color, 255)
        cr, cg, cb, ca = [c / 255.0 for c in color]
        penx = float(pad)
        for ch in text:
            g = self.chars.get(ord(ch))
            if g is None:
                penx += px * 0.4 + spacing * s
                continue
            gw, gh = g["w"], g["h"]
            if gw > 0 and gh > 0:
                glyph = self._mask[g["y"]:g["y"] + gh, g["x"]:g["x"] + gw]
                # scale glyph mask to target size
                tw, th = max(1, int(round(gw * s))), max(1, int(round(gh * s)))
                # MEMOISED per (glyph, target size). The LANCZOS resize plus its
                # PIL round-trip was the dominant per-call cost, and it ran for
                # EVERY character on EVERY cache-missing string — i.e. every
                # frame for the score, combo, accuracy, hit counts and clock,
                # because their text changes. `_render`'s own (text, px, colour)
                # cache cannot help a string that changes.
                # The mask is COLOUR-INDEPENDENT (the tint is applied later in
                # _blit_mask), so the key needs no colour and one entry serves
                # every tint. Byte-identical: same resize, same filter, same
                # values — only recomputation is skipped.
                # Measured: full render of a changing 7-digit string was
                # 0.409 ms/frame; dynamic-text composition was 62% of
                # `sprite_build` and freezing it was worth +31% fps.
                _gk = (ord(ch), tw, th)
                gm = self._gmask.get(_gk)
                if gm is None:
                    gm = np.array(Image.fromarray((glyph * 255).astype(np.uint8))
                                  .resize((tw, th), Image.LANCZOS)  # type: ignore[attr-defined]
                                  ).astype(np.float32) / 255.0
                    if len(self._gmask) > 4096:
                        self._gmask.clear()
                    self._gmask[_gk] = gm
                # BMFont offsets are measured from the cell top-left.
                dx = int(round(penx + g["xo"] * s))
                dy = int(round(pad + g["yo"] * s))
                _blit_mask(out, gm, dx, dy, (cr, cg, cb, ca))
            penx += g["xadv"] * s + spacing * s
        out8 = (np.clip(out, 0, 1) * 255).astype(np.uint8)
        return out8


def _blit_mask(dst, mask, x, y, color):
    """Alpha-blit a coverage mask tinted `color` (r,g,b,a 0..1) into dst RGBA."""
    H, W = dst.shape[:2]
    mh, mw = mask.shape
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + mw), min(H, y + mh)
    if x1 <= x0 or y1 <= y0:
        return
    sub = mask[y0 - y:y1 - y, x0 - x:x1 - x]
    a = sub * color[3]
    region = dst[y0:y1, x0:x1]
    for c in range(3):
        region[..., c] = region[..., c] * (1 - a) + color[c] * a
    region[..., 3] = region[..., 3] + a * (1 - region[..., 3])


def get_font(weight: str = "Regular") -> TorusFont:
    if weight not in _cache:
        _cache[weight] = TorusFont(weight)
    return _cache[weight]
