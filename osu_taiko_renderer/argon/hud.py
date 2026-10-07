"""Argon taiko HUD: score (top-left), accuracy + PP (top-right), combo
(bottom-left), key counter B1–B4 (bottom-right), song progress (bottom).
Numbers use the Argon counter font; labels use Torus — matching lazer's
ArgonScoreCounter / ArgonAccuracyCounter / ArgonComboCounter / etc.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from .counter import ArgonCounter
from .font import get_font
from osu_taiko_renderer.render.envflag import envflag

_LABEL = (188, 200, 214)          # muted Torus label colour
_WHITE = (255, 255, 255)
_ACCENT = (0x66, 0xcc, 0xff)      # Argon blue

# osu! mod bitmask → acronym (display order matters: difficulty then time then fl)
_MODS = [(1 << 1, "EZ"), (1 << 4, "HR"), (1 << 0, "NF"), (1 << 3, "HD"),
         (1 << 8, "HT"), (1 << 9, "NC"), (1 << 6, "DT"), (1 << 10, "FL"),
         (1 << 5, "SD"), (1 << 14, "PF"), (1 << 12, "SO"), (1 << 7, "RX")]


def mod_acronyms(mods: int) -> list[str]:
    out = [a for bit, a in _MODS if mods & bit]
    if (mods & (1 << 9)) and "DT" in out:    # NC implies DT bit; show NC only
        out.remove("DT")
    return out


# Argon grade colours (OsuColour.ForRank).
_GRADE_COL = {"SS": (255, 221, 85), "X": (255, 221, 85), "SSH": (200, 220, 255),
              "S": (255, 204, 34), "SH": (200, 220, 255),
              "A": (0xb3, 0xd9, 0x44), "B": (0x66, 0xcc, 0xff),
              "C": (0xcb, 0x3c, 0xec), "D": (0xed, 0x11, 0x21)}


# R3D_TAIKO_ROUND=1: round-to-nearest on the 8-bit store instead of truncating.
# taiko is the ONLY engine that truncates here — catch (render/flashlight.py:162) and
# std (render/hud.py) both use np.rint. Truncation is a BIASED estimator: it loses on
# average half a level on every blended pixel, which is why the GPU-composited HUD
# measured uniformly +1 brighter (99.4% of differing pixels at +1, ZERO negative).
# With this on, the CPU path matches the GPU's RGBA8 round-to-nearest exactly, so the
# GPU HUD becomes byte-identical rather than "1 LSB off".
from osu_taiko_renderer.render.envflag import ROUND as _ROUND  # noqa: E402


def _to8(x):
    """Clip to 0..255 and store as uint8, rounding per _ROUND."""
    import numpy as _np
    c = _np.clip(x, 0, 255)
    return _np.rint(c).astype(_np.uint8) if _ROUND else c.astype(_np.uint8)


def _fillband(rgb, x0, x1, y, h, color, alpha):
    """Alpha-fill a horizontal band [x0,x1) × [y,y+h) of rgb with color."""
    W = rgb.shape[1]
    x0, x1 = max(0, int(x0)), min(W, int(x1))
    if x1 <= x0:
        return
    if _SINK is not None:
        # GL straight-alpha blend is dst*(1-a) + src*a — the same expression as
        # below, so an untextured translucent sprite is exact bar 8-bit rounding.
        _sink_solid(x0, x1, int(y), int(y) + int(h), color, float(alpha))
        return
    seg = rgb[y:y + h, x0:x1, :3].astype(np.float32)
    seg = seg * (1 - alpha) + np.array(color, np.float32) * alpha
    rgb[y:y + h, x0:x1, :3] = _to8(seg)


# perf: per-texture cache of the float32 conversion + alpha terms used by
# _blit. The HUD composites the same (font/counter-cached) RGBA arrays every
# frame; precomputing `1-a` and `rgb*a` once per texture halves the per-blit
# math while producing bit-identical results (same float32 ops, elementwise —
# cropping commutes with them). id() keys are safe because the cache keeps a
# strong ref to the source (id can't be reused while cached); hard-clear bound.
_PM_CACHE: dict = {}
_PM_CACHE_MAX = 1024


def _pm_terms(src):
    """(1-a, rgb*a, oy, ox) of `src`, cropped to the tight alpha>0 bounding
    box (blending where a==0 is an exact float no-op: x*1.0 + s*0.0 == x, so
    skipping those pixels is bit-identical — font/wedge textures carry large
    transparent pads). None = fully transparent (nothing to composite)."""
    key = id(src)
    hit = _PM_CACHE.get(key)
    if hit is not None and hit[0] is src:
        return hit[1]
    mask = src[..., 3] != 0
    rows = mask.any(axis=1)
    if not rows.any():
        terms = None
    else:
        cols = mask.any(axis=0)
        y0 = int(np.argmax(rows))
        y1 = len(rows) - int(np.argmax(rows[::-1]))
        x0 = int(np.argmax(cols))
        x1 = len(cols) - int(np.argmax(cols[::-1]))
        s = src[y0:y1, x0:x1].astype(np.float32)
        a = s[..., 3:4] / 255.0
        terms = (1 - a, s[..., :3] * a, y0, x0)
    if len(_PM_CACHE) > _PM_CACHE_MAX:
        _PM_CACHE.clear()
    _PM_CACHE[key] = (src, terms)
    return terms


# R3D_TAIKO_HUD_CPROF=1: cProfile ONLY the HUD's overlay() call, to get per-function
# CALL COUNTS (cProfile counts C builtins too, including ndarray methods, which is the
# whole point -- we want to know how many numpy dispatches an element costs). Times
# reported here are inflated by the per-frame enable/disable, so read counts from this
# and timings from R3D_TAIKO_STAGE.
_HUD_CPROF = envflag("R3D_TAIKO_HUD_CPROF")
if _HUD_CPROF:
    import atexit as _cp_atx
    import cProfile as _cp_mod
    import io as _cp_io
    import pstats as _cp_pstats
    import sys as _cp_sys

    _CP = _cp_mod.Profile()

    @_cp_atx.register
    def _cp_dump():
        buf = _cp_io.StringIO()
        st = _cp_pstats.Stats(_CP, stream=buf)
        st.sort_stats("tottime").print_stats(40)
        print("[hud-cprof] HUD overlay() only, all frames:", file=_cp_sys.stderr)
        print(buf.getvalue(), file=_cp_sys.stderr, flush=True)


_BLIT_PROF = __import__("os").environ.get("R3D_TAIKO_HUDPROF")
_BP: dict = {}
_BSTAT: dict = {}
if _BLIT_PROF:
    import atexit as _bp_atx
    import sys as _bp_sys
    import time as _bp_time

    @_bp_atx.register
    def _bp_dump():
        if not _BP:
            return
        n = max(1, _BP.pop(("frames",), (1, 0))[0])
        rows = sorted(_BP.items(), key=lambda kv: -kv[1][1])
        print(f"[hud-prof] per frame, by CALL SITE (hud.py:line), {int(n)} frames:",
              file=_bp_sys.stderr)
        tot = 0.0
        for (site,), (cnt, dt) in rows:
            tot += dt
            print(f"[hud-prof]   line {site:<5} {dt / n * 1e3:7.3f} ms"
                  f"  {cnt / n:6.2f} calls", file=_bp_sys.stderr)
        print(f"[hud-prof]   {'TOTAL':<10} {tot / n * 1e3:7.3f} ms",
              file=_bp_sys.stderr)
        f2 = max(1, _SINK_STATS["frames"])
        print(f"[hud-prof] GL SINK: {_SINK_STATS['uploads']/f2:.2f} uploads/frame, "
              f"{_SINK_STATS['upload_s']/f2*1e3:.3f} ms/frame in write_dyn, "
              f"{_SINK_STATS['sprites']/f2:.2f} sprites/frame", file=_bp_sys.stderr)
        stat = sorted(k for k, v in _BSTAT.items() if len(v) == 1)
        dyn = sorted((len(_BSTAT[k]), k) for k in _BSTAT if len(_BSTAT[k]) > 1)
        sms = sum(_BP.get((k,), (0, 0.0))[1] for k in stat) / n * 1e3
        print(f"[hud-prof] STATIC sites (1 distinct blit all render): "
              f"{stat}  = {sms:.3f} ms/frame", file=_bp_sys.stderr)
        print(f"[hud-prof] DYNAMIC (variants, line): {dyn}", file=_bp_sys.stderr)


# --- GL sink -----------------------------------------------------------------
# When a sink is installed, _blit and _rect EMIT GL SPRITES instead of compositing
# on the CPU, and `overlay` runs unmodified. Each element is still composed on the
# CPU exactly as before (so glyph rendering and every alpha-domain overlap inside
# ArgonCounter are untouched); only the final blend onto the frame moves to the GPU.
# That keeps the delta at the ~1 LSB rounding floor and — unlike a reserved-grid
# layout — moves NOTHING on screen.
#
# Textures are keyed on an EXPLICIT PER-CALL-SITE NAME passed as `_blit(..., key=)`,
# unique per element, and re-uploaded only when the element's bytes actually change,
# so the frame-invariant elements upload once and the dynamic ones pay a
# glTexSubImage2D.
import time as _hud_time
_bp_now = _hud_time.perf_counter
_SINK_STATS = {"upload_s": 0.0, "uploads": 0, "frames": 0, "sprites": 0}
# Typed `Any`, not left bare, and NOT `dict | None`. Three reasons, in order of
# how much time each cost to learn:
#   1. Bare `_SINK = None` makes mypy infer the static type as exactly `None`.
#      Under mypyc that becomes a None-typed slot, so `global _SINK; _SINK = {...}`
#      stores a dict into it and the next READ raises
#      "TypeError: None object expected; got dict" -- from `if _SINK is not None`,
#      a line that cannot fail in interpreted Python.
#   2. `dict | None` type-checks but is not indexable without narrowing, and this
#      module indexes `_SINK[...]` in 19 places across several functions. Narrowing
#      a module global does not survive a call, so each site would need its own
#      local alias or assert.
#   3. `Any` keeps the object boxed, which costs nothing here: every access is a
#      dict lookup either way.
# The sentinel stays None (not {}) on purpose -- the guard is `is not None`, and an
# empty dict would pass it while being falsy everywhere else.
_SINK: Any = None


class _ShimFrame:
    """Stands in for the frame while the sink is active. `overlay` only ever reads
    `.shape` off it — every write goes through _blit or _rect, both of which emit
    sprites instead. Avoids allocating a throwaway 6 MB array per frame."""

    __slots__ = ("shape",)

    def __init__(self, h, w):
        self.shape = (h, w, 3)


def _sink_solid(x0, x1, y0, y1, col, alpha):
    if _SKPROF:
        _t0 = _sk_time.perf_counter()
        try:
            return _sink_solid_inner(x0, x1, y0, y1, col, alpha)
        finally:
            _SK["solid"] += _sk_time.perf_counter() - _t0
            _SK["n_solid"] += 1
    return _sink_solid_inner(x0, x1, y0, y1, col, alpha)


def _sink_solid_inner(x0, x1, y0, y1, col, alpha):
    """Emit an untextured quad (texture_key=None -> the renderer's 1x1 white), with
    colour and alpha straight from u_color. Used for the solid rects, the hit-error
    window bands and the fading tick marks."""
    from osu_taiko_renderer.beatmap.models import Sprite
    x0, x1, y0, y1 = int(x0), int(x1), int(y0), int(y1)
    if x1 <= x0 or y1 <= y0 or alpha <= 0.0:
        return
    _SINK["sprites"].append(Sprite(
        (x0 + x1) / 2.0, (y0 + y1) / 2.0, x1 - x0, y1 - y0,
        texture_key=None,
        color=(col[0] / 255.0, col[1] / 255.0, col[2] / 255.0, float(alpha))))


def _rect(rgb, y0, y1, x0, x1, col):
    """Opaque solid rectangle. Was `rgb[y0:y1, x0:x1] = col` inline; routed through
    here so the sink can emit it as a single untextured GL sprite (texture_key=None
    resolves to the renderer's 1x1 white texture, so colour comes straight from
    u_color). No blending is involved either way, so this is exact."""
    y0, y1, x0, x1 = int(y0), int(y1), int(x0), int(x1)
    if y1 <= y0 or x1 <= x0:
        return
    if _SINK is not None:
        _sink_solid(x0, x1, y0, y1, col, 1.0)
        return
    rgb[y0:y1, x0:x1, :3] = col


def _blit(rgb, src, x, y, anchor="tl", key=None):
    """Alpha-composite RGBA uint8 `src` onto RGB uint8 `rgb`. anchor picks the
    reference corner: tl, tr, bl, br, tc, bc, cc.

    `key` names this call site so the GL sink can give the element its own
    persistent texture. Pass a short stable string, unique per call site. See
    _sink_blit_inner for why a name and not a line number."""
    if _SINK is not None:
        _sink_blit(src, x, y, anchor, key)
        return
    if _BLIT_PROF:
        _t0 = _bp_time.perf_counter()
        try:
            _r = _blit_inner(rgb, src, x, y, anchor)
        finally:
            _k = (_bp_sys._getframe(1).f_lineno,)
            _c, _d = _BP.get(_k, (0, 0.0))
            _BP[_k] = (_c + 1, _d + _bp_time.perf_counter() - _t0)
            # STATIC DETECTION: track the set of distinct (source bytes,
            # placement) seen per call site. A site with exactly one entry is
            # provably invariant across the whole render and can be uploaded
            # once and drawn as a GL sprite with no fidelity change beyond the
            # 1-LSB blend rounding.
            _h = (hash(src.tobytes()), src.shape, int(round(x)), int(round(y)),
                  anchor)
            _s = _BSTAT.setdefault(_k[0], set())
            if len(_s) < 8:
                _s.add(_h)
        return _r
    return _blit_inner(rgb, src, x, y, anchor)


from osu_taiko_renderer.render.envflag import GPU_NUM as _GPU_NUM  # noqa: E402


def _anchor_topleft(src, x, y, anchor):
    """The integer top-left _blit_inner resolves for (src, x, y, anchor), before
    it shifts by the alpha-bbox crop offset. Reproduced exactly so a GL quad
    drawn over the FULL (uncropped) image lands on the same pixels — the
    transparent margin contributes nothing under straight alpha."""
    h, w = src.shape[:2]
    if "r" in anchor:
        x -= w
    elif "c" in anchor[1:]:
        x -= w // 2
    if "b" in anchor:
        y -= h
    return int(round(x)), int(round(y)), w, h


def _num_sprite(gl, key, src, x, y, anchor):
    """Draw a per-frame HUD number image as one GL sprite instead of a CPU blit.

    The image is composed on the CPU exactly as before (so glyph rendering and
    the alpha-domain overlap inside ArgonCounter are untouched) and written into
    a PERSISTENT texture; only the final composite onto the frame moves to GL.
    The residual delta is therefore the same ~1 LSB as the effects work
    (numpy float32->clip->truncate vs GL round-to-nearest into RGBA8), NOT a
    change in glyph shape — which is what a per-digit atlas would have cost.

    Returns a Sprite, or None to fall back to the CPU blit."""
    from osu_taiko_renderer.beatmap.models import Sprite
    if gl is None:
        return None
    r = gl.write_dyn(key, src)
    if r is None:
        return None
    w, h, us, vs = r
    x0, y0, fw, fh = _anchor_topleft(src, x, y, anchor)
    return Sprite(x0 + fw / 2.0, y0 + fh / 2.0, fw, fh, texture_key=key,
                  color=(1.0, 1.0, 1.0, 1.0), uv_scale=(us, vs))


_SINK_OVF_DIAG = envflag("R3D_TAIKO_SINK_OVF")
# Per-texture reserve ceiling. 64 MB holds a 4096x4096 RGBA, and comfortably
# holds the pathological 776x13504 scorebar (41.9 MB) that motivated row 15.
_SINK_MAX_BYTES = int(__import__("os").environ.get(
    "R3D_TAIKO_SINK_MAX_MB", "64")) * 1024 * 1024
_OVF_SEEN: set = set()


_SKPROF = envflag("R3D_TAIKO_SINKPROF")
_SK = {"blit": 0.0, "solid": 0.0, "n_blit": 0, "n_solid": 0, "frames": 0,
       "same_obj": 0, "new_obj": 0}
if _SKPROF:
    import atexit as _sk_atx
    import time as _sk_time

    @_sk_atx.register
    def _sk_dump():
        import sys as _sk
        n = max(1, _SK["frames"])
        print(f"[sink-prof] over {n} frames: _sink_blit={_SK['blit']/n*1e3:.4f} ms "
              f"({_SK['n_blit']/n:.1f} calls)  _sink_solid={_SK['solid']/n*1e3:.4f} ms "
              f"({_SK['n_solid']/n:.1f} calls)", file=_sk.stderr, flush=True)
        print(f"[sink-prof] element content: cached_array={_SK['same_obj']/n:.1f}/frame  "
              f"fresh_array={_SK['new_obj']/n:.1f}/frame", file=_sk.stderr, flush=True)


def _sink_blit(src, x, y, anchor, key=None):
    if _SKPROF:
        _t0 = _sk_time.perf_counter()
        try:
            return _sink_blit_inner(src, x, y, anchor, key)
        finally:
            _SK["blit"] += _sk_time.perf_counter() - _t0
            _SK["n_blit"] += 1
    return _sink_blit_inner(src, x, y, anchor, key)


def _sink_blit_inner(src, x, y, anchor, site_id=None):
    """Emit one HUD element as a GL sprite. Keyed per call site so each element gets
    its own persistent texture; the bytes are re-uploaded only when they change,
    which makes the frame-invariant elements a one-time cost."""
    from osu_taiko_renderer.beatmap.models import Sprite
    gl = _SINK["gl"]
    # Key on the call site PLUS how many times that site has fired this frame.
    # The site name alone is NOT unique: several call sites sit inside loops and
    # blit a different element per iteration (the GREAT/OK/MISS hit counter fires one
    # site 3x, the B1-B4 key counter fires one site 4x, the mod pills likewise). With
    # a site-only key those elements shared a texture, last write won, and the frame
    # came back with one element's art under another's name — measured as 3595 pixels
    # at max|d| up to 246, e.g. a red glyph where a blue one belonged. Hence `idx`.
    #
    # WHY A NAME AND NOT A LINE NUMBER: this used to read the caller's line via
    # sys._getframe(2).f_lineno, which made the stack depth load-bearing -- adding
    # any wrapper silently re-keyed every element. Worse, it is incompatible with
    # mypyc: compiled functions push no Python frame, so _getframe either raises
    # "call stack is not deep enough" (fully compiled) or lands on the nearest
    # interpreted frame and mis-keys silently (partially compiled). Callers now
    # pass an explicit `key=`; see _blit. The fallback below keeps any caller that
    # has not been migrated working, and spells its key exactly as before.
    if site_id is None:
        import sys as _sk_sys
        site_id = f"s{_sk_sys._getframe(3 if _SKPROF else 2).f_lineno}"
    site = site_id
    seen = _SINK["seen"]
    idx = seen.get(site, 0)
    seen[site] = idx + 1
    h, w = src.shape[:2]
    # NB: the fallback above yields site="s688", so an unmigrated caller still
    # spells its texture key exactly as it did before ("hud_s688_0").
    key = f"hud_{site}_{idx}"
    cap = _SINK["caps"].get(key)
    if cap is None or w > cap[0] or h > cap[1]:
        # Reserve generously so a later, larger variant of the same element does not
        # force a reallocation mid-render (write_dyn refuses an oversized write).
        cw, ch = max(w, (cap[0] if cap else 0)) * 2, max(h, (cap[1] if cap else 0)) * 2
        # Clamp to the GL TEXTURE LIMIT, not the frame size (ledger row 15).
        #
        # The old clamp was `min(cw, frame_w), min(ch, frame_h)`, which made any element
        # LARGER THAN THE FRAME impossible to reserve -- it overflowed, the sink returned
        # None, and the whole HUD fell back to CPU compositing for the rest of the render.
        # That is not hypothetical: a 1x1 `scorebar-bg` scaled to 776x13504 is exactly the
        # shape that triggers it, and a frame is only 1080 tall. An element bigger than the
        # frame is legitimate (it is simply clipped when drawn); the texture just has to
        # hold it.
        #
        # It matters more now: under R3D_TAIKO_GPU_YUV the CPU fallback CANNOT run (the
        # readback is planar YUV, not RGB), so this overflow went from "slow frame" to
        # "failed render".
        # Clamp to max(frame, element) — NOT to the element alone and not to the frame
        # alone. Elements that fit the frame keep the old behaviour exactly (2x headroom
        # capped at frame size, so VRAM is unchanged for every normal skin); elements
        # LARGER than the frame get exactly what they need instead of being refused.
        # Doubling past the frame would reserve 33 MB for a full-frame element where
        # 8.3 MB used to do, across ~22 textures.
        _mt = _SINK["max_tex"]
        cw = min(cw, max(_SINK["w"], w), _mt)
        ch = min(ch, max(_SINK["h"], h), _mt)
        # Guard the reserve, not the element: the 2x headroom exists to avoid
        # reallocation, so drop it before refusing a texture that would otherwise fit.
        if cw * ch * 4 > _SINK_MAX_BYTES:
            cw, ch = min(max(w, 1), _mt), min(max(h, 1), _mt)
        if w > cw or h > ch:
            _SINK["overflow"] += 1
            if _SINK_OVF_DIAG:
                import sys as _ov
                _k = (site, w, h)
                if _k not in _OVF_SEEN:
                    _OVF_SEEN.add(_k)
                    import traceback as _tb
                    st = _tb.format_stack(limit=4)[0].strip().replace("\n", " | ")
                    print(f"[sink-overflow] element {w}x{h} > reserve {cw}x{ch} "
                          f"(frame {_SINK['w']}x{_SINK['h']}, GL max {_SINK['max_tex']})"
                          f"  {st}",
                          file=_ov.stderr, flush=True)
            return
        gl.dyn_texture(key, cw, ch)
        _SINK["caps"][key] = (cw, ch)
        _SINK["last"].pop(key, None)
    prev = _SINK["last"].get(key)
    if _SKPROF:
        if prev is not None and prev[0] is src:
            _SK["same_obj"] += 1          # generator returned a CACHED array
        else:
            _SK["new_obj"] += 1           # generator built a FRESH array
    if prev is None or prev[0] is not src:
        _u0 = _bp_now()
        r = gl.write_dyn(key, src)
        _SINK["upload_s"] += _bp_now() - _u0
        _SINK["uploads"] += 1
        if r is None:
            _SINK["overflow"] += 1
            return
        _SINK["last"][key] = (src, r)
    else:
        r = prev[1]
    _w, _h, us, vs = r
    x0, y0, fw, fh = _anchor_topleft(src, x, y, anchor)
    _SINK["sprites"].append(Sprite(x0 + fw / 2.0, y0 + fh / 2.0, fw, fh,
                                   texture_key=key, color=(1.0, 1.0, 1.0, 1.0),
                                   uv_scale=(us, vs)))


def _blit_inner(rgb, src, x, y, anchor="tl"):
    h, w = src.shape[:2]
    if "r" in anchor:
        x -= w
    elif "c" in anchor[1:]:
        x -= w // 2
    if "b" in anchor:
        y -= h
    x, y = int(round(x)), int(round(y))
    terms = _pm_terms(src)
    if terms is None:
        return
    inv, sa, oy, ox = terms
    # shift to the cropped rect (anchoring above used the FULL src shape,
    # exactly as before)
    x += ox
    y += oy
    h, w = inv.shape[:2]
    H, W = rgb.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    crop = (slice(y0 - y, y1 - y), slice(x0 - x, x1 - x))
    region = rgb[y0:y1, x0:x1, :3].astype(np.float32)
    region = region * inv[crop] + sa[crop]
    rgb[y0:y1, x0:x1, :3] = _to8(region)


class _Roll:
    """lazer RollingCounter tween: the DISPLAYED value chases the target over
    ROLL_MS with an OutQuint ease, instead of snapping. Frame-rate independent
    -- driven by the gameplay clock t (ms), not a per-frame step -- and, like
    lazer's RollingCounter, restarts from the current interpolated value when
    the target changes mid-roll. Snaps on the first frame / seek / rewind."""

    def __init__(self, duration_ms=250.0):
        self.dur = float(duration_ms)
        self._from = 0.0
        self._to = None
        self._start = 0.0
        self._disp = 0.0
        self._last_t = None

    def update(self, target, t):
        target = float(target)
        t = float(t)
        # first frame / seek / rewind / big gap -> snap (no phantom roll)
        if (self._last_t is None or t < self._last_t
                or (t - self._last_t) > 1000.0):
            self._disp = self._from = target
            self._to = target
            self._start = t
            self._last_t = t
            return self._disp
        self._last_t = t
        if self._to is None or target != self._to:   # new target -> new roll
            self._from = self._disp
            self._to = target
            self._start = t
        p = (t - self._start) / self.dur if self.dur > 0 else 1.0
        if p >= 1.0:
            self._disp = self._to
        elif p <= 0.0:
            self._disp = self._from
        else:
            q = 1.0 - p
            e = 1.0 - q * q * q * q * q            # Easing.OutQuint
            self._disp = self._from + (self._to - self._from) * e
        return self._disp


class ArgonHud:
    def __init__(self, resolution, meta, bm, first, last, sim, cfg=None):
        self.w, self.h = resolution
        # GL sink state (see _sink_blit): reserved texture size and last-uploaded
        # array, per HUD element. Per-instance so a second render in-process cannot
        # inherit stale textures.
        self._sink_caps: dict = {}
        self._sink_last: dict = {}
        self.meta = meta
        self.bm = bm
        self.first = first
        self.last = max(last, first + 1)
        self.sim = sim
        self.cfg = cfg
        self.counter = ArgonCounter()
        # lazer RollingCounter: displayed score/accuracy roll to the sim value
        # over ~250ms (OutQuint) rather than snapping when a note lands.
        self._roll_score = _Roll(250.0)
        self._roll_acc = _Roll(250.0)
        self.bold = get_font("Bold")
        self.semi = get_font("SemiBold")
        from pathlib import Path
        wp = Path(__file__).resolve().parent / "glyphs" / "argon_wedge.png"
        self.wedge = np.array(Image.open(wp).convert("RGBA")) if wp.is_file() else None
        self._wedge_scaled = None
        self._results = None        # cached results-screen overlay (RGBA)
        # Per-element skin HUD: the skin's digit font (score/combo) + scorebar HP
        # bar when present, else Argon fallback. Argon keeps its own chrome
        # (progress / key counter / hit-error / title) that the legacy HUD lacks.
        from osu_taiko_renderer.hud.skin_hud_elements import (SkinDigitFont,
                                                              SkinHealthBar)
        _sk = getattr(sim, "skin", None)
        self._sfont = SkinDigitFont(_sk, getattr(_sk, "score_prefix", "score"),
                                    "-comma -dot -percent -x",
                                    (getattr(_sk, "score_overlap", 0) or 0) / 100.0)
        self._cfont = SkinDigitFont(_sk, getattr(_sk, "combo_prefix", "combo"),
                                    "-x", (getattr(_sk, "combo_overlap", 0) or 0) / 100.0)
        self._hpbar = SkinHealthBar(_sk)
        self._tint_cache: dict = {}   # (id(glyph), colour) -> (glyph, tinted)
        from osu_taiko_renderer.hud.hud_layout import SkinHudLayout
        self.layout = SkinHudLayout(getattr(cfg, "skin_dir", None))

        # Attached from outside by render.py after construction (the leaderboard
        # is baked once, and the featured avatar is read off disk there). They MUST
        # be declared here even though nothing in this file assigns them: under
        # mypyc this class compiles to a native class with a fixed attribute
        # table and NO __dict__, so `hud.board = ...` from another module raises
        # AttributeError on anything not listed. render.py wraps that attach in
        # `except Exception: pass`, so the failure is SILENT -- the leaderboard and
        # avatar would just vanish from the results screen with nothing logged.
        self.board = None
        self.featured_avatar_bytes = None
        self._sink_max_tex = 0      # GL_MAX_TEXTURE_SIZE, queried once
        self._sink_warned = False

    def _label(self, text, px, color=_LABEL):
        return self.bold.render(text, px, color=color)

    def _num(self, font, text, px):
        """Skin digit font if the skin ships one, else the Argon LED counter."""
        text = text
        if font is not None and font.present:
            return font.render(text, px)
        return self.counter.render(text, px)

    def _tint_lum(self, img, color):
        """Recolour an RGBA glyph to `color` by luminance (keeps shape + alpha)
        so the skin's cyan digit font can be coloured per judgement.

        MEMOISED. The only caller is the GREAT/OK/MISS hit counter, which called
        this three times EVERY frame on bitmaps that come straight out of the
        font's own cache -- so the same (glyph, colour) pair was recoloured over
        and over while the counts sat unchanged between hits. Each call was a
        full float32 convert + luminance + 3 channel writes + _to8 over the whole
        glyph. Ablation measured the hit counter at 0.257 ms/frame, 24% of the
        entire HUD cost, in 6 sprites.

        Identical inputs give an identical array, so this is byte-identical --
        only recomputation is skipped. The id() key is safe because the entry
        keeps a strong ref to the source (its id cannot be reused while cached)
        and identity is re-validated on hit, the same contract as
        argon/compositor._SCALE_CACHE.
        """
        key = (id(img), color)
        hit = self._tint_cache.get(key)
        if hit is not None and hit[0] is img:
            return hit[1]
        out = img.astype(np.float32)
        lum = (0.299 * out[..., 0] + 0.587 * out[..., 1] + 0.114 * out[..., 2]) / 255.0
        for _i in range(3):
            out[..., _i] = lum * color[_i]
        res = _to8(out)
        if len(self._tint_cache) > 256:
            self._tint_cache.clear()
        self._tint_cache[key] = (img, res)
        return res

    def _key_active(self, t, window=110):
        """Whether each key (B1..B4 = rim-L, centre-L, centre-R, rim-R) had a
        press within the last `window` ms — lights its activity bar."""
        import bisect
        out = []
        for z in ("rl", "cl", "cr", "rr"):
            edges = self.sim._zedges[z]
            i = bisect.bisect_right(edges, t) - 1
            out.append(i >= 0 and (t - edges[i]) <= window)
        return out

    def prepare_numbers(self, scene, gl):
        """Compose the score and combo images and emit them as GL sprites, at
        DRAW time (before readback) so they land inside the main pass.

        Called once per frame from the render loop; `overlay` then reuses the
        images stashed here instead of recomputing them. That split matters
        because `_roll_score` is a stateful RollingCounter — it must advance
        exactly once per frame, in frame order, or the displayed score tween
        diverges. Everything else in the HUD still composites on the CPU after
        readback, so it draws OVER these, as it did before.

        Returns [] (and stashes nothing) if GL is unavailable or the image is
        larger than the reserved texture, so `overlay` falls back to the blits.
        """
        if not _GPU_NUM or gl is None:
            return []
        from osu_taiko_renderer.argon.counter import _CSTAT as _cs
        _cs["frames"] += 1
        w, h, t = self.w, self.h, scene.time_ms
        mx, my = int(w * 0.018), int(h * 0.03)
        out = []
        disp_score = self._roll_score.update(scene.score, t)
        sc = self._num(self._sfont, str(int(disp_score)), h * 0.056)
        gl.dyn_texture("hud_score", int(w * 0.5), int(h * 0.12))
        _sp = self.layout.place("score", sc.shape[1], sc.shape[0], w, h, mx, my)
        if _sp is not None:
            s_sprite = _num_sprite(gl, "hud_score", sc,
                                   int(_sp[0]), int(_sp[1]), "tl")
        else:
            s_sprite = _num_sprite(gl, "hud_score", sc, w - mx, my, "tr")
        ctex = self._num(self._cfont, f"{int(scene.combo)}x", h * 0.072)
        gl.dyn_texture("hud_combo", int(w * 0.5), int(h * 0.14))
        _bp = self.layout.place("combo", ctex.shape[1], ctex.shape[0], w, h, mx, my)
        if _bp is not None:
            c_sprite = _num_sprite(gl, "hud_combo", ctex,
                                   int(_bp[0]), int(_bp[1]), "tl")
        else:
            c_sprite = _num_sprite(gl, "hud_combo", ctex, mx, h - my, "bl")
        if s_sprite is None or c_sprite is None:
            self._pre = None            # fall back wholesale, never half-and-half
            return []
        out.append(s_sprite)
        out.append(c_sprite)
        self._pre = (sc, ctex, disp_score)
        return out

    def overlay_gl(self, scene, gl):
        """Run the whole HUD in sprite-emitting mode and return the GL sprites.

        Called at DRAW time so the sprites land inside the main pass. `overlay` itself
        is reused verbatim — every element is still composed on the CPU byte-for-byte
        as before; only the composite onto the frame becomes a GL draw. That is why
        this needs no layout change: nothing moves, unlike a reserved-grid rework.

        Returns None if anything overflowed its reserved texture, so the caller can
        fall back to the CPU path for that frame rather than dropping an element."""
        global _SINK
        # GL_MAX_TEXTURE_SIZE queried ONCE per renderer, not per frame: it is a
        # driver constant and ctx.info builds a dict of every GL limit. Fallback 4096
        # is the GL 3.3 floor every conformant implementation guarantees.
        _mt = getattr(self, "_sink_max_tex", 0)
        if not _mt:
            try:
                _mt = int(gl.ctx.info.get("GL_MAX_TEXTURE_SIZE", 4096))
            except Exception:  # noqa: BLE001 — a limit query must never break a render
                _mt = 4096
            self._sink_max_tex = _mt
        _SINK = {"gl": gl, "sprites": [], "caps": self._sink_caps,
                 "last": self._sink_last, "overflow": 0, "seen": {},
                 "w": self.w, "h": self.h, "upload_s": 0.0, "uploads": 0,
                 "max_tex": _mt}
        try:
            if _HUD_CPROF:
                # Wrapped HERE, not with `python -m cProfile`: the HUD runs on the
                # composite thread and cProfile only follows the thread that enabled
                # it, so a top-level profile shows an empty HUD. Enable/disable per
                # frame distorts absolute times, so read the CALL COUNTS from this and
                # take timings from R3D_TAIKO_STAGE instead.
                _CP.enable()
                try:
                    self.overlay(_ShimFrame(self.h, self.w), scene)
                finally:
                    _CP.disable()
            else:
                self.overlay(_ShimFrame(self.h, self.w), scene)
            _SINK_STATS["upload_s"] += _SINK["upload_s"]
            _SINK_STATS["uploads"] += _SINK["uploads"]
            _SINK_STATS["frames"] += 1
            _SK["frames"] += 1
            _SINK_STATS["sprites"] += len(_SINK["sprites"])
            if _SINK["overflow"]:
                # Loud once: a silent per-frame fallback is WORSE than not moving the
                # HUD at all, because overlay then runs twice per frame.
                if not getattr(self, "_sink_warned", False):
                    import sys as _w
                    print("[taiko-renderer] GPU HUD: element overflowed its reserved "
                          "texture — falling back to the CPU overlay this frame",
                          file=_w.stderr)
                    self._sink_warned = True
                return None
            return _SINK["sprites"]
        finally:
            _SINK = None

    def overlay(self, rgb: Any, scene) -> Any:
        # rgb is Any, not np.ndarray: overlay_gl passes a _ShimFrame duck-type
        # (only .shape is read) so the GL path never allocates a full frame.
        # rgb is mutated in place (writable flipped view from the PBO pop) —
        # the old full-frame ascontiguousarray copy was pure render-thread cost.
        if _BLIT_PROF:
            _c, _d = _BP.get(("frames",), (0, 0.0))
            _BP[("frames",)] = (_c + 1, _d)
        w, h, t = self.w, self.h, scene.time_ms
        mx, my = int(w * 0.018), int(h * 0.03)
        lab_px = max(11, int(h * 0.016))
        # skin scorebar HP bar (top-left) — Argon has none; push the top-left
        # score block below it. No skin scorebar -> no bar (unchanged Argon).
        hp_h = self._hpbar.draw(rgb, w, h, scene.hp, _blit) if self._hpbar.present else 0

        # --- score: skin-positioned (MainHUDComponents.json) or top-RIGHT ---
        # prepare_numbers() already advanced the roll and composed these images
        # this frame, and drew them as GL sprites — reuse them for layout and
        # skip their blits. `_pre` is consumed so a stale entry can never leak
        # into a later frame.
        _pre = getattr(self, "_pre", None)
        self._pre = None
        if _pre is not None:
            sc, _pre_ctex, disp_score = _pre
        else:
            disp_score = self._roll_score.update(scene.score, t)
            sc = self._num(self._sfont, str(int(disp_score)), h * 0.056)
        _sp = self.layout.place("score", sc.shape[1], sc.shape[0], w, h, mx, my)
        if _sp is not None:
            _sx, _sy = _sp
            if _pre is None:
                _blit(rgb, sc, _sx, _sy, "tl", key="score_placed")
            s_right, s_top = _sx + sc.shape[1], _sy
        else:
            if _pre is None:
                _blit(rgb, sc, w - mx, my, "tr", key="score_default")
            s_right, s_top = w - mx, my

        # --- accuracy (directly below the score block) ---
        disp_acc = self._roll_acc.update(scene.accuracy, t)
        pct = max(0.0, min(100.0, disp_acc * 100.0))
        ah = h * 0.05
        ay = s_top + sc.shape[0] + int(h * 0.010)
        if self._sfont.present:
            _blit(rgb, self._sfont.render(f"{pct:.2f}%", ah * 0.62),
                  s_right, ay, "tr", key="acc_pct_skinfont")
        else:
            whole = str(int(pct))
            frac = f"{pct:06.2f}".split(".")[1]
            pcttex = self.counter.render("%", ah * 0.5)
            _blit(rgb, pcttex, s_right, ay, "tr", key="acc_pct_sign")
            fx = s_right - pcttex.shape[1]
            fractex = self.counter.render(frac, ah * 0.5)
            _blit(rgb, fractex, fx, ay, "tr", key="acc_frac")
            dottex = self.counter.render(".", ah * 0.5)
            _blit(rgb, dottex, fx - fractex.shape[1], ay, "tr", key="acc_dot")
            wholetex = self.counter.render(whole, ah)
            _blit(rgb, wholetex, fx - fractex.shape[1] - dottex.shape[1], ay, "tr", key="acc_whole")

        # --- pp (below accuracy) ---
        ppy = ay + int(ah) + int(lab_px * 0.6)
        _blit(rgb, self._label("PP", lab_px), s_right, ppy, "tr", key="pp_label")
        pptex = self.counter.render(str(int(round(scene.pp))), h * 0.03)
        _blit(rgb, pptex, s_right, ppy + int(lab_px * 1.2), "tr", key="pp_value")

        # --- mods (below PP): coloured acronym pills ---
        mods = mod_acronyms(int(getattr(self.meta, "mods", 0) or 0))
        if mods:
            my2 = ppy + int(lab_px * 1.2) + int(h * 0.03) + int(h * 0.012)
            px = int(round(s_right))   # slice indices must be int (was float -> crash on modded plays)
            pill_px = max(12, int(h * 0.02))
            for ac in reversed(mods):
                tex = self.bold.render(ac, pill_px, color=_WHITE)
                pad = int(pill_px * 0.4)
                pw, ph = tex.shape[1] + pad * 2, tex.shape[0] + pad
                x0 = px - pw
                _rect(rgb, my2, my2 + int(ph), x0, px, (44, 50, 60))
                _blit(rgb, tex, x0 + pad, my2 + pad // 2, "tl", key="mod_pill")
                px = x0 - int(w * 0.006)

        # --- hit counter (mid-left): GREAT / OK / MISS, per --hit-counter ---
        if getattr(self.cfg, "show_hit_counter", True) and getattr(scene, "counts", None) is not None:
            _g, _o, _m = scene.counts
            _cnp = h * 0.055
            _clp = self.layout.place("counter", int(w * 0.10), int(h * 0.22), w, h, mx, my)
            if _clp is not None:
                _cx, _cyy = int(_clp[0]), int(_clp[1])
            else:
                _cx = mx
                _cyy = int(self.sim.geo.center_y + self.sim.geo.pf_h / 2.0) + int(h * 0.045)
            for _v, _lb, _cl in ((_g, "GREAT", (95, 200, 255)),
                                 (_o, "OK", (150, 235, 90)),
                                 (_m, "MISS", (255, 95, 95))):
                if self._sfont.present:
                    _nt = self._tint_lum(self._sfont.render(str(int(_v)), _cnp), _cl)
                else:
                    _nt = self.bold.render(str(int(_v)), _cnp, color=_cl)
                _blit(rgb, _nt, _cx, _cyy, "tl", key="hitcount_num")
                _blit(rgb, self._label(_lb, lab_px, color=_cl),
                      _cx, _cyy + _nt.shape[0] - int(lab_px * 0.1), "tl", key="hitcount_label")
                _cyy += _nt.shape[0] + int(lab_px * 1.7)

        # --- combo: skin-positioned or bottom-left ---
        ctex = (_pre_ctex if _pre is not None
                else self._num(self._cfont, f"{int(scene.combo)}x", h * 0.072))
        _bp = self.layout.place("combo", ctex.shape[1], ctex.shape[0], w, h, mx, my)
        if _bp is not None:
            _bx, _by = int(_bp[0]), int(_bp[1])
            if _pre is None:
                _blit(rgb, ctex, _bx, _by, "tl", key="combo_placed")
            # the COMBO label stays on the CPU — it is static chrome, not a
            # per-frame number, so it has nothing to gain from a dyn texture
            _blit(rgb, self._label("COMBO", lab_px, color=_ACCENT),
                  _bx, _by - int(lab_px * 1.2), "tl", key="combo_label_placed")
        else:
            if _pre is None:
                _blit(rgb, ctex, mx, h - my, "bl", key="combo_default")
            _blit(rgb, self._label("COMBO", lab_px, color=_ACCENT),
                  mx, h - my - ctex.shape[0] - int(lab_px * 0.4), "bl", key="combo_label_default")

        # --- key counter B1–B4 (bottom-right): activity bar / label / count,
        # each column centred (ArgonKeyCounter layout) ---
        counts = self.sim.key_counts(t)
        active = self._key_active(t)
        kw = int(w * 0.042)
        right = w - mx
        bar_w, bar_h = int(kw * 0.62), max(2, int(h * 0.004))
        for i in range(4):           # left→right B1..B4
            c = counts[i]
            cx = right - (3 - i) * kw - kw // 2
            num = self.bold.render(str(c), int(h * 0.026), color=_WHITE)
            lab = self.bold.render(f"B{i + 1}", lab_px, color=_LABEL)
            base_y = h - my
            _blit(rgb, num, cx, base_y, "bc", key="keycount_num")
            _blit(rgb, lab, cx, base_y - num.shape[0] - 2, "bc", key="keycount_label")
            by = base_y - num.shape[0] - lab.shape[0] - 8
            col = _WHITE if active[i] else (70, 78, 90)
            _rect(rgb, by, by + bar_h, cx - bar_w // 2, cx + bar_w // 2, col)

        # --- song progress (bottom centre) ---
        # prog_frac, not frac: the accuracy block above binds `frac` to a STR
        # (the decimal digits), and reusing the name for a float is a type
        # conflict that mypyc turns into a hard runtime failure.
        prog_frac = max(0.0, min(1.0, (t - self.first) / (self.last - self.first)))
        bar_y = h - int(h * 0.012)
        bx0, bx1 = int(w * 0.16), int(w * 0.84)
        _rect(rgb, bar_y, bar_y + 3, bx0, bx1, (70, 78, 90))
        fillx = bx0 + int((bx1 - bx0) * prog_frac)
        _rect(rgb, bar_y, bar_y + 3, bx0, fillx, _ACCENT)
        _blit(rgb, self.semi.render(_fmt_time(t - self.first),
                                    int(h * 0.016), color=_LABEL),
              bx0, bar_y - 4, "bl", key="time_elapsed")
        _blit(rgb, self.semi.render(_fmt_time(self.last - self.first),
                                    int(h * 0.016), color=_LABEL),
              bx1, bar_y - 4, "br", key="time_total")

        # --- hit-error / UR bar (bottom centre, above the progress bar) ---
        self._draw_hit_error(rgb, t)

        # --- player / title (top centre, spectator-style) ---
        title = f"{self.bm.artist} - {self.bm.title} [{self.bm.version}]".strip(" -")
        info = self.semi.render(f"{self.meta.player_name}  ·  {title}",
                                int(h * 0.018), color=_LABEL)
        _blit(rgb, info, w // 2, my, "tc", key="title_info")
        return rgb

    def _draw_hit_error(self, rgb, t):
        """Horizontal hit-error meter (lazer BarHitErrorMeter): OK/GREAT window
        bands, centre line, recent-hit ticks (fading), and the UR value."""
        w, h = self.w, self.h
        gw = getattr(self.sim, "great_w", 35.0)
        ow = getattr(self.sim, "ok_w", 80.0)
        if ow <= 0:
            return
        cx = w // 2
        half = int(w * 0.11)
        by = h - int(h * 0.05)
        bh = max(4, int(h * 0.012))
        scale = half / ow
        great_blue = (0x66, 0xcc, 0xff)
        ok_green = (0x88, 0xb3, 0x00)
        # window bands (dim base, then OK band, then GREAT band on top)
        gwp = int(gw * scale)
        _rect(rgb, by, by + bh, cx - half, cx + half, (40, 46, 56))
        _fillband(rgb, cx - int(ow * scale), cx + int(ow * scale), by, bh, ok_green, 0.5)
        _fillband(rgb, cx - gwp, cx + gwp, by, bh, great_blue, 0.6)
        # centre line
        _rect(rgb, by - 3, by + bh + 3, cx - 1, cx + 1, _WHITE)
        # recent hit ticks (fade with age)
        for err, res, age in self.sim.recent_errors(t):
            x = int(cx + max(-ow, min(ow, err)) * scale)
            a = max(0.0, 1.0 - age / 3500.0)
            col = great_blue if res == "great" else ok_green
            ty0, ty1 = by - int(bh * 0.8), by + bh + int(bh * 0.8)
            if _SINK is not None:
                _sink_solid(max(0, x - 1), x + 2, ty0, ty1, col, a)
            else:
                seg = rgb[ty0:ty1, max(0, x - 1):x + 2, :3].astype(np.float32)
                seg = seg * (1 - a) + np.array(col, np.float32) * a
                rgb[ty0:ty1, max(0, x - 1):x + 2, :3] = np.clip(
                    seg, 0, 255).astype(np.uint8)
        # UR value
        ur = self.sim.ur_at(t)
        if ur > 0:
            tex = self.bold.render(f"{ur:.2f} UR", int(h * 0.016),
                                   color=_LABEL)
            _blit(rgb, tex, cx, by - bh - 4, "bc", key="ur_value")

    # --- results screen (Argon ranking panel) -------------------------------

    def _compose(self, ov, src, x, y, anchor="tl"):
        """Alpha-composite RGBA uint8 `src` onto RGBA float overlay `ov`."""
        h, w = src.shape[:2]
        if "r" in anchor:
            x -= w
        elif "c" in anchor[1:]:
            x -= w // 2
        if "b" in anchor:
            y -= h
        x, y = int(round(x)), int(round(y))
        H, W = ov.shape[:2]
        x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
        if x1 <= x0 or y1 <= y0:
            return
        s = src[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32)
        a = s[..., 3:4] / 255.0
        reg = ov[y0:y1, x0:x1]
        reg[..., :3] = reg[..., :3] * (1 - a) + s[..., :3] * a
        reg[..., 3:4] = np.clip(reg[..., 3:4] + a * (255 - reg[..., 3:4]), 0, 255)

    def _build_results(self):
        w, h = self.w, self.h
        m, bm = self.meta, self.bm
        ov = np.zeros((h, w, 4), np.float32)
        ov[..., 3] = 255                               # opaque black bg: clean results, no gameplay-HUD bleed (like std/catch)
        cx = w // 2
        grade = str(getattr(m, "grade", "D") or "D")
        mods_i = int(getattr(m, "mods", 0) or 0)
        silver = bool(mods_i & ((1 << 3) | (1 << 10)))      # HD or FL → silver SS/S
        if silver and grade.upper() in ("SS", "S"):
            gcol = _GRADE_COL["SSH"]
        else:
            gcol = _GRADE_COL.get(grade.upper(), _WHITE)
        # FEATURED player avatar — circular chip, top-centre, above the title.
        try:
            from osu_taiko_renderer.hud.lb_cards import bake_avatar_circle
            _av_px = int(h * 0.11)
            _chip = bake_avatar_circle(_av_px, m.player_name,
                                       getattr(self, "featured_avatar_bytes", None))
            self._compose(ov, np.array(_chip), cx, int(h * 0.045), "tc")
        except Exception:  # noqa: BLE001 — avatar never breaks the results card
            pass
        # title / difficulty / player (top) — nudged down to clear the avatar chip
        title = f"{bm.artist} - {bm.title}".strip(" -")
        self._compose(ov, self.semi.render(title, int(h * 0.026), color=_WHITE),
                      cx, int(h * 0.135), "tc")
        self._compose(ov, self.bold.render(
            f"[{bm.version}]   played by {m.player_name}", int(h * 0.018),
            color=_LABEL), cx, int(h * 0.175), "tc")
        # big grade letter
        self._compose(ov, self.bold.render(grade, int(h * 0.24), color=gcol),
                      cx, int(h * 0.15), "tc")
        # score (segmented), centred
        sctex = self.counter.render(str(int(m.score)), h * 0.085)
        self._compose(ov, sctex, cx, int(h * 0.43), "tc")
        # accuracy | max combo | pp   (label above value)
        pct = max(0.0, min(100.0, float(getattr(m, "accuracy", 0.0))))
        pp = int(round(getattr(self.sim, "_final_pp", 0.0)))
        cells = [("ACCURACY", f"{pct:.2f}".replace("100.00", "100") + "%"),
                 ("MAX COMBO", f"{int(getattr(m, 'max_combo', 0))}x"),
                 ("PP", str(pp))]
        cw = int(w * 0.16)
        y_lab = int(h * 0.56)
        for i, (lab, val) in enumerate(cells):
            ccx = cx + (i - 1) * cw
            self._compose(ov, self.bold.render(lab, int(h * 0.016), color=_LABEL),
                          ccx, y_lab, "tc")
            self._compose(ov, self.counter.render(val, h * 0.044),
                          ccx, y_lab + int(h * 0.028), "tc")
        # GREAT / OK / MISS counts (coloured)
        g3 = (int(getattr(m, "count_300", 0)), int(getattr(m, "count_100", 0)),
              int(getattr(m, "count_miss", 0)))
        judge = [("GREAT", g3[0], _GRADE_COL["B"]), ("OK", g3[1], (0x88, 0xb3, 0x00)),
                 ("MISS", g3[2], (0xed, 0x11, 0x21))]
        y_j = int(h * 0.69)
        # jlab/jval/jcol, not lab/val/col: the `cells` loop above already bound
        # `val` to str, and rebinding it to int here is a type conflict that mypyc
        # turns into a hard failure (the interpreter never cared).
        for i, (jlab, jval, jcol) in enumerate(judge):
            ccx = cx + (i - 1) * cw
            self._compose(ov, self.bold.render(jlab, int(h * 0.017), color=jcol),
                          ccx, y_j, "tc")
            self._compose(ov, self.counter.render(str(jval), h * 0.04),
                          ccx, y_j + int(h * 0.026), "tc")
        # mods
        mods = mod_acronyms(int(getattr(m, "mods", 0) or 0))
        if mods:
            row = "  ".join(mods)
            self._compose(ov, self.bold.render(row, int(h * 0.022), color=_ACCENT),
                          cx, int(h * 0.82), "tc")
        return ov.astype(np.uint8)

    def draw_results(self, rgb, op):
        """Cross-fade the Argon results overlay onto the frozen final frame, then
        composite the per-map leaderboard flank cards (fading in with `op`)."""
        if self._results is None:
            self._results = self._build_results()
        ov = self._results
        a = (ov[..., 3:4].astype(np.float32) / 255.0) * max(0.0, min(1.0, op))
        out = rgb.astype(np.float32) * (1 - a) + ov[..., :3].astype(np.float32) * a
        out = _to8(out)
        # flank leaderboard: composited per-frame at the results opacity so the
        # cards unfurl with the fade. Only when a board was attached; fully
        # fail-soft — a board must never break a render.
        if getattr(self, "board", None) is not None:
            try:
                from osu_taiko_renderer.hud.lb_cards import draw_board
                pim = Image.fromarray(out, "RGB").convert("RGBA")
                draw_board(pim, self.board, max(0.0, min(1.0, op)))
                out = np.asarray(pim.convert("RGB"))
            except Exception:  # noqa: BLE001 — leaderboard never breaks a render
                pass
        return out


def _fmt_time(ms):
    s = max(0, int(ms // 1000))
    return f"{s // 60}:{s % 60:02d}"
