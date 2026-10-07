"""Additive effects pass over the readback frame: hit explosions and floating
GREAT/OK/MISS judgement text — both additive-blended like osu!lazer Argon.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageChops, ImageFilter

from . import _const as C
from .font import get_font
from .textures import (bake_drum_flash, bake_explosion, bake_note_flash,
                       bake_ring, _GLOW_PAD)

_JUDGE_TEXT = {"great": "GREAT", "ok": "OK", "miss": "MISS"}
_JUDGE_COL = {"great": C.JUDGE_GREAT, "ok": C.JUDGE_OK, "miss": C.JUDGE_MISS}

# perf: per-(texture, size[, tint]) caches of the resize + float32 conversion
# work that used to run per active popup per frame. The cached arrays hold the
# exact same values the inline computation produced (same resize filter, same
# op order), so the composited output is bit-identical — only recomputation is
# skipped. id() keys are safe because the cache keeps a strong ref to the
# source array (its id can't be reused while cached); bounded by a hard clear.
import os as _os_fx
from osu_taiko_renderer.render.envflag import envflag
from osu_taiko_renderer.render.envflag import GPU_FX as _GPU_FX, GPU_SJ as _GPU_SJ  # noqa: E402
# R3D_TAIKO_MERGE_RUNS: emit ALL ring pieces (additive) then ALL popups (straight
# alpha), instead of ring-then-popup per judgement. The per-judgement interleave
# creates 2.91 single-sprite blend runs per frame, and a run of one sprite pays full
# per-run overhead. Grouping collapses them.
#
# NOT free: it reorders a popup relative to a LATER judgement's rings. Both originate
# at the hit target and the popup rises from it, so they can overlap while both are
# alive — where they do, the composite differs. Measured before trusting.
from osu_taiko_renderer.render.envflag import MERGE_RUNS as _MERGE_RUNS  # noqa: E402

# R3D_TAIKO_GPU_SJ: move the two effects GPU_FX left behind on the CPU into the
# GL pass — the SKIN judgement burst (LegacyHitExplosion: the skin's
# taiko-hit300/100/0 sprite punching at the hit target) and the legacy-lane
# hit flash. Needs GPU_FX.
#
# These were skipped when the fixture was skinless, i.e. "unmeasured". Every
# real skin ships taiko-hit*, so on the production path the skin burst is the
# ENTIRE remaining `effects` stage: with this on, composite() is a no-op.
#
# Pre-existing gap this also closes: under GPU_FX alone the legacy-lane hit
# flash is dropped silently (composite() sees exps=() and gpu_fx_sprites never
# emitted it), so a skin shipping taiko-bar-right but NO taiko-hit* loses its
# hit-target flash. Only reachable with GPU_FX on, which is not Red's default.

_SCALE_CACHE: dict = {}     # (id(tex), tw, th, tint) -> (tex, src_rgb, a01, oy, ox)
_SCALE_CACHE_MAX = 256


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


def _scaled_f32(tex, tw, th, tint=None):
    """(src_rgb float32 [tinted], alpha/255 float32, oy, ox) of `tex` resized
    to tw×th — cached, cropped to the tight alpha>0 bounding box (compositing
    where a==0 is an exact float no-op, so skipping those pixels is
    bit-identical). Values match the previous per-call computation exactly.
    Returns None when the resized texture is fully transparent."""
    key = (id(tex), tw, th, tint)
    hit = _SCALE_CACHE.get(key)
    if hit is not None and hit[0] is tex:
        return hit[1]
    # BILINEAR (not LANCZOS): this runs per active popup per frame and the
    # textures are soft glows/text — the quality difference is invisible, the
    # speedup is ~3x. (One-time bakes stay LANCZOS.)
    im8 = np.asarray(Image.fromarray(tex).resize((tw, th), Image.BILINEAR))
    mask = im8[..., 3] != 0
    rows = mask.any(axis=1)
    if not rows.any():
        entry = None
    else:
        cols = mask.any(axis=0)
        y0 = int(np.argmax(rows))
        y1 = len(rows) - int(np.argmax(rows[::-1]))
        x0 = int(np.argmax(cols))
        x1 = len(cols) - int(np.argmax(cols[::-1]))
        im = im8[y0:y1, x0:x1].astype(np.float32)
        src_rgb, a01 = im[..., :3], im[..., 3:4] / 255.0
        if tint is not None and tint != (1.0, 1.0, 1.0):
            src_rgb = src_rgb * np.asarray(tint, dtype=np.float32)
        entry = (src_rgb, a01, y0, x0)
    if len(_SCALE_CACHE) > _SCALE_CACHE_MAX:
        _SCALE_CACHE.clear()
    _SCALE_CACHE[key] = (tex, entry)
    return entry


def _add_tex(rgb, tex, cx, cy, tw, th, intensity, tint=(1.0, 1.0, 1.0)):
    """Additive-blend RGBA `tex` (resized to tw×th) centred at (cx,cy)."""
    tw, th = int(round(tw)), int(round(th))
    if tw < 1 or th < 1 or intensity <= 0:
        return
    entry = _scaled_f32(tex, tw, th, tint)
    if entry is None:
        return
    src_rgb, a01, oy, ox = entry
    ch, cw = src_rgb.shape[:2]
    H, W = rgb.shape[:2]
    # top-left of the FULL resized tex (as before), then the crop offset
    x0, y0 = int(round(cx - tw / 2)) + ox, int(round(cy - th / 2)) + oy
    sx0, sy0 = max(0, -x0), max(0, -y0)
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = min(W, x0 + cw), min(H, y0 + ch)
    if dx1 <= dx0 or dy1 <= dy0:
        return
    h, w = dy1 - dy0, dx1 - dx0
    sa = a01[sy0:sy0 + h, sx0:sx0 + w] * intensity
    sc = src_rgb[sy0:sy0 + h, sx0:sx0 + w]
    region = rgb[dy0:dy1, dx0:dx1, :3].astype(np.float32) + sc * sa
    rgb[dy0:dy1, dx0:dx1, :3] = _to8(region)


_BRIGHT_LUT = None


def bloom(rgb, *, thresh=130, strength=0.55, step=6):
    """Cheap additive bloom (approximates osu!lazer's glow). All work is done in
    C via PIL: downscale → bright-pass (point LUT) → gaussian-blur the small
    image → upscale → saturating add. Avoids per-frame numpy full-frame floats."""
    global _BRIGHT_LUT
    if _BRIGHT_LUT is None:
        _BRIGHT_LUT = [int(max(0, v - thresh) * strength) for v in range(256)]
    img = Image.fromarray(rgb)
    H, W = rgb.shape[:2]
    sw, sh = max(1, W // step), max(1, H // step)
    small = img.resize((sw, sh), Image.BILINEAR).point(_BRIGHT_LUT * 3)
    small = small.filter(ImageFilter.GaussianBlur(radius=max(2, sw * 0.02)))
    up = small.resize((W, H), Image.BILINEAR)
    return np.array(ImageChops.add(img, up))


def _blit_straight(rgb, tex, cx, cy, tw, th, alpha):
    """Straight-alpha composite RGBA `tex` (resized to tw×th) centred at (cx,cy),
    scaled by `alpha` (for fades). For skin judgement images (not additive)."""
    tw, th = int(round(tw)), int(round(th))
    if tw < 1 or th < 1 or alpha <= 0:
        return
    entry = _scaled_f32(tex, tw, th)
    if entry is None:
        return
    src_rgb, a01, oy, ox = entry
    ch, cw = src_rgb.shape[:2]
    H, W = rgb.shape[:2]
    x0, y0 = int(round(cx - tw / 2)) + ox, int(round(cy - th / 2)) + oy
    sx0, sy0 = max(0, -x0), max(0, -y0)
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = min(W, x0 + cw), min(H, y0 + ch)
    if dx1 <= dx0 or dy1 <= dy0:
        return
    h, w = dy1 - dy0, dx1 - dx0
    sa = a01[sy0:sy0 + h, sx0:sx0 + w] * alpha
    sc = src_rgb[sy0:sy0 + h, sx0:sx0 + w]
    region = rgb[dy0:dy1, dx0:dx1, :3].astype(np.float32)
    region = region * (1 - sa) + sc * sa
    rgb[dy0:dy1, dx0:dx1, :3] = _to8(region)


def _prebake_add(im8):
    """(rgb f32, alpha/255 f32, oy, ox, full_h, full_w) of an RGBA uint8
    texture, cropped to the tight alpha>0 bbox (additive blend where a==0
    adds exactly +0.0 — skipping is bit-identical). None = fully transparent."""
    mask = im8[..., 3] != 0
    rows = mask.any(axis=1)
    fh, fw = im8.shape[:2]
    if not rows.any():
        return None
    cols = mask.any(axis=0)
    y0 = int(np.argmax(rows))
    y1 = fh - int(np.argmax(rows[::-1]))
    x0 = int(np.argmax(cols))
    x1 = fw - int(np.argmax(cols[::-1]))
    im = im8[y0:y1, x0:x1].astype(np.float32)
    return (im[..., :3], im[..., 3:4] / 255.0, y0, x0, fh, fw)


def _add_prescaled(rgb, pre, cx, cy, intensity):
    """Additive-blend a prescaled texture centred at (cx,cy). `pre` is the
    _prebake_add tuple — the float32 conversion + /255 normalisation used to
    run per call per frame; values are identical."""
    if pre is None:
        return
    sc_full, a01, oy, ox, fh, fw = pre
    ch, cw = sc_full.shape[:2]
    H, W = rgb.shape[:2]
    # top-left of the FULL texture (as before), then the crop offset
    x0, y0 = int(round(cx - fw / 2)) + ox, int(round(cy - fh / 2)) + oy
    sx0, sy0 = max(0, -x0), max(0, -y0)
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = min(W, x0 + cw), min(H, y0 + ch)
    if dx1 <= dx0 or dy1 <= dy0:
        return
    h, w = dy1 - dy0, dx1 - dx0
    sa = a01[sy0:sy0 + h, sx0:sx0 + w] * intensity
    sc = sc_full[sy0:sy0 + h, sx0:sx0 + w]
    region = rgb[dy0:dy1, dx0:dx1, :3].astype(np.float32) + sc * sa
    rgb[dy0:dy1, dx0:dx1, :3] = _to8(region)


class ArgonEffects:
    def __init__(self, geo, skin_dir=None):
        self.geo = geo
        # Hit explosion tinted by the NOTE accent (don=red / kat=blue), matching
        # lazer's ArgonHitExplosion. The old CENTRE/RIM_HIT_GRAD are the pink/cyan
        # INPUT-DRUM gradients; with bake_explosion's former solid white core they
        # flashed a WHITE disc on every centre hit (a big don read as a big white
        # circle), and being near-white they wash to solid white where the additive
        # bursts overlap on a dense stream. Near-pure accent colours saturate to
        # clean red / blue instead (the dominant channel clips first).
        self.exp = {
            False: bake_explosion((C.DON_TOP, C.DON_BOT), (255, 32, 32, 255)),  # centre/don
            True: bake_explosion((C.KAT_TOP, C.KAT_BOT), (32, 156, 255, 255)),  # rim/kat
        }
        self._ring = bake_ring(64, 8)   # white judgement RingExplosion ring
        self.font = get_font("SemiBold")
        self._jcache: dict[str, np.ndarray] = {}
        self._rng_cache: dict = {}      # (rt, piece, travel) -> (angle, dist)
        # Skin judgement images (taiko-hit300/100/0) — used for the gameplay
        # popups instead of Torus text when the skin provides them.
        from osu_taiko_renderer.skin.taiko_skin import TaikoSkin
        skin = TaikoSkin(skin_dir)
        self._skin_judge = {}
        for res, name in (("great", "taiko-hit300"), ("ok", "taiko-hit100"),
                          ("miss", "taiko-hit0")):
            img = skin.load(name)
            if img is not None:
                self._skin_judge[res] = img
        # Strong/big-note variants (geki GREAT / big OK): skins ship these as
        # taiko-hit300g / taiko-hit100k. Missing -> fall back to the normal
        # result sprite (see _judge_tex).
        for res, name in (("great_big", "taiko-hit300g"), ("ok_big", "taiko-hit100k")):
            img = skin.load(name)
            if img is not None:
                self._skin_judge[res] = img
        # A skin that ships ANY taiko-hit judgement sprite is a LEGACY skin: its
        # hit explosion is the LegacyHitExplosion (that sprite bursting at the
        # target), so the Argon accent glow / hit-target flash is NOT drawn on
        # top (see composite()). Per-result resolution is now per element in
        # _judge_tex: a result the skin SHIPS uses its sprite (a blank/transparent
        # sprite = intentionally invisible, e.g. taiko-hit300 blanked for "no 300
        # popup"); a result the skin OMITS falls back to Argon text (osu's
        # user->default chain — we have no bundled legacy default skin).
        self._use_skin_judge = bool(self._skin_judge)
        # Legacy-lane hit-target flash (issue #118): a skin shipping taiko-bar-right
        # draws its OWN (opaque, dark) hit target, so the Argon per-hit explosion —
        # a soft glow that reads clearly through the translucent Argon target — is
        # nearly invisible over it, i.e. legacy skins never appeared to "flash" the
        # hit on the drum. lazer's hit target flashes the note colour on every hit
        # (don=red / kat=blue, the KiaiHitExplosion/HitTarget accent); mirror that
        # with a solid accent disc at the target, faded on the explosion envelope,
        # so legacy skins flash like Argon. Argon (no legacy lane) is unchanged.
        self._legacy_lane = bool(skin.has("taiko-bar-right"))
        self._hit_flash = {}
        self._hit_flash_rgba = {}     # raw RGBA uint8, for the GL-sprite path
        if self._legacy_lane:
            base = bake_note_flash().astype(np.float32)   # white fill disc
            d = int(round(self.geo.note_d * 0.95))
            # note accent (don=red / kat=blue); additive over the dark target so
            # the dominant channel saturates to a clean red / blue fill.
            for is_rim, tint in ((False, (1.0, 0.05, 0.05)),
                                 (True, (0.05, 0.62, 1.0))):
                col = base.copy()
                col[..., :3] *= np.asarray(tint, np.float32)
                im8 = np.asarray(Image.fromarray(col.astype(np.uint8))
                                 .resize((d, d), Image.LANCZOS))
                self._hit_flash[is_rim] = _prebake_add(im8)
                self._hit_flash_rgba[is_rim] = im8
        # Pre-scale explosion textures to the two note sizes (resizing every
        # frame per active explosion was a render-time hotspot). Stored as
        # _prebake_add tuples (float32 rgb + alpha/255, alpha-bbox-cropped) so
        # the per-frame additive blend skips the per-call /255 normalisation
        # and all fully-transparent margins (identical values either way).
        self._exp_scaled = {}
        self._exp_rgba = {}      # raw RGBA uint8, for the GL-sprite path
        self._drum_rgba = {}
        for is_rim in (False, True):
            for big in (False, True):
                d = int(round(geo.big_d if big else geo.note_d))
                im8 = np.asarray(Image.fromarray(self.exp[is_rim])
                                 .resize((d, d), Image.LANCZOS))
                self._exp_scaled[(is_rim, big)] = _prebake_add(im8)
                self._exp_rgba[(is_rim, big)] = im8
        # Pre-scale the 4 drum-flash quadrants to the drum size.
        dd = int(round(geo.drum_d))
        self._drum_scaled = {}
        for is_rim in (False, True):
            for left in (True, False):
                _dpad = int(round(dd * _GLOW_PAD))
                im8 = np.asarray(Image.fromarray(bake_drum_flash(ring=is_rim, left=left))
                                 .resize((_dpad, _dpad), Image.LANCZOS))
                self._drum_scaled[(is_rim, left)] = _prebake_add(im8)
                self._drum_rgba[(is_rim, left)] = im8

    def gpu_fx_sprites(self, exps, drums):
        """Additive GL sprites for the drum flashes + hit explosions, replacing
        the `_add_prescaled` loops in composite(). Alpha curves are copied
        verbatim from composite() so the intensity per frame is identical; the
        residual difference is GL's round-to-nearest 8-bit blend vs numpy's
        float32 -> clip -> truncate, i.e. ~1 LSB. Positions match
        _add_prescaled, which centres the FULL texture on (cx, cy)."""
        from osu_taiko_renderer.beatmap.models import Sprite
        g = self.geo
        out = []

        def _snap(c, full):
            """_add_prescaled places the texture at INTEGER offsets:
            `x0 = int(round(c - full/2)) + ox`. Centring a GL quad on the raw
            float centre instead (drum_x=126.6, center_y=444.4) puts its edges
            on fractional boundaries, so GL bilinearly resamples the texture by
            up to half a pixel. Measured: max|d| up to 129 on steep glow
            gradients while the TYPICAL differing pixel was only 1 LSB — i.e.
            the tail was resampling, not rounding. Reproduce the CPU's integer
            placement so GL samples texel-aligned."""
            return int(round(c - full / 2.0)) + full / 2.0
        for is_rim, left, a in drums:
            im = self._drum_rgba.get((is_rim, left))
            if im is None or a <= 0.0:
                continue
            fh, fw = im.shape[:2]
            out.append(Sprite(_snap(g.drum_x, fw), _snap(g.center_y, fh), fw, fh,
                              texture_key=f"fx_drum_{int(is_rim)}_{int(left)}",
                              color=(1.0, 1.0, 1.0, float(a)), additive=True))
        for is_rim, age, big, res in (() if self._use_skin_judge else exps):
            if res == "great":
                if age < C.EXPLOSION_GREAT_IN_MS:
                    a = age / C.EXPLOSION_GREAT_IN_MS
                else:
                    f = 1.0 - (age - C.EXPLOSION_GREAT_IN_MS) / C.EXPLOSION_GREAT_OUT_MS
                    a = max(0.0, f) ** 4
            else:
                f = 1.0 - (age - C.EXPLOSION_GREAT_IN_MS) / C.EXPLOSION_OK_OUT_MS
                a = C.EXPLOSION_OK_PEAK * max(0.0, f)
            if a <= 0.001:
                continue
            im = self._exp_rgba.get((is_rim, big))
            if im is None:
                continue
            fh, fw = im.shape[:2]
            out.append(Sprite(_snap(g.target_x, fw), _snap(g.center_y, fh), fw, fh,
                              texture_key=f"fx_exp_{int(is_rim)}_{int(big)}",
                              color=(1.0, 1.0, 1.0, float(a)), additive=True))
            # legacy lane: the solid note-colour flash filling the hit target,
            # same envelope and placement as composite()'s _add_prescaled.
            if _GPU_SJ and self._legacy_lane:
                fim = self._hit_flash_rgba.get(is_rim)
                if fim is not None:
                    gh, gw = fim.shape[:2]
                    out.append(Sprite(_snap(g.target_x, gw),
                                      _snap(g.center_y, gh), gw, gh,
                                      texture_key=f"fx_flash_{int(is_rim)}",
                                      color=(1.0, 1.0, 1.0,
                                             float(min(1.0, a * 0.85))),
                                      additive=True))
        return out

    def bind_gl(self, renderer):
        """Give the compositor the GL renderer so the judgement/ring textures can
        be baked and uploaded on first use. Only called under R3D_TAIKO_GPU_FX."""
        self._gl = renderer
        self._gl_keys: set = set()

    def _upload_resized(self, key, tex, tw, th):
        """Register `tex` resized to tw×th under `key`, once. BILINEAR to match
        _scaled_f32 exactly — the CPU path resizes with PIL BILINEAR and blits
        1:1, so baking at the same size with the same filter and drawing the quad
        1:1 keeps GL on mip level 0 and out of the filter-mismatch trap."""
        gl = getattr(self, "_gl", None)
        if gl is None:
            return None
        if key in self._gl_keys:
            return key
        im8 = np.asarray(Image.fromarray(tex).resize((tw, th), Image.BILINEAR))
        gl.upload_texture(key, im8)
        self._gl_keys.add(key)
        return key

    def _ring_gl(self, si):
        return self._upload_resized(f"fx_ring_{si}", self._ring, si, si)

    def _judge_gl(self, res, big, tw, th):
        tex, _ = self._judge_tex(res, big)
        if tex is None:
            return None
        return self._upload_resized(
            f"fx_judge_{res}_{int(big)}_{tw}x{th}", tex, tw, th)

    def gpu_judge_sprites(self, judges):
        """GL sprites for the Argon judgement bursts: the RingExplosion pieces
        (additive, result-tinted) then the GREAT/OK popup text (straight alpha),
        emitted per judgement in that order so the run-splitting draw reproduces
        lazer's z-order exactly as the CPU chain did.

        The two are moved TOGETHER on purpose. The CPU chain runs ring-then-popup
        per judgement, both after readback; moving only the popups into the GL
        pass would put them UNDER the CPU-composited rings instead of over them.

        Ring pieces are drawn from textures pre-baked at their final integer
        sizes (see _ring_gl) rather than by scaling the 64px source on the GPU:
        the CPU path downscales with PIL BILINEAR, while a scaled GL quad would
        use bilinear+mipmap, and at 64->13 those filters diverge visibly.

        NOT moved: the legacy/skin judgement sprite path (`is_skin`), which needs
        a skin shipping taiko-hit300/100/0 — unmeasured here, so it stays on the
        CPU in composite()."""
        import math
        from osu_taiko_renderer.beatmap.models import Sprite
        g = self.geo
        out = []

        def _snap(c, full_i):
            # matches _add_tex / _blit_straight: int(round(c - tw/2)) + ox
            return int(round(c - full_i / 2.0)) + full_i / 2.0

        popups = [] if _MERGE_RUNS else None
        for res, age, rt, big in judges:
            tex, is_skin = self._judge_tex(res, big)
            if tex is None:
                continue
            if is_skin:
                if not _GPU_SJ:
                    continue                  # skin path stays on the CPU
                # LegacyHitExplosion: the skin sprite BURSTS at the hit target
                # (no upward float). Envelope, size rounding, resize filter and
                # integer placement all copied from composite()'s
                # _blit_straight -> _scaled_f32, which is BILINEAR.
                sscale, salpha = self._legacy_explosion_anim(age)
                if salpha <= 0.01:
                    continue
                sh0, sw0 = tex.shape[0], tex.shape[1]
                stw = int(round(sw0 * sscale))
                sth = int(round(sh0 * sscale))
                if stw < 1 or sth < 1:
                    continue
                skey = self._judge_gl(res, big, stw, sth)
                if skey is None:
                    continue
                ssp = Sprite(_snap(g.target_x, stw), _snap(g.center_y, sth),
                             stw, sth, texture_key=skey,
                             color=(1.0, 1.0, 1.0, float(salpha)))
                (popups if popups is not None else out).append(ssp)
                continue
            # --- RingExplosion pieces (additive, tinted) -------------------
            spec = self._RING_SPEC.get(res)
            ga = max(0.0, (1.0 - age / 1000.0)) ** 5
            if spec is not None and ga > 0.004:
                n_small, n_large, tmult = spec
                col = tuple(c / 255.0 for c in _JUDGE_COL[res][:3])
                sc = g.scale
                travel = 58.0 * sc * tmult
                p = min(age, 600) / 600.0
                rad = 0.6 + 0.4 * (1.0 - (1.0 - p) ** 5)
                pieces = [9.0 * sc] * n_small + [14.0 * sc] * n_large
                for i, size in enumerate(pieces):
                    rkey = (int(rt), i, travel)
                    hit = self._rng_cache.get(rkey)
                    if hit is None:
                        import random
                        rng = random.Random((int(rt) * 1000003) ^ (i * 2654435761))
                        hit = (rng.uniform(0.0, 360.0),
                               rng.uniform(travel / 2.0, travel))
                        if len(self._rng_cache) > 4096:
                            self._rng_cache.clear()
                        self._rng_cache[rkey] = hit
                    d, dist = hit
                    cur = dist * rad
                    si = int(round(size))
                    if si < 1:
                        continue
                    key = self._ring_gl(si)
                    if key is None:
                        continue
                    cx = g.target_x + math.cos(d) * cur
                    cy = g.center_y + math.sin(d) * cur
                    out.append(Sprite(_snap(cx, si), _snap(cy, si), si, si,
                                      texture_key=key,
                                      color=(col[0], col[1], col[2], float(ga)),
                                      additive=True))
            # --- the popup text (straight alpha, on top) -------------------
            pr = age / C.JUDGE_MOVE_MS
            ease = 1.0 - (1.0 - pr) ** 5
            scale = 1.0 + 0.4 * ease
            alpha = max(0.0, (1.0 - pr) ** 5)
            if alpha <= 0.01:
                continue
            yoff = (0.6 + 0.4 * ease) * g.pf_h
            h0, w0 = tex.shape[0], tex.shape[1]
            tw, th = int(round(w0 * scale)), int(round(h0 * scale))
            if tw < 1 or th < 1:
                continue
            key = self._judge_gl(res, big, tw, th)
            if key is None:
                continue
            sp = Sprite(_snap(g.target_x, tw),
                        _snap(g.center_y - yoff, th), tw, th,
                        texture_key=key,
                        color=(1.0, 1.0, 1.0, float(alpha)))
            (popups if popups is not None else out).append(sp)
        if popups:
            out.extend(popups)        # all additive first, then all straight-alpha
        return out

    def gpu_fx_textures(self):
        """{texture_key: RGBA uint8} to upload once before the frame loop."""
        d = {}
        for (is_rim, left), im in self._drum_rgba.items():
            d[f"fx_drum_{int(is_rim)}_{int(left)}"] = im
        for (is_rim, big), im in self._exp_rgba.items():
            d[f"fx_exp_{int(is_rim)}_{int(big)}"] = im
        for is_rim, im in self._hit_flash_rgba.items():
            d[f"fx_flash_{int(is_rim)}"] = im
        return d

    def _judge_tex(self, result, big=False):
        """(tex, is_skin) for a judgement result. Per-element osu resolution:
        the USER skin's sprite when it ships one for this result — including a
        blank/transparent sprite, which renders as nothing downstream (the skin
        intentionally hides that judgement, e.g. taiko-hit300 blanked for a
        "no 300 popup" look) — else the Argon text fallback (osu's user->default
        chain; no bundled legacy default, so Argon is the default). Big/geki
        notes prefer the *g/*k variant (taiko-hit300g / -100k), falling back to
        the normal-note sprite, then to Argon text."""
        if big and (result + "_big") in self._skin_judge:
            key, is_skin = result + "_big", True
        elif result in self._skin_judge:
            key, is_skin = result, True
        else:
            key, is_skin = "argon_" + result, False
        if key not in self._jcache:
            if is_skin:
                img = self._skin_judge[key]
                th = int(self.geo.note_d * 1.1)
                tw = max(1, int(th * img.shape[1] / img.shape[0]))
                self._jcache[key] = np.array(
                    Image.fromarray(img).resize((tw, th), Image.LANCZOS))
            else:
                # ArgonJudgementPiece: plain straight-alpha OsuFont text, no glow
                # halo (the only burst is the separate RingExplosion).
                px = self.geo.note_d * 0.46
                self._jcache[key] = self.font.render(
                    _JUDGE_TEXT[result], px, color=_JUDGE_COL[result],
                    spacing=C.JUDGE_SPACING * self.geo.scale)
        return self._jcache[key], is_skin

    @staticmethod
    def _legacy_explosion_anim(age):
        """osu!lazer LegacyHitExplosion.Animate envelope (scale, alpha) at `age`
        ms: FadeInFromZero(120) → FadeOut(180); ScaleTo 0.6 →(96ms)1.1 →(48ms)0.9
        →(24ms)1.0, linear segments. The skin's taiko-hit300/100/0 sprite is the
        legacy HIT EXPLOSION — it bursts AT the hit target, it does not float up
        like Argon's judgement text."""
        if age < 120.0:
            alpha = age / 120.0
        elif age < 300.0:
            alpha = 1.0 - (age - 120.0) / 180.0
        else:
            alpha = 0.0
        if age < 96.0:
            scale = 0.6 + 0.5 * (age / 96.0)                 # 0.6 → 1.1
        elif age < 144.0:
            scale = 1.1 - 0.2 * ((age - 96.0) / 48.0)        # 1.1 → 0.9
        elif age < 168.0:
            scale = 0.9 + 0.1 * ((age - 144.0) / 24.0)       # 0.9 → 1.0
        else:
            scale = 1.0
        return scale, max(0.0, alpha)

    _RING_SPEC = {"great": (4, 4, 1.0), "ok": (4, 0, 0.6)}   # (small,large,travel_x); miss none

    def _ring_burst(self, rgb, res, age, rt, g):
        """lazer taiko ArgonJudgementPiece.RingExplosion: white hollow rings
        burst outward from the hit target, tinted by the result colour,
        additive. travel 58, start_position_ratio 0.6, fade 1000ms OutQuint.
        Seeded on the judged time so it is stable across frames."""
        spec = self._RING_SPEC.get(res)
        if spec is None:
            return
        n_small, n_large, tmult = spec
        ga = max(0.0, (1.0 - age / 1000.0)) ** 5
        if ga <= 0.004:
            return
        import math, random
        col = tuple(c / 255.0 for c in _JUDGE_COL[res][:3])
        sc = g.scale
        travel = 58.0 * sc * tmult
        p = min(age, 600) / 600.0
        rad = 0.6 + 0.4 * (1.0 - (1.0 - p) ** 5)
        pieces = [9.0 * sc] * n_small + [14.0 * sc] * n_large
        for i, size in enumerate(pieces):
            # perf: the seeded draws depend only on (rt, i, travel) — cache
            # them across frames instead of constructing a fresh
            # random.Random per piece per frame (values identical).
            rkey = (int(rt), i, travel)
            hit = self._rng_cache.get(rkey)
            if hit is None:
                rng = random.Random((int(rt) * 1000003) ^ (i * 2654435761))
                hit = (rng.uniform(0.0, 360.0),
                       rng.uniform(travel / 2.0, travel))
                if len(self._rng_cache) > 4096:
                    self._rng_cache.clear()
                self._rng_cache[rkey] = hit
            d, dist = hit
            cur = dist * rad
            _add_tex(rgb, self._ring, g.target_x + math.cos(d) * cur,
                     g.center_y + math.sin(d) * cur, size, size, ga, tint=col)

    def cpu_work(self, exps, judges, drums=()) -> bool:
        """Could composite() draw anything for these lists? Errs towards yes: a
        yes costs one slow frame, a wrong no would lose an effect."""
        if not _GPU_FX:
            return bool(drums) or bool(exps) or bool(judges)
        if _GPU_SJ:
            return False                 # everything composite() draws is in the pass
        return any(self._judge_tex(j[0], j[3])[1] for j in judges)

    def composite(self, rgb, exps, judges, drums=()):
        # rgb arrives as a writable (possibly flipped-view) frame from the PBO
        # pop and is mutated region-by-region in place — the old full-frame
        # ascontiguousarray copy ran on the render thread every frame and is
        # exactly what the writer thread's tobytes() already pays for.
        g = self.geo
        # R3D_TAIKO_GPU_FX: the drum flashes and hit explosions are drawn as
        # additive GL sprites in the main pass instead (see gpu_fx_sprites).
        # They are prebaked textures blended additively at fixed positions —
        # 30.6 of the 60.5 numpy calls per frame, and measured at 55 ns/px vs
        # the flashlight's 0.91 ns/px, i.e. the cost is per-call overhead on
        # ~278-pixel arrays, not pixel work.
        if _GPU_FX:
            drums = ()
            exps = ()
            # Argon judgement bursts (ring pieces + popup text) move too — see
            # gpu_judge_sprites. The SKIN judgement path is not moved, so keep
            # any skin-sprite judgements here.
            judges = (() if _GPU_SJ else
                      [j for j in judges if self._judge_tex(j[0], j[3])[1]])
        # input-drum press flashes (additive — clean bright pop + glow)
        for is_rim, left, a in drums:
            _add_prescaled(rgb, self._drum_scaled[(is_rim, left)],
                           g.drum_x, g.center_y, a)
        # hit explosions at the target (additive). A LEGACY skin (ships its own
        # taiko-hit300/100/0 sprites) uses those AS the hit explosion
        # (LegacyHitExplosion, drawn in the judges loop below); lazer does NOT
        # also draw the Argon accent glow / hit-target flash on top. Forcing them
        # was the "light on every click" a skin that blanked its explosion (blank
        # taiko-hit300 / taiko-glow) did not want. Skip the Argon explosion
        # entirely when the skin supplies legacy judgement sprites.
        for is_rim, age, big, res in (() if self._use_skin_judge else exps):
            if res == "great":
                if age < C.EXPLOSION_GREAT_IN_MS:
                    a = age / C.EXPLOSION_GREAT_IN_MS
                else:
                    f = 1.0 - (age - C.EXPLOSION_GREAT_IN_MS) / C.EXPLOSION_GREAT_OUT_MS
                    a = max(0.0, f) ** 4
            else:
                f = 1.0 - (age - C.EXPLOSION_GREAT_IN_MS) / C.EXPLOSION_OK_OUT_MS
                a = C.EXPLOSION_OK_PEAK * max(0.0, f)
            if a <= 0.001:
                continue
            _add_prescaled(rgb, self._exp_scaled[(is_rim, big)],
                           g.target_x, g.center_y, a)
            # legacy lane: add the solid note-colour flash filling the hit target
            # (the Argon explosion alone is invisible over an opaque skin target).
            if self._legacy_lane:
                _add_prescaled(rgb, self._hit_flash[is_rim],
                               g.target_x, g.center_y, min(1.0, a * 0.85))
        # judgement popups. Two distinct lazer behaviours:
        #   * SKIN (legacy) → LegacyHitExplosion: the taiko-hit300/100/0 sprite
        #     BURSTS at the hit target (centred on the drum), scale-punch + fade.
        #     It does NOT float up (the old code reused the Argon text's upward
        #     drift, so the ring hovered near the mascot's feet instead of on the
        #     hit target — issue #117).
        #   * ARGON → ArgonJudgementPiece: GREAT/OK text rises off the target and
        #     fades, with the RingExplosion burst.
        for res, age, rt, big in judges:
            tex, is_skin = self._judge_tex(res, big)
            if tex is None:
                continue
            h0, w0 = tex.shape[0], tex.shape[1]
            if is_skin:
                # LegacyHitExplosion: the skin's taiko-hit sprite BURSTS at the
                # hit target (no upward float). A blank/transparent sprite blits
                # to nothing — the skin's intentionally-hidden judgement.
                scale, alpha = self._legacy_explosion_anim(age)
                if alpha <= 0.01:
                    continue
                _blit_straight(rgb, tex, g.target_x, g.center_y,
                               w0 * scale, h0 * scale, alpha)
                continue
            self._ring_burst(rgb, res, age, rt, g)   # Argon RingExplosion
            p = age / C.JUDGE_MOVE_MS
            ease = 1.0 - (1.0 - p) ** 5                # OutQuint (move/scale)
            scale = 1.0 + 0.4 * ease
            alpha = max(0.0, (1.0 - p) ** 5)           # FadeOutFromOne, OutQuint
            if alpha <= 0.01:
                continue
            yoff = (0.6 + 0.4 * ease) * g.pf_h
            # straight alpha (lazer draws the judgement as a normal SpriteText —
            # not additive).
            _blit_straight(rgb, tex, g.target_x, g.center_y - yoff,
                           w0 * scale, h0 * scale, alpha)
        return rgb
