"""R3D_TAIKO_GPU_YUV must hand the encoder the bytes ffmpeg itself would have
produced from the same RGB frame. Two links, both tested here:

  ffmpeg  ==  rgb_to_yuv420p (numpy)      where this machine's ffmpeg uses that
                                          arithmetic (ffmpeg_matches_twin)
  rgb_to_yuv420p  ==  the shader pair     skipped where there is no GL context

swscale is not the same on every build: ffmpeg 8.1 on arm64 matches the twin on
every sample, ffmpeg 6.1.1 on x86-64 gives chroma one level apart on ~9% of
noise samples. The engine therefore PROBES the local ffmpeg and leaves the
conversion to ffmpeg where they differ.

Noise is the test image on purpose: a smooth picture hides the chroma filter
(a 2x2 box average agrees with the real eight-row filter on a gradient and on
only 7% of noise samples). The first version of this conversion was that box
average, "validated" against a routine this engine's pipe never reaches.

The switches that depend on each other are resolved in one module; the last
tests hold that a half-set combination cannot lose a layer.

Runnable two ways:  pytest tests/test_gpu_yuv.py   OR   python tests/test_gpu_yuv.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from osu_taiko_renderer.render import gl  # noqa: E402


def _ffmpeg(rgb: np.ndarray) -> "np.ndarray | None":
    ff = shutil.which("ffmpeg")
    if ff is None:
        return None
    h, w = rgb.shape[:2]
    p = subprocess.run(
        [ff, "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
         f"{w}x{h}", "-i", "pipe:0", "-pix_fmt", "yuv420p", "-f", "rawvideo",
         "pipe:1"], input=rgb.tobytes(), capture_output=True, timeout=60)
    if p.returncode or len(p.stdout) != w * h * 3 // 2:
        return None
    return np.frombuffer(p.stdout, np.uint8)


def _images():
    rng = np.random.default_rng(20261006)
    for w, h in ((1280, 720), (854, 480), (640, 360), (64, 16), (2, 12)):
        yield f"noise {w}x{h}", rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    # one-pixel black/white rows: the hardest case for the vertical filter,
    # and it drives the sums past both clip limits
    stripes = np.zeros((96, 160, 3), np.uint8)
    stripes[1::2] = 255
    yield "stripes", stripes
    sat = np.zeros((48, 64, 3), np.uint8)
    sat[:24, :, 0] = 255
    sat[24:, :, 2] = 255
    yield "saturated red over blue", sat


def test_probe_agrees_with_a_whole_frame():
    # the engine decides from a 128x96 noise frame; a 1280x720 one must agree,
    # whichever way this machine's ffmpeg goes
    rgb = np.random.default_rng(99).integers(0, 256, (720, 1280, 3), dtype=np.uint8)
    ref = _ffmpeg(rgb)
    if ref is None:
        print("SKIP (no usable ffmpeg)")
        return
    whole = bool((gl.rgb_to_yuv420p(rgb) == ref).all())
    assert gl.ffmpeg_matches_twin() == whole


def test_off_unless_wanted_and_off_where_ffmpeg_differs():
    import osu_taiko_renderer.render.envflag as sw
    if not sw.GPU_YUV:                       # not asked for and not this platform's default
        assert gl._GPU_YUV is False
    if not gl.ffmpeg_matches_twin():         # wanted or not: this ffmpeg converts differently
        assert gl._GPU_YUV is False
    if sw.GPU_YUV and gl.ffmpeg_matches_twin():
        assert gl._GPU_YUV is True


def test_twin_equals_ffmpeg_on_every_sample():
    if not gl.ffmpeg_matches_twin():
        print("SKIP (this ffmpeg build converts differently; the GPU conversion "
              "stays off here)")
        return
    ran = 0
    for name, rgb in _images():
        ref = _ffmpeg(rgb)
        if ref is None:
            print("SKIP (no usable ffmpeg)")
            return
        got = gl.rgb_to_yuv420p(rgb)
        bad = int((got != ref).sum())
        assert bad == 0, f"{name}: {bad} of {ref.size} samples differ from ffmpeg"
        ran += 1
    assert ran == 7


def test_chroma_weights_are_the_full_precision_ones():
    # the 8-bit rounding of these (-4 -11 31 112 ...) is the near miss
    assert sum(gl._CHROMA_TAPS) == 8192
    assert gl._CHROMA_TAPS == tuple(reversed(gl._CHROMA_TAPS))


def _planes(spr):
    y = np.frombuffer(spr._fbo_y.read(components=1, alignment=1), np.uint8)
    u = np.frombuffer(spr._fbo_uv.read(components=1, alignment=1, attachment=0),
                      np.uint8)
    v = np.frombuffer(spr._fbo_uv.read(components=1, alignment=1, attachment=1),
                      np.uint8)
    return np.concatenate([y, u, v])


def test_shader_equals_twin_from_the_scene_in_its_row_order():
    """The planes come out top row first although the scene is stored bottom
    row first. A picture with a different top and bottom is what catches a
    flipped read: noise alone would only say "wrong", not "upside down"."""
    try:
        spr = gl.SpriteRenderer(320, 180)
    except Exception:  # noqa: BLE001 -- no GL device on this box
        print("SKIP (no GL context)")
        return
    rng = np.random.default_rng(7)
    for trial in range(3):
        picture = rng.integers(0, 256, (180, 320, 4), dtype=np.uint8)   # top-down
        if trial == 1:
            picture[:90] //= 4              # dark top half, bright bottom half
        if trial == 2:                      # stripes through the clip limits
            picture[:] = 0
            picture[1::2] = 255
        # row 0 of the scene texture is the picture's BOTTOM row
        stored = picture[::-1]
        spr._scene_tex.write(np.ascontiguousarray(stored).tobytes())
        spr._convert_yuv()
        want = gl.rgb_to_yuv420p(np.ascontiguousarray(picture[..., :3]))
        bad = int((_planes(spr) != want).sum())
        assert bad == 0, f"trial {trial}: {bad} of {want.size} samples differ"
    spr.release()


def test_shader_equals_twin_from_a_cpu_frame():
    # the slow road (a frame the CPU finished) and the results screen
    try:
        spr = gl.SpriteRenderer(320, 180)
    except Exception:  # noqa: BLE001
        print("SKIP (no GL context)")
        return
    rng = np.random.default_rng(11)
    frames = [rng.integers(0, 256, (180, 320, 3), dtype=np.uint8) for _ in range(5)]
    frames[1][:90] //= 4
    got = []
    for f in frames:
        r = spr.yuv_from_rgb(f)
        if r is not None:
            got.append(np.array(r))
    got += [np.array(r) for r in spr.read_yuv_drain()]
    assert len(got) == len(frames)                       # none lost, none doubled
    for i, (g, f) in enumerate(zip(got, frames)):        # and in order
        bad = int((g != gl.rgb_to_yuv420p(f)).sum())
        assert bad == 0, f"frame {i}: {bad} samples differ"
    spr.release()


def test_sync_rgb_read_is_top_down():
    try:
        spr = gl.SpriteRenderer(64, 32)
    except Exception:  # noqa: BLE001
        print("SKIP (no GL context)")
        return
    picture = np.random.default_rng(3).integers(0, 256, (32, 64, 4), dtype=np.uint8)
    stored = picture[::-1]
    spr._scene_tex.write(np.ascontiguousarray(stored).tobytes())
    assert (spr.read_rgb()[..., :3] == picture[..., :3]).all()
    spr.release()


def _resolved(env: dict) -> dict:
    code = ("import json, osu_taiko_renderer.render.envflag as s;"
            "print(json.dumps({k: getattr(s, k) for k in "
            "('GPU_FX','GPU_SJ','GPU_FL','FL_EXACT','GPU_NUM','GPU_HUD','GPU_BREAK',"
            "'MERGE_RUNS','ROUND','MAP_READBACK','SOCKET_PIPE','RESULTS_AHEAD',"
            "'GPU_YUV','INSTANCED')}))")
    e = {k: v for k, v in os.environ.items() if not k.startswith("R3D_")}
    e.update(env)
    import json
    out = subprocess.run([sys.executable, "-c", code], env=e, capture_output=True,
                         text=True, cwd=str(Path(__file__).resolve().parents[1]),
                         check=True).stdout
    return json.loads(out)


def test_a_switch_without_what_it_needs_is_off_everywhere():
    # GPU_FL=1 alone used to turn the CPU spotlight off (flashlight.py) while
    # render.py never drew the GPU one: no flashlight at all. What each switch
    # needs is switched OFF by hand here, so the test means the same on a
    # platform where the set is on by default.
    deps = ("GPU_FX", "GPU_SJ", "GPU_FL", "FL_EXACT", "GPU_NUM", "GPU_HUD", "GPU_BREAK", "MERGE_RUNS")
    r = _resolved({"R3D_TAIKO_GPU_FX": "0", "R3D_TAIKO_GPU_FL": "1", "R3D_TAIKO_GPU_NUM": "1",
                   "R3D_TAIKO_GPU_HUD": "1", "R3D_TAIKO_GPU_BREAK": "1",
                   "R3D_TAIKO_MERGE_RUNS": "1"})
    assert not any(r[k] for k in deps), r
    r = _resolved({"R3D_TAIKO_GPU_FX": "1", "R3D_TAIKO_GPU_FL": "0", "R3D_TAIKO_GPU_HUD": "0",
                   "R3D_TAIKO_GPU_BREAK": "1"})
    assert r["GPU_FX"] and r["GPU_SJ"], r         # effects bring the skin judgements
    assert not r["FL_EXACT"] and not r["GPU_BREAK"], r
    r = _resolved({"R3D_TAIKO_GPU_FX": "1", "R3D_TAIKO_GPU_FL": "1"})
    assert r["GPU_FL"] and r["FL_EXACT"], r       # the GPU flashlight is the exact one


_ALL_ON = {k: "1" for k in (
    "R3D_TAIKO_GPU_FX", "R3D_TAIKO_GPU_FL", "R3D_TAIKO_GPU_NUM", "R3D_TAIKO_GPU_HUD", "R3D_TAIKO_GPU_BREAK",
    "R3D_TAIKO_MERGE_RUNS", "R3D_TAIKO_ROUND", "R3D_MAP_READBACK",
    "R3D_MAC_SOCKET_PIPE", "R3D_TAIKO_RESULTS_AHEAD", "R3D_TAIKO_GPU_YUV",
    "R3D_TAIKO_INSTANCED")}


def test_nothing_is_on_unless_asked_and_stock_wins():
    none = _resolved({})
    if sys.platform != "darwin":
        assert not any(none.values()), none            # nothing is on by default off a Mac
    else:
        # the Mac default: the whole set except the two opt-in switches
        opt_in = {"INSTANCED", "GPU_YUV"}
        assert all(v for k, v in none.items() if k not in opt_in), none
        assert not any(none[k] for k in opt_in), none
    on = _resolved(_ALL_ON)
    mac_only = {"MAP_READBACK", "SOCKET_PIPE"}
    for k, v in on.items():
        assert v or (k in mac_only and sys.platform != "darwin"), (k, on)
    # R3D_TAIKO_STOCK=1 is the stock render path whatever else is set
    assert not any(_resolved(dict(_ALL_ON, R3D_TAIKO_STOCK="1")).values())


def test_zero_means_off_and_every_module_reads_the_same_answer():
    r = _resolved({"R3D_TAIKO_GPU_FX": "0", "R3D_TAIKO_GPU_FL": "1"})
    assert not r["GPU_FX"] and not r["GPU_FL"], r
    code = ("import osu_taiko_renderer.render.envflag as s;"
            "from osu_taiko_renderer.render import flashlight as f;"
            "from osu_taiko_renderer.argon import compositor as c, hud as h;"
            "assert f._GPU_FL is s.GPU_FL and c._GPU_FX is s.GPU_FX;"
            "assert c._GPU_SJ is s.GPU_SJ and h._GPU_NUM is s.GPU_NUM")
    for env in ({}, {"R3D_TAIKO_GPU_FL": "1"},
                {"R3D_TAIKO_GPU_FX": "1", "R3D_TAIKO_GPU_FL": "1",
                 "R3D_TAIKO_GPU_NUM": "1"}):
        e = {k: v for k, v in os.environ.items() if not k.startswith("R3D_")}
        e.update(env)
        subprocess.run([sys.executable, "-c", code], env=e, check=True,
                       cwd=str(Path(__file__).resolve().parents[1]))


def _main_under(rc=None, raises=None, any_fast=False):
    """Run the package's __main__ with the CLI and the re-exec stubbed.
    Returns (exit code or "re-run", the environment a re-run was started with)."""
    import runpy
    import osu_taiko_renderer.render.envflag as sw
    from osu_taiko_renderer import cli
    saved = (cli.main, os.execve, sw.ANY_FAST)
    reran = []

    class _Reexec(Exception):
        pass

    def fake_main():
        if raises is not None:
            raise raises
        return rc

    def fake_execve(exe, argv, env):
        reran.append((argv, env))
        raise _Reexec()
    cli.main, os.execve, sw.ANY_FAST = fake_main, fake_execve, any_fast
    code = None
    try:
        try:
            runpy.run_module("osu_taiko_renderer", run_name="__main__")
        except SystemExit as e:
            code = e.code
        except _Reexec:
            code = "re-run"
    finally:
        cli.main, os.execve, sw.ANY_FAST = saved
    return code, (reran[0] if reran else None)


def test_a_render_that_fails_with_a_switch_on_is_run_again_on_the_stock_path():
    code, rerun = _main_under(raises=RuntimeError("GL error"), any_fast=True)
    assert code == "re-run" and rerun[1]["R3D_TAIKO_STOCK"] == "1"
    assert rerun[0][1:3] == ["-m", "osu_taiko_renderer"]
    assert _main_under(rc=1, any_fast=True)[0] == "re-run"
    assert _main_under(raises=SystemExit(3), any_fast=True)[0] == "re-run"
    # and the re-run cannot loop: with R3D_TAIKO_STOCK=1 nothing counts as a switch
    r = _resolved(dict(_ALL_ON, R3D_TAIKO_STOCK="1"))
    assert not any(r.values())


def test_nothing_else_is_run_again():
    assert _main_under(rc=0, any_fast=True) == (0, None)             # it worked
    assert _main_under(rc=2, any_fast=False) == (2, None)            # the stock path's own failure
    try:
        _main_under(raises=RuntimeError("x"), any_fast=False)        # ... and its exception is not swallowed
        raise AssertionError("expected the exception")
    except RuntimeError:
        pass
    try:
        _main_under(raises=KeyboardInterrupt(), any_fast=True)       # a cancelled job is not re-run
        raise AssertionError("expected KeyboardInterrupt")
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    for _n, _f in sorted(globals().items()):
        if _n.startswith("test_") and callable(_f):
            _f()
            print("ok  ", _n)
    print("all taiko GPU colour-conversion tests PASSED")
