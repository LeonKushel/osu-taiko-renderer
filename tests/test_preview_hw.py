"""The inline preview on the Mac's media engine (render/preview_hw.py, ported
from std PR #11).

What must hold:
  * the master's ffmpeg arguments do not depend on which encoder makes the
    preview, so the master cannot change;
  * asking is not getting: R3D_PREVIEW_VT and the platform default decide what
    is wanted, the probe decides what is used;
  * an ffmpeg failure that names videotoolbox switches the media engine off for
    the next render, and nothing else does;
  * where a hardware session really opens, the real two-output command makes a
    master and a preview of the right length, with and without audio.

Runnable two ways:  pytest tests/test_preview_hw.py   OR   python tests/test_preview_hw.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from osu_taiko_renderer.beatmap.models import RenderConfig  # noqa: E402
from osu_taiko_renderer.render import preview_hw as phw  # noqa: E402
from osu_taiko_renderer.render import render as R  # noqa: E402


class _Captured(Exception):
    pass


def _cmd(hw: bool, *, audio=None, total=60.0, stream=False, compact=None,
         res=(1280, 720), out="m.mp4", prev="m.embed.mp4", live=False,
         encoder="libx264"):
    """The ffmpeg command taiko would start, without starting it."""
    saved = (R.subprocess.Popen, phw._vt_decision, os.environ.get("R3D_PREVIEW_LIVE"),
             os.environ.get("R3D_TAIKO_LOUDNORM_CACHE"))
    got = []

    def fake_popen(cmd, **kw):
        got.append(list(cmd))
        raise _Captured()
    R.subprocess.Popen = fake_popen
    phw._vt_decision = hw
    os.environ["R3D_TAIKO_LOUDNORM_CACHE"] = "0"       # never build a cache in a test
    if live:
        os.environ["R3D_PREVIEW_LIVE"] = "1"
    else:
        os.environ.pop("R3D_PREVIEW_LIVE", None)
    try:
        cfg = RenderConfig(resolution=res, fps=60, encoder=encoder)
        try:
            R._spawn_ffmpeg(cfg, Path(out), Path(audio) if audio else None, 0, 1.0, total,
                            preview_path=Path(prev), stream_master=stream,
                            compact_path=Path(compact) if compact else None)
        except _Captured:
            pass
        assert len(got) == 1, "ffmpeg was not started exactly once"
        return got[0]
    finally:
        R.subprocess.Popen, phw._vt_decision = saved[0], saved[1]
        for k, v in (("R3D_PREVIEW_LIVE", saved[2]), ("R3D_TAIKO_LOUDNORM_CACHE", saved[3])):
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_master_arguments_do_not_depend_on_the_preview_encoder():
    for extra in ({}, {"stream": True}, {"compact": "m-embed-sm.mp4"},
                  {"audio": "a.wav"}, {"audio": "a.wav", "stream": True},
                  {"audio": "a.wav", "compact": "m-embed-sm.mp4"}):
        cpu, hw = _cmd(False, **extra), _cmd(True, **extra)
        i, j = cpu.index("m.mp4"), hw.index("m.mp4")
        # the one difference allowed before the master's output path: the
        # lead-in filter at the end of the PREVIEW branch of the graph
        g = hw.index("-filter_complex") + 1
        lead = "," + phw.vt_lead_in_filter(30)
        assert hw[g].count(lead + "[vp]") == 1 and hw[g].count("tpad") == 1
        graph = hw[g].replace(lead, "")
        if "audio" in extra:
            # ... and, with audio, the preview's audio ended at the video's known length
            end = f";[ap_full]atrim=end={60.0:.6f}[ap]"
            assert graph.endswith(end) and graph.count("[ap_full]") == 2, graph
            graph = graph[:-len(end)].replace("[ap_full]", "[ap]")
        same = hw[:g] + [graph] + hw[g + 1:j + 1]
        assert cpu[:i + 1] == same, f"master part differs with {extra}"
        assert "h264_videotoolbox" not in cpu
        assert hw.count("h264_videotoolbox") == 1


def test_shortest_is_not_used_on_a_preview_with_a_lead_in():
    # it compares the streams before the lead-in is cut off: with the audio ended
    # at the video's length it would cut the last half second of video
    cpu, hw = _cmd(False, audio="a.wav"), _cmd(True, audio="a.wav")
    tail = lambda c: c[c.index("m.mp4") + 1:]
    assert "-shortest" in tail(cpu) and "-shortest" not in tail(hw)
    assert "-shortest" in hw[:hw.index("m.mp4")]            # the master keeps it


def test_no_known_length_with_audio_keeps_the_cpu_preview():
    # the lead-in needs the video's length to end the preview's audio at
    cmd = _cmd(True, audio="a.wav", total=None)
    assert "h264_videotoolbox" not in cmd and "tpad" not in cmd[cmd.index("-filter_complex") + 1]
    assert "h264_videotoolbox" in _cmd(True, total=None)    # video only: nothing to end


def test_a_master_on_a_hardware_encoder_keeps_the_preview_on_x264():
    # then the preview's would be a SECOND hardware session, which can be refused
    for enc in ("h264_nvenc", "h264_videotoolbox"):
        cmd = _cmd(True, encoder=enc)
        tail = cmd[cmd.index("m.mp4") + 1:]
        assert "libx264" in tail and "h264_videotoolbox" not in tail, (enc, tail[:8])
        assert "tpad" not in cmd[cmd.index("-filter_complex") + 1]


def test_the_cpu_preview_is_what_it_was():
    cmd = _cmd(False, audio="a.wav")
    assert "h264_videotoolbox" not in cmd and cmd.count("libx264") == 2
    tail = cmd[cmd.index("m.mp4") + 1:]
    vbps = R._preview_video_bps(60.0)
    assert tail[tail.index("-maxrate") + 1] == str(int(vbps * 1.25))
    assert tail[tail.index("-preset") + 1] == "veryfast"


def test_hw_preview_arguments():
    cmd = _cmd(True, compact="m-embed-sm.mp4")
    i = cmd.index("h264_videotoolbox")
    tail = cmd[i:cmd.index("m.embed.mp4")]
    vbps = R._preview_video_bps(60.0)
    assert tail[tail.index("-b:v") + 1] == str(int(vbps * phw._VT_RATE))
    assert tail[tail.index("-allow_sw") + 1] == "0"     # never Apple's software encoder
    assert tail[tail.index("-constant_bit_rate") + 1] == "1" and "-maxrate" not in tail
    n = phw._vt_lead_frames(30)
    assert n == 15
    assert tail[tail.index("-force_key_frames") + 1] == f"expr:eq(n,{n})"
    bsf = tail[tail.index("-bsf:v") + 1]
    assert bsf.startswith("noise=drop=lt(pts*tb") and f"setts=pts=PTS-{n}/30/TB:dts=DTS-{n}/30/TB" in bsf
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert f"scale=-2:720,tpad=start={n}:start_mode=clone[vp]" in graph
    assert tail[tail.index("-g") + 1] == "30"           # a keyframe every second, as before
    assert "-threads" not in tail
    # the Discord copy is a deliverable: it stays on x264
    k = cmd.index("m.embed.mp4")
    assert "libx264" in cmd[k:] and "h264_videotoolbox" not in cmd[k:]


def test_the_live_preview_sink_is_the_same_for_both_encoders():
    d = tempfile.mkdtemp()
    try:
        p = os.path.join(d, "m.embed.mp4")
        cpu, hw = _cmd(False, prev=p, live=True), _cmd(True, prev=p, live=True)
        assert cpu[cpu.index("-f", cpu.index("libx264", cpu.index("m.mp4"))):] \
            == hw[hw.index("-f", hw.index("h264_videotoolbox")):]
        assert hw[-1].endswith("live.m3u8")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _decide(env, mac_default, probe_says):
    saved = (os.environ.get("R3D_PREVIEW_VT"), phw._MAC_DEFAULT, phw._vt_probe,
             phw._vt_decision)
    asked = []

    def fake_probe(ignore_off=False):
        asked.append(ignore_off)
        return probe_says
    try:
        if env is None:
            os.environ.pop("R3D_PREVIEW_VT", None)
        else:
            os.environ["R3D_PREVIEW_VT"] = env
        phw._MAC_DEFAULT = mac_default
        phw._vt_probe = fake_probe
        phw._vt_decision = None
        return phw.preview_on_media_engine(), asked
    finally:
        if saved[0] is None:
            os.environ.pop("R3D_PREVIEW_VT", None)
        else:
            os.environ["R3D_PREVIEW_VT"] = saved[0]
        phw._MAC_DEFAULT, phw._vt_probe, phw._vt_decision = saved[1:]


def test_decision_rule():
    assert _decide(None, True, True) == (True, [False])     # a Mac, session opens
    assert _decide(None, True, False)[0] is False           # a Mac, no session
    assert _decide(None, False, True) == (False, [])        # not a Mac: never asked
    assert _decide("0", True, True) == (False, [])          # switched off by hand
    assert _decide("1", False, True) == (True, [True])      # asked for anywhere; overrides the day off
    assert _decide("1", False, False)[0] is False           # asking is not getting


def test_only_a_failure_naming_videotoolbox_switches_it_off():
    d = tempfile.mkdtemp()
    saved = phw._r3d_cache_dir
    phw._r3d_cache_dir = lambda: d
    try:
        hw, cpu = _cmd(True), _cmd(False)
        mark = os.path.join(d, "vt-preview-off")
        assert not phw.note_preview_failure(cpu, b"[h264_videotoolbox @ 0x1] Error")
        assert not phw.note_preview_failure(hw, b"No space left on device")
        assert not os.path.exists(mark)
        assert phw.note_preview_failure(
            hw, b"[h264_videotoolbox @ 0x1] Error: cannot create compression session: -12908")
        assert os.path.exists(mark)
        assert phw._vt_probe() is False                     # off for the day, without probing
        with open(mark, "w") as fh:
            fh.write("1")                                   # the day is over
        assert phw._vt_probe() == phw._vt_probe(ignore_off=True)
    finally:
        phw._r3d_cache_dir = saved
        shutil.rmtree(d, ignore_errors=True)


def _real(n, with_audio):
    ff, fp = shutil.which("ffmpeg"), shutil.which("ffprobe")
    d = tempfile.mkdtemp()
    saved = phw._r3d_cache_dir
    phw._r3d_cache_dir = lambda: d                          # the real probe, a private cache
    try:
        if not ff or not fp or not phw._vt_probe():
            print("SKIP (no ffmpeg with a hardware H.264 session here)")
            return None
        w, h = 320, 180
        wav = None
        if with_audio:
            wav = os.path.join(d, "a.wav")                  # longer than the video, as a real mix is
            subprocess.run([ff, "-v", "error", "-f", "lavfi", "-i",
                            "sine=frequency=440:sample_rate=48000:duration=7", "-ac", "2", wav], check=True)
        out, prev = os.path.join(d, "m.mp4"), os.path.join(d, "m.embed.mp4")
        cmd = _cmd(True, audio=wav, total=n / 60.0, res=(w, h), out=out, prev=prev)
        base = np.random.default_rng(3).integers(0, 256, (h, w, 3), dtype=np.uint8)
        frames = b"".join(np.roll(base, 3 * i, axis=1).tobytes() for i in range(n))
        p = subprocess.run(cmd, input=frames, capture_output=True, cwd=d, timeout=180)
        assert p.returncode == 0, p.stderr.decode(errors="replace")[-400:]
        streams = lambda f: json.loads(subprocess.run(
            [fp, "-v", "error", "-count_packets", "-show_entries",
             "stream=codec_type,codec_name,nb_read_packets,height,start_time,duration", "-of", "json", f],
            capture_output=True, text=True).stdout)["streams"]
        first = sorted((float(q["pts_time"]), q["flags"]) for q in json.loads(subprocess.run(
            [fp, "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time,flags",
             "-of", "json", prev], capture_output=True, text=True).stdout)["packets"])[0]
        dec = subprocess.run([ff, "-v", "error", "-i", prev, "-map", "0:v", "-fps_mode", "passthrough",
                              "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"], capture_output=True)
        return streams(out), streams(prev), first, dec
    finally:
        phw._r3d_cache_dir = saved
        shutil.rmtree(d, ignore_errors=True)


def test_real_two_output_encode_where_a_session_opens():
    n = 120
    r = _real(n, with_audio=False)
    if r is None:
        return
    master, prev, first, dec = r
    mv = next(s for s in master if s["codec_type"] == "video")
    pv = next(s for s in prev if s["codec_type"] == "video")
    assert (mv["codec_name"], int(mv["nb_read_packets"]), int(mv["height"])) == ("h264", n, 180)
    pk = int(pv["nb_read_packets"])
    assert pv["codec_name"] == "h264" and int(pv["height"]) == 720 and abs(pk - n // 2) <= 1, pv
    # the lead-in is gone: the file starts on a keyframe at time 0 and decodes
    # to exactly as many frames as it has packets, with nothing to conceal
    assert abs(first[0]) < 1e-6 and "K" in first[1], first
    assert dec.returncode == 0 and not dec.stderr.strip(), dec.stderr[-200:]
    assert len(dec.stdout) == pk * 1280 * 720, (len(dec.stdout) / (1280 * 720), pk)


def test_real_encode_with_audio_keeps_every_frame_and_ends_the_audio_with_the_video():
    """The lead-in lengthens the video's timeline until it is cut off, and the
    audio is not lengthened with it. Both ways of getting that wrong are silent:
    half a second of extra audio, or the last half second of video cut."""
    n = 240                                                 # 4 s at 60 fps
    r = _real(n, with_audio=True)
    if r is None:
        return
    _, prev, first, _ = r
    v = next(s for s in prev if s["codec_type"] == "video")
    a = next(s for s in prev if s["codec_type"] == "audio")
    assert int(v["nb_read_packets"]) == n // 2, v           # every frame, none cut at the end
    assert abs(float(v["start_time"])) < 0.05 and abs(float(a["start_time"])) < 0.05, (v, a)
    assert abs(float(a["duration"]) - n / 60.0) < 0.06, a   # the audio ends with the video


if __name__ == "__main__":
    for _n, _f in sorted(globals().items()):
        if _n.startswith("test_") and callable(_f):
            _f()
            print("ok  ", _n)
    print("all taiko hardware-preview tests PASSED")
