import os
import sys

import osu_taiko_renderer.render.envflag as _sw
from osu_taiko_renderer.cli import main


def _run() -> int:
    """Run the render. A render that fails while any speed switch is on
    (render/envflag.py) is run again on the stock path in a fresh process, so
    a switch can cost time but cannot lose a job: the rule std's and catch's
    Metal paths have. With no switch on this is `main()` and nothing else. A
    cancelled job (SIGTERM, Ctrl-C) is not re-run."""
    try:
        rc = main()
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else (1 if e.code else 0)
    except Exception as e:  # noqa: BLE001 - never let a speed switch lose a render
        if not _sw.ANY_FAST:
            raise
        print(f"\n[taiko] render raised {e!r}", file=sys.stderr, flush=True)
        rc = 1
    if rc and _sw.ANY_FAST:
        print(f"[taiko] render failed with speed switches on (rc={rc}) -> "
              f"re-running on the stock path", file=sys.stderr, flush=True)
        env = dict(os.environ, R3D_TAIKO_STOCK="1")
        os.execve(sys.executable,
                  [sys.executable, "-m", "osu_taiko_renderer"] + sys.argv[1:], env)
    return rc or 0


raise SystemExit(_run())
