from pathlib import Path
import os
import subprocess
import sys


site = Path("/content/biohub-r3/sitepkgs")
site.mkdir(parents=True, exist_ok=True)
packages = [
    "Deprecated==1.2.18",
    "wrapt==1.17.3",
    "polars==1.44.2",
    "polars-runtime-32==1.44.2",
]
subprocess.check_call([
    sys.executable,
    "-m",
    "pip",
    "install",
    "-q",
    "--no-deps",
    "--target",
    str(site),
    *packages,
])
sys.path.insert(0, str(site))
os.environ["PYTHONPATH"] = str(site) + os.pathsep + os.environ.get("PYTHONPATH", "")
for name in ["zarr", "geff", "tracksdata", "hoct"]:
    try:
        mod = __import__(name)
        print("import_ok", name, getattr(mod, "__version__", "unknown"))
    except Exception as exc:
        print("import_fail", name, type(exc).__name__, repr(exc))
