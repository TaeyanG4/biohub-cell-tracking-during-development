from pathlib import Path
import os
import subprocess
import sys

ROOT = Path("/content/biohub-r3")
BOOT = ROOT / "bootstrap"
VENV = ROOT / "venv"
ROOT.mkdir(parents=True, exist_ok=True)
BOOT.mkdir(parents=True, exist_ok=True)

if not (VENV / "bin" / "python").exists():
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q", "--target", str(BOOT), "virtualenv"
    ])
    env = os.environ.copy()
    env["PYTHONPATH"] = str(BOOT) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.check_call([
        sys.executable, "-m", "virtualenv", "--system-site-packages", str(VENV)
    ], env=env)

py = VENV / "bin" / "python"
subprocess.check_call([
    str(py), "-m", "pip", "install", "-q",
    "hoct==0.2.0",
    "spatial-graph==0.1.1",
    "zarr==3.2.1",
])

probe = """
import torch, hoct, tracksdata, zarr, polars
print('torch', torch.__version__)
print('cuda', torch.cuda.is_available())
print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print('hoct', hoct.__version__)
print('tracksdata', getattr(tracksdata, '__version__', 'unknown'))
print('zarr', zarr.__version__)
print('polars', polars.__version__)
"""
subprocess.check_call([str(py), "-c", probe])
print("venv_ready", py)
