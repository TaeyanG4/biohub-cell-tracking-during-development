from pathlib import Path
import os
import shutil
import subprocess
import sys

ROOT = Path("/content/biohub-r3")
VENV = ROOT / "venv"
KAGGLE_DIR = ROOT / "kaggle"
ROOT.mkdir(parents=True, exist_ok=True)
KAGGLE_DIR.mkdir(parents=True, exist_ok=True)

if not VENV.exists():
    subprocess.check_call([sys.executable, "-m", "venv", "--system-site-packages", str(VENV)])

py = VENV / "bin" / "python"
subprocess.check_call([str(py), "-m", "pip", "install", "-q", "--upgrade", "pip"])
subprocess.check_call([
    str(py), "-m", "pip", "install", "-q",
    "kaggle==2.2.4",
    "hoct==0.2.0",
    "spatial-graph==0.1.1",
    "pooch==1.9.0",
])

src = Path("/content/biohub_credentials.json")
dst = KAGGLE_DIR / "credentials.json"
if src.exists():
    shutil.copy2(src, dst)
    os.chmod(dst, 0o600)

print("biohub_root", ROOT)
print("venv_python", py)
print("credentials_present", dst.exists())
print("disk")
subprocess.run(["df", "-h", "/content"], check=False)
