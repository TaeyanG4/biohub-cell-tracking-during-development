from pathlib import Path
import subprocess
import sys

site = Path("/content/biohub-r3/sitepkgs")
site.mkdir(parents=True, exist_ok=True)
cmd = [
    sys.executable, "-m", "pip", "install", "--dry-run", "--target", str(site),
    "hoct==0.2.0", "spatial-graph==0.1.1", "zarr==3.2.1",
]
print("running", " ".join(cmd))
result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
print(result.stdout[-20000:])
print("returncode", result.returncode)
