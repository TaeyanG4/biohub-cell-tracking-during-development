from pathlib import Path
import subprocess

py = Path("/content/biohub-r3/venv/bin/python")
cmd = [str(py), "-m", "pip", "install", "hoct==0.2.0", "spatial-graph==0.1.1", "zarr==3.2.1"]
print("running", " ".join(cmd))
result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
print(result.stdout[-12000:])
print("returncode", result.returncode)
