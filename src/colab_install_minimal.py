from pathlib import Path
import os
import subprocess
import sys

SITE = Path("/content/biohub-r3/sitepkgs")
SITE.mkdir(parents=True, exist_ok=True)

packages = [
    "hoct==0.2.0",
    "tracksdata==0.1.0rc10",
    "spatial-graph==0.1.1",
    "zarr==3.2.1",
    "bidict==0.24.1",
    "geff==1.3.1.1.3",
    "geff-spec==1.3.0",
    "ilpy==0.6.0",
    "imagecodecs==2026.8.16",
    "numcodecs==0.15.1",
    "rustworkx==0.18.1",
    "donfig==0.8.1.post1",
    "ct3==3.4.0.post5",
    "witty==0.3.2",
    "rstar-python==0.2.0",
]

cmd = [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--target", str(SITE), *packages]
print("installing", len(packages), "packages into", SITE)
subprocess.check_call(cmd)

sys.path.insert(0, str(SITE))
os.environ["PYTHONPATH"] = str(SITE) + os.pathsep + os.environ.get("PYTHONPATH", "")

mods = ["zarr", "geff", "spatial_graph", "tracksdata", "hoct"]
for name in mods:
    try:
        mod = __import__(name)
        print("import_ok", name, getattr(mod, "__version__", "unknown"))
    except Exception as exc:
        print("import_fail", name, type(exc).__name__, repr(exc))

try:
    import torch
    from hoct.inference import extract_edge_features
    from hoct.features import create_graph
    print("hoct_feature_api_ok", bool(extract_edge_features), bool(create_graph))
    print("torch", torch.__version__, "gpu", torch.cuda.get_device_name(0))
except Exception as exc:
    print("hoct_feature_api_fail", type(exc).__name__, repr(exc))
