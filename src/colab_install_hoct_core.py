from __future__ import annotations

import subprocess
import sys


core = [
    "tracksdata==0.1.0rc10",
    "spatial-graph==0.1.1",
    "zarr==3.2.1",
    "gurobipy==12.0.3",
    "pydantic>=2.12.5",
    "pyyaml>=6.0.3",
    "rich>=14.3.1",
]

print("installing core packages", flush=True)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *core])
print("installing hoct without dependencies", flush=True)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "hoct==0.2.0"])

import torch
import tracksdata
import zarr
import hoct
import spatial_graph

print("torch", torch.__version__)
print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("hoct", getattr(hoct, "__version__", "unknown"))
print("tracksdata", getattr(tracksdata, "__version__", "unknown"))
print("zarr", zarr.__version__)
print("spatial_graph", getattr(spatial_graph, "__version__", "unknown"))
print("HOCT_CORE_READY")
