from pathlib import Path
import sys


site = Path("/content/biohub-r3/sitepkgs")
sys.path.insert(0, str(site))

import polars as pl
import torch
import tracksdata
import hoct
from hoct.features import create_graph
from hoct.inference import extract_edge_features

s = pl.Series("x", [1.0, 2.0], dtype=pl.Float16)
print("polars", pl.__version__, "float16_dtype", str(s.dtype), "sum", float(s.sum()))
print("tracksdata", getattr(tracksdata, "__version__", "unknown"))
print("hoct", getattr(hoct, "__version__", "unknown"))
print("hoct_api", bool(create_graph), bool(extract_edge_features))
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
