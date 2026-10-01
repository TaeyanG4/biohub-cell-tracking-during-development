import importlib.util

names = [
    "numpy", "pandas", "scipy", "torch", "zarr", "polars", "tracksdata",
    "hoct", "spatial_graph", "pooch", "h5py", "pydantic", "yaml", "rich",
]

for name in names:
    spec = importlib.util.find_spec(name)
    print(name, bool(spec), None if spec is None else spec.origin)
