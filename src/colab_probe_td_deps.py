import importlib.util

names = [
    "bidict", "blosc2", "dask", "geff", "ilpy", "imagecodecs", "numba",
    "numcodecs", "psygnal", "pyarrow", "rustworkx", "skimage", "sqlalchemy",
    "tqdm", "donfig", "google_crc32c", "ct3", "witty", "rstar",
]
for name in names:
    spec = importlib.util.find_spec(name)
    print(name, bool(spec), None if spec is None else spec.origin)
