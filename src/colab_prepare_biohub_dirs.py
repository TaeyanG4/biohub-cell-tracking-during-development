from pathlib import Path

root = Path("/content/biohub-r3")
for name in ["src", "models", "inputs", "outputs"]:
    (root / name).mkdir(parents=True, exist_ok=True)
print("prepared", root)
