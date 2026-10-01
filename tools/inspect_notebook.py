import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('kaggle_notebooks/latest_review/reyhan_0947/biohub-cell-tracking-0-947-lb.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    lines = src.split('\n')
    cell_type = cell['cell_type']
    if cell_type == 'markdown':
        header = lines[0] if lines else ''
        print(f"Cell {i:2d} [MD]: {header[:80]}")
    else:
        found = False
        for l in lines:
            l_strip = l.strip()
            if l_strip and not l_strip.startswith('#'):
                print(f"Cell {i:2d} [Code]: {l_strip[:80]}")
                found = True
                break
        if not found:
            print(f"Cell {i:2d} [Code]: (empty or comments only)")
