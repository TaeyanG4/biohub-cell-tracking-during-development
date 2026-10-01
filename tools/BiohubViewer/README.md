# BiohubViewer

Small Windows GUI for the Kaggle **Biohub - Cell Tracking During Development** data.

## Use

Double-click `BiohubViewer.exe`, then choose a `.zarr` folder.

The viewer will try to locate a matching `.geff` ground-truth folder automatically. You can also choose one manually.

### Features

- Explorer-style drive/folder tree for quickly switching between datasets
- Optional one-click opening of `.zarr` folders
- Editable path bar and one-click jump to the project `data` folder
- T (time) and Z (depth) sliders
- Z maximum-intensity projection (MIP)
- GEFF node overlay
- GEFF edge / track overlay
- Node ID labels
- Adjustable Z tolerance and track trail length
- Drag and drop `.zarr` / `.geff` folders
- Automatic contrast by percentile

## Keyboard

- Left / Right: previous / next time
- Up / Down: previous / next Z slice
- M: toggle MIP
- N: toggle nodes
- E: toggle edges

## Source

`biohub_viewer_gui.py`

## Rebuild

Run `build.ps1` from PowerShell. It requires Python, PyQt6, zarr, numpy, and PyInstaller.
