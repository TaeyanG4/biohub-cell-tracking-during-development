"""Text transforms that add an 'output' mode to x138's V1284 coordinate-refinement patch.

In x138's 'candidate' mode the refined (sub-voxel) centres feed everything downstream:
the trilinear edge-feature lookup, positional features, edge distances, the ILP graph
and post-processing. The edge model was trained on detector voxels, and shifted
lookups lower its edge probabilities (fewer candidate edges above threshold).

'output' mode keeps association exactly as in 'zero' mode (detector voxels in, identical
edge probabilities and ILP input) and swaps the refined centres in only when
predict_video finalises its coordinates, i.e. before graph building, post-processing
and submission. Both functions are shared by the local runner and the C012 builder so
the code tested locally is the code that ships.
"""

from __future__ import annotations

MODULE_CACHE_OLD = "_CACHE = None\n"
MODULE_CACHE_NEW = "_CACHE = None\n_OUTPUT_ROWS = []\n"
MODULE_RETURN_OLD = "        raise RuntimeError('invalid V1284 displacement')\n    return result\n"
MODULE_RETURN_NEW = '''        raise RuntimeError('invalid V1284 displacement')
    if mode == 'output':
        _OUTPUT_ROWS.append(result)
        return arr.astype(np.float32)
    return result


def finalize(coords, ds_arr):
    """'output' mode: association ran on detector voxels; swap the refined centres in now."""
    global _OUTPUT_ROWS
    if os.environ.get('V1284_MODE') != 'output':
        return coords
    rows, _OUTPUT_ROWS = _OUTPUT_ROWS, []
    refined = np.concatenate(rows) if rows else np.empty((0, 4), np.float32)
    if refined.shape != coords.shape or not np.array_equal(refined[:, 0], coords[:, 0]):
        raise RuntimeError(('V1284 output rows misaligned', refined.shape, coords.shape))
    out = coords.astype(np.float32).copy()
    out[:, 1:] = refined[:, 1:] * np.asarray(ds_arr, dtype=np.float32)
    return out
'''

SCRIPT_IMPORT_OLD = "from v1284_coordinate_refinement import refine as _v1284_refine, index_features as _v1284_index\n"
SCRIPT_IMPORT_NEW = ("from v1284_coordinate_refinement import refine as _v1284_refine, index_features as _v1284_index, "
                     "finalize as _v1284_finalize\n")
SCRIPT_FINAL_OLD = "    # Preserve refined geometry through association and graph output.\n"
SCRIPT_FINAL_NEW = ("    # Preserve refined geometry through association and graph output.\n"
                    "    coords = _v1284_finalize(coords, ds_arr)\n")


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"{label}: expected one anchor, found {text.count(old)}")
    return text.replace(old, new, 1)


def output_mode_module(source: str) -> str:
    """x138's v1284_coordinate_refinement.py source -> same plus 'output' mode and finalize()."""
    source = _replace_once(source, MODULE_CACHE_OLD, MODULE_CACHE_NEW, "module cache")
    return _replace_once(source, MODULE_RETURN_OLD, MODULE_RETURN_NEW, "module return")


def output_mode_script(text: str) -> str:
    """Patched predict script (x138 V1284 patch applied) -> finalize() call after coordinate scaling."""
    text = _replace_once(text, SCRIPT_IMPORT_OLD, SCRIPT_IMPORT_NEW, "script import")
    return _replace_once(text, SCRIPT_FINAL_OLD, SCRIPT_FINAL_NEW, "script finalize")
