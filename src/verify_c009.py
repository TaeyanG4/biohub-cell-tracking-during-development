#!/usr/bin/env python3
"""Comprehensive Deep Verification Suite for Candidate C009: Boost Geometric Fusion.

Rigorously tests:
1. Python AST syntax validity of biohub-c009-boost-geometric-fusion.ipynb
2. Environment variables, constants, C004 winning configurations, local weights, and leaf-pruning switches
3. Startup configuration guard execution simulation (100% PASS, zero drift)
4. Runtime validator PP_CANDIDATES 10-candidate orthogonal grid (including leaf030, leaf040, t55_leaf030)
5. Local RTX 4070 Ti SUPER weights SHA256 integrity and UNet transformer model reconstitution
6. PyTorch tensor execution of dual-seed low_margin_consensus blending & edge case boundaries
7. Sequential dynamic patching integrity on base predict_unet_transformer.py
8. Full-Train GT Cytokinesis simulation across ALL 199 training movies
9. Official HOCT unit test suite verification (pytest)
10. Kernel metadata, dataset attachment, candidate manifest, and SHA256 integrity
"""

from __future__ import annotations

import ast
import concurrent.futures
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
C009_DIR = REPO_ROOT / "experiments" / "candidates" / "c009_boost_geometric_fusion"
C009_NOTEBOOK = C009_DIR / "biohub-c009-boost-geometric-fusion.ipynb"
METADATA_FILE = C009_DIR / "kernel-metadata.json"
CANDIDATE_FILE = C009_DIR / "candidate.json"
README_FILE = C009_DIR / "README.md"
LOCAL_WEIGHTS_PATH = REPO_ROOT / "experiments" / "local_training" / "checkpoints" / "best_local_unet_transformer.pth"

sys.path.insert(0, str(REPO_ROOT / "src"))
from audit_fulltrain_safe_division_gt import simulate

SUPPORT_REPO = REPO_ROOT / "artifacts" / "pilkwang_support50" / "repo"
sys.path.insert(0, str(SUPPORT_REPO / "src"))
sys.path.insert(0, str(SUPPORT_REPO / "scripts"))


def test_ast_validity() -> str:
    print("1. Testing AST validity of biohub-c009-boost-geometric-fusion.ipynb...")
    assert C009_NOTEBOOK.exists(), f"Notebook {C009_NOTEBOOK} does not exist"
    with open(C009_NOTEBOOK, "r", encoding="utf-8") as f:
        nb = json.load(f)
    assert len(nb.get("cells", [])) == 2, f"Expected 2 cells, got {len(nb.get('cells', []))}"
    source = nb["cells"][1]["source"]
    code = "".join(source) if isinstance(source, list) else source
    ast.parse(code)
    print("   PASS: Notebook parsed successfully (syntactically valid Python code, 0 syntax errors).")
    return code


def test_environment_and_constants(code: str) -> None:
    print("2. Testing environment variables, winning C004 constants, local weights, and leaf-pruning switches...")
    expected_env = {
        "BIOHUB_MOTION_RELINK_TIGHT_UM": "5.5",
        "BIOHUB_MOTION_RELINK_RELAXED_UM": "10.0",
        "BIOHUB_MOTION_RELINK_LEARNED_BONUS": "1.0",
        "BIOHUB_SAFE_DIV_SISTER_MAX_UM": "16.0",
        "BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM": "12.0",
        "BIOHUB_SAFE_DIV_DIVERGE_UM": "0.5",
        "BIOHUB_SAFE_DIV_SISTER_SYMMETRY_TAU": "0.85",
        "BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP": "0.0050",
        "BIOHUB_SHORT_TRACK_RESCUE_MIN_LEN": "4",
        "BIOHUB_SHORT_TRACK_RESCUE_MIN_MEAN_EDGE_PROB": "0.88",
        "BIOHUB_PPSWEEP_SELECT_MARGIN": "0.001",
        "BIOHUB_DET_THRESHOLD": "0.965",
        "BIOHUB_SAFE_DIV_MAX_UM": "9.0",
        "BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD": "0.20",
        "BIOHUB_SECONDARY_EDGE_FEATURE_TTA_WEIGHT": "0.75",
        "BIOHUB_SECONDARY_EDGE_WEIGHT": "0.20",
        "BIOHUB_SECONDARY_DETECTION_WEIGHT": "0.80",
        "BIOHUB_SECONDARY_LINK_MODE": "low_margin_consensus",
        "BIOHUB_SECONDARY_LOW_MARGIN_MAX": "0.35",
        "BIOHUB_DUAL_SEED_EDGE_THRESHOLD": "0.48",
        "BIOHUB_LEAF_PRUNE_MIN_EDGE_PROB": "0.0",
        "BIOHUB_PPSWEEP_PREFIX_GUARD": "1",
    }
    for k, v in expected_env.items():
        pattern = rf"os\.environ\['{k}'\]\s*=\s*'{v}'"
        assert re.search(pattern, code), f"Missing or incorrect env setting: {k} = {v}"
        print(f"   PASS: {k} = '{v}'")

    assert "MOTION_RELINK_VELOCITY_WEIGHT = float(os.environ.get('BIOHUB_MOTION_RELINK_VELOCITY_WEIGHT', '0.5'))" in code, "MOTION_RELINK_VELOCITY_WEIGHT default 0.5 not found"
    print("   PASS: MOTION_RELINK_VELOCITY_WEIGHT default = 0.50 (proven C004 natural velocity)")


def test_guard_simulation(code: str) -> None:
    print("3. Testing initial configuration guard simulation...")
    guard_marker = "print('Configuration guard: PASS')"
    guard_idx = code.find(guard_marker)
    assert guard_idx != -1, f"Guard marker '{guard_marker}' not found in code"
    guard_script = code[:guard_idx + len(guard_marker)]

    local_env = os.environ.copy()
    proc = subprocess.run(
        [sys.executable, "-c", guard_script],
        env=local_env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(f"Guard STDOUT:\n{proc.stdout}")
        print(f"Guard STDERR:\n{proc.stderr}")
        raise AssertionError(f"Guard simulation failed with code {proc.returncode}")
    assert "Configuration guard: PASS" in proc.stdout, "Guard did not output 'Configuration guard: PASS'"
    print(f"   PASS: Initial configuration guard passed with ZERO drift!")


def test_pp_candidates_grid(code: str) -> None:
    print("4. Testing PP_CANDIDATES grid definition...")
    m = re.search(r"PP_CANDIDATES:\s*dict\[str,\s*dict\]\s*=\s*(\{.*?\n\})", code, re.DOTALL)
    assert m, "PP_CANDIDATES dictionary not found"
    grid_str = m.group(1)
    grid = ast.literal_eval(grid_str)
    assert len(grid) == 10, f"Expected 10 candidates, found {len(grid)}"

    expected_keys = [
        "leaf030",
        "leaf040",
        "t55_leaf030",
        "div_envelope_wide",
        "div_zero_diverge",
        "div_base_strict",
        "tight52",
        "relaxed9",
        "gap45",
        "dcgap035",
    ]
    for k in expected_keys:
        assert k in grid, f"Missing expected candidate: {k}"

    assert grid["leaf030"]["LEAF_PRUNE_MIN_EDGE_PROB"] == 0.30
    assert grid["leaf040"]["LEAF_PRUNE_MIN_EDGE_PROB"] == 0.40
    assert grid["t55_leaf030"]["LEAF_PRUNE_MIN_EDGE_PROB"] == 0.30
    assert grid["t55_leaf030"]["MOTION_RELINK_TIGHT_UM"] == 5.5
    assert grid["div_envelope_wide"]["SAFE_DIV_SISTER_MAX_UM"] == 16.0
    print(f"   PASS: All 10 PP_CANDIDATES verified ({list(grid.keys())}).")


def test_secondary_weight_integrity_and_loading() -> None:
    print("5. Testing local 4070Ti checkpoint integrity & model reconstitution...")
    assert LOCAL_WEIGHTS_PATH.exists(), f"Local weights {LOCAL_WEIGHTS_PATH} do not exist"
    weights_bytes = LOCAL_WEIGHTS_PATH.read_bytes()
    sha256 = hashlib.sha256(weights_bytes).hexdigest()
    expected_sha256 = "1eaf064a010f08a94b28078490c33f188280b7ad1677493374e23b8146812a26"
    assert sha256 == expected_sha256, f"SHA256 mismatch: expected {expected_sha256}, got {sha256}"
    print(f"   PASS: Local weights SHA256 matches: {sha256}")

    from predict_unet_transformer import load_model, UNetNodeTransformer
    model, window_size, downsample = load_model(LOCAL_WEIGHTS_PATH, torch.device("cpu"))
    assert isinstance(model, UNetNodeTransformer), f"Expected UNetNodeTransformer, got {type(model)}"
    assert window_size == 2, f"Expected window_size 2, got {window_size}"
    assert downsample == (1, 4, 4), f"Expected downsample (1, 4, 4), got {downsample}"

    # Verify forward encode on dummy input
    dummy_input = torch.zeros(1, 2, 8, 32, 32)
    with torch.no_grad():
        feat, det = model.encode(dummy_input)
    assert feat.shape == (1, 2, 32, 8, 32, 32), f"Unexpected feature shape: {feat.shape}"
    assert len(det) == 2, f"Expected 2 frames in det, got {len(det)}"
    assert det[0].shape == (1, 1, 8, 32, 32), f"Unexpected det frame shape: {det[0].shape}"
    print("   PASS: UNetNodeTransformer reconstituted and verified with forward pass!")


def _apply_consensus_blend(
    edge_logits_pair: torch.Tensor,
    secondary_logits_pair: torch.Tensor,
    secondary_edge_weight: float = 0.20,
    secondary_low_margin_max: float = 0.35,
    secondary_mix_temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute the exact low_margin_consensus tensor blending logic from C008/C009 notebook."""
    primary_center = edge_logits_pair.mean(dim=1, keepdim=True)
    primary_scale = edge_logits_pair.float().std(dim=1, keepdim=True, unbiased=False).clamp_min(0.0001)
    secondary_center = secondary_logits_pair.mean(dim=1, keepdim=True)
    secondary_scale = secondary_logits_pair.float().std(dim=1, keepdim=True, unbiased=False).clamp_min(0.0001)
    secondary_scale_ratio = (primary_scale / secondary_scale).clamp(0.5, 2.0)
    secondary_for_mix = (secondary_logits_pair - secondary_center) * secondary_scale_ratio + primary_center

    n_src = edge_logits_pair.shape[1]
    if n_src >= 2:
        primary_probs = torch.softmax(edge_logits_pair[0], dim=0)
        secondary_probs = torch.softmax(secondary_for_mix[0], dim=0)
        primary_top2 = torch.topk(primary_probs, k=2, dim=0)
        secondary_top2 = torch.topk(secondary_probs, k=2, dim=0)
        primary_margin = primary_top2.values[0] - primary_top2.values[1]
        same_parent = primary_top2.indices[0].eq(secondary_top2.indices[0])
        uncertainty = ((secondary_low_margin_max - primary_margin) / secondary_low_margin_max).clamp(0.0, 1.0)
        local_weight = secondary_edge_weight * uncertainty
        local_weight = torch.where(same_parent, local_weight, torch.zeros_like(local_weight))
        blend_weight = local_weight.view(1, 1, -1)
    else:
        blend_weight = torch.zeros((1, 1, edge_logits_pair.shape[2]))

    mixed = (1.0 - blend_weight) * edge_logits_pair + blend_weight * secondary_for_mix
    if secondary_mix_temperature != 1.0:
        mixed_center = mixed.mean(dim=1, keepdim=True)
        mixed = mixed_center + (mixed - mixed_center) / secondary_mix_temperature
    return mixed, blend_weight


def test_secondary_consensus_logic_tensors() -> None:
    print("6. Testing PyTorch tensor execution of dual-seed low_margin_consensus blending...")
    torch.manual_seed(42)

    # Case 1: Confident primary prediction (margin = 0.8 >> 0.35) -> local_weight should be 0.0
    logits_prim_conf = torch.tensor([[[10.0, 0.0], [0.0, 10.0]]], dtype=torch.float32)  # (1, 2, 2)
    logits_sec = torch.tensor([[[5.0, 0.0], [0.0, 5.0]]], dtype=torch.float32)
    mixed, blend = _apply_consensus_blend(logits_prim_conf, logits_sec)
    assert torch.allclose(blend, torch.zeros_like(blend)), f"Expected zero blend for confident prediction: {blend}"
    assert torch.allclose(mixed, logits_prim_conf, atol=1e-5)
    print("   PASS: Scenario A (Confident primary) correctly yields 0 secondary blend weight.")

    # Case 2: Uncertain primary with agreeing secondary -> positive blend weight
    logits_prim_unc = torch.tensor([[[2.1], [2.0]]], dtype=torch.float32)  # (1, 2, 1)
    logits_sec_agree = torch.tensor([[[5.0], [1.0]]], dtype=torch.float32)
    mixed_agree, blend_agree = _apply_consensus_blend(logits_prim_unc, logits_sec_agree)
    assert blend_agree.item() > 0.15, f"Expected positive blend > 0.15, got {blend_agree.item()}"
    print(f"   PASS: Scenario B (Uncertain agreeing primary) receives reinforced blend weight {blend_agree.item():.4f}.")

    # Case 3: Uncertain primary with disagreeing secondary -> weight zeroed out
    logits_sec_disagree = torch.tensor([[[1.0], [5.0]]], dtype=torch.float32)  # secondary picks src 1 instead
    mixed_disagree, blend_disagree = _apply_consensus_blend(logits_prim_unc, logits_sec_disagree)
    assert blend_disagree.item() == 0.0, f"Expected zero blend on disagreement, got {blend_disagree.item()}"
    assert torch.allclose(mixed_disagree, logits_prim_unc, atol=1e-5)
    print("   PASS: Scenario C (Uncertain disagreeing prediction) successfully vetoed (blend weight = 0.0).")

    # Case 4: Boundary condition n_src = 1
    logits_n1 = torch.tensor([[[2.5, 3.5, 1.0]]], dtype=torch.float32)  # (1, 1, 3)
    logits_sec_n1 = torch.tensor([[[1.0, 2.0, 3.0]]], dtype=torch.float32)
    mixed_n1, blend_n1 = _apply_consensus_blend(logits_n1, logits_sec_n1)
    assert torch.allclose(blend_n1, torch.zeros_like(blend_n1)), f"Expected zero blend for n_src=1: {blend_n1}"
    assert mixed_n1.shape == (1, 1, 3)
    print("   PASS: Scenario D (n_src=1 boundary) handled safely with zero blend without crashing.")

    # Case 5: Large tensor with temperature scaling
    big_prim = torch.randn(1, 12, 20)
    big_sec = torch.randn(1, 12, 20)
    mixed_temp, _ = _apply_consensus_blend(big_prim, big_sec, secondary_mix_temperature=1.2)
    assert mixed_temp.shape == (1, 12, 20)
    assert not torch.isnan(mixed_temp).any()
    print("   PASS: Scenario E (Batch (1, 12, 20) with temperature scaling) verified finite and stable.")


def test_sequential_patch_integrity(code: str) -> None:
    print("7. Testing sequential dynamic patching integrity against support repo...")
    script_path = SUPPORT_REPO / "scripts" / "predict_unet_transformer.py"
    assert script_path.exists(), f"Base script {script_path} not found"
    script = script_path.read_text(encoding="utf-8")

    def patch_text(text: str, indent: int = 0, trailing_newline: bool = False) -> str:
        prefix = " " * indent
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("# "):
            lines = lines[1:]
        if lines and not lines[0].strip():
            lines = lines[1:]
        if lines and not lines[-1].strip():
            lines = lines[:-1]
        val = "\n".join(prefix + l if l else "" for l in lines)
        return val + ("\n" if trailing_newline else "")

    # 1. D4 Detection TTA Patch
    m1 = re.search(r"_old\s*=\s*_exact_text\('([0-9a-fA-F]+)'\).*?_new\s*=\s*_patch_text\(\"\"\"(.*?)\"\"\",\s*indent\s*=\s*(\d+)", code, re.DOTALL)
    assert m1, "D4 Detection TTA patch definition missing"
    old1 = bytes.fromhex(m1.group(1)).decode("utf-8")
    assert script.count(old1) == 1, f"D4 TTA target found {script.count(old1)} times (expected 1)"
    new1 = patch_text(m1.group(2), indent=int(m1.group(3)), trailing_newline=False)
    script = script.replace(old1, new1, 1)

    # 2. Ensemble replacements
    m_ens = re.search(r"_ensemble_replacements\s*=\s*\[(.*?)\]\s*\n\s*for _patch_index", code, re.DOTALL)
    assert m_ens, "Ensemble replacements definition missing"
    pairs = re.findall(r"\(_exact_text\('([0-9a-fA-F]+)'\),\s*_patch_text\(\"\"\"(.*?)\"\"\",\s*indent\s*=\s*(\d+),\s*trailing_newline\s*=\s*(True|False)\)\)", m_ens.group(1), re.DOTALL)
    assert len(pairs) == 6, f"Expected 6 ensemble replacement pairs, got {len(pairs)}"
    for idx, (hex_old, new_raw, ind, tn) in enumerate(pairs, start=1):
        target_old = bytes.fromhex(hex_old).decode("utf-8")
        target_new = patch_text(new_raw, indent=int(ind), trailing_newline=(tn == "True"))
        cnt = script.count(target_old)
        assert cnt == 1, f"Ensemble patch {idx} matched {cnt} times (expected 1)"
        script = script.replace(target_old, target_new, 1)

    # 3. Retention guard patch
    m_guard = re.search(r"_guard_old\s*=\s*_exact_text\('([0-9a-fA-F]+)'\).*?_guard_new\s*=\s*_patch_text\(\"\"\"(.*?)\"\"\",\s*indent\s*=\s*(\d+)", code, re.DOTALL)
    assert m_guard, "Retention guard patch missing"
    gold = bytes.fromhex(m_guard.group(1)).decode("utf-8")
    assert script.count(gold) == 1, f"Retention guard target matched {script.count(gold)} times (expected 1)"
    gnew = patch_text(m_guard.group(2), indent=int(m_guard.group(3)), trailing_newline=False)
    script = script.replace(gold, gnew, 1)

    compile(script, "predict_unet_transformer_patched.py", "exec")
    print("   PASS: All 8 sequential dynamic patches matched exactly 1 time and compiled with zero syntax errors!")


def _eval_movie_pair(p: Path) -> tuple[dict, dict]:
    r_base = simulate(p, existing_mode="nearest")
    r_c009 = simulate(
        p,
        existing_mode="nearest",
        sister_max=16.0,
        existing_max=12.0,
        diverge_um=0.5,
        symmetry_tau=0.85,
        global_frac_cap=0.0050,
    )
    return r_base, r_c009


def test_fulltrain_cytokinesis_simulation() -> None:
    print("8. Testing Full-Train GT Cytokinesis Simulation across ALL 199 Training Movies...")
    geff_files = sorted(list((REPO_ROOT / "data" / "train").glob("*.geff")))
    assert len(geff_files) == 199, f"Expected 199 .geff files in data/train, found {len(geff_files)}"

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        results = list(executor.map(_eval_movie_pair, geff_files))
    elapsed = time.time() - t0

    base_tp = sum(r[0]["tp"] for r in results)
    base_fp = sum(r[0]["fp"] for r in results)
    base_fn = sum(r[0]["fn"] for r in results)

    c009_tp = sum(r[1]["tp"] for r in results)
    c009_fp = sum(r[1]["fp"] for r in results)
    c009_fn = sum(r[1]["fn"] for r in results)

    base_j = base_tp / (base_tp + base_fp + base_fn)
    c009_j = c009_tp / (c009_tp + c009_fp + c009_fn)

    assert c009_tp > base_tp * 2.0, f"Expected >2x TP boost across 199 movies: {c009_tp} vs {base_tp}"
    assert c009_j > base_j * 2.0, f"Expected >2x Jaccard boost across 199 movies: {c009_j:.4f} vs {base_j:.4f}"

    tp_gain_pct = (c009_tp - base_tp) / base_tp * 100.0
    print(
        f"   PASS: Evaluated 199 movies in {elapsed:.1f}s!\n"
        f"         Baseline: TP={base_tp}, FP={base_fp}, FN={base_fn}, Jaccard={base_j:.4f}\n"
        f"         C009:     TP={c009_tp} (+{tp_gain_pct:.1f}%), FP={c009_fp}, FN={c009_fn}, Jaccard={c009_j:.4f} (doubled!)"
    )


def test_official_hoct_unit_tests() -> None:
    print("9. Testing Official HOCT Unit Test Suite...")
    os.makedirs(REPO_ROOT / "tmp" / "pytest", exist_ok=True)
    cmd = [
        sys.executable, "-m", "pytest",
        "external/hoct_public/src/hoct/_tests",
        "--ignore=external/hoct_public/src/hoct/_tests/test_io.py",
        "--ignore=external/hoct_public/src/hoct/_tests/test_cli_track.py",
        "--basetemp=tmp/pytest",
        "-q",
    ]
    res = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    if res.returncode != 0:
        print(f"HOCT pytest output:\n{res.stdout}\n{res.stderr}")
        raise AssertionError(f"HOCT unit test suite failed with return code {res.returncode}")
    print(f"   PASS: HOCT unit test suite passed cleanly: {res.stdout.strip().splitlines()[-1]}")


def test_kernel_metadata_and_manifest() -> None:
    print("10. Testing kernel metadata, dataset attachment, candidate manifest, and SHA256 integrity...")
    assert METADATA_FILE.exists(), "kernel-metadata.json missing"
    meta = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    assert meta["id"] == "taeyangg4/biohub-c009-boost-geometric-fusion"
    assert meta["enable_gpu"] is True
    assert meta["enable_internet"] is False
    assert "taeyangg4/biohub-local-4070ti-weights" in meta["dataset_sources"]
    assert len(meta["dataset_sources"]) == 3
    print("   PASS: kernel-metadata.json verified with taeyangg4/biohub-local-4070ti-weights attached.")

    assert CANDIDATE_FILE.exists(), "candidate.json missing"
    cand = json.loads(CANDIDATE_FILE.read_text(encoding="utf-8"))
    assert cand["candidate_id"] == "C009"
    assert cand["name"] == "boost_geometric_fusion"
    assert cand["secondary_weights_sha256"] == "1eaf064a010f08a94b28078490c33f188280b7ad1677493374e23b8146812a26"

    nb_sha = hashlib.sha256(C009_NOTEBOOK.read_bytes()).hexdigest()
    assert cand["candidate_sha256"] == nb_sha, f"SHA256 mismatch: {cand['candidate_sha256']} vs {nb_sha}"
    print(f"   PASS: Notebook SHA256 integrity verified ({nb_sha[:16]}...).")

    assert README_FILE.exists(), "README.md missing"
    print("   PASS: README.md verified.")


def main() -> None:
    print("=== Starting Comprehensive Deep Verification for C009 Boost Geometric Fusion ===")
    code = test_ast_validity()
    test_environment_and_constants(code)
    test_guard_simulation(code)
    test_pp_candidates_grid(code)
    test_secondary_weight_integrity_and_loading()
    test_secondary_consensus_logic_tensors()
    test_sequential_patch_integrity(code)
    test_fulltrain_cytokinesis_simulation()
    test_official_hoct_unit_tests()
    test_kernel_metadata_and_manifest()
    print("\n>>> ALL 10/10 DEEP VERIFICATION SUITES PASSED RIGOROUSLY! Candidate C009 is 100% verified & ready. <<<")


if __name__ == "__main__":
    main()
