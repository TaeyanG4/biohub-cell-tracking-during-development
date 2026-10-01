from pathlib import Path

import pandas as pd


root = Path(__file__).resolve().parents[1]
base = root / "experiments" / "exp_dctta_hoct_det0965_nolinefit" / "output_api"
before = pd.read_csv(base / "submission_before_hoct_veto.csv")
final = pd.read_csv(base / "submission.csv")

nodes = final[final.row_type.eq("node")].copy()
before_edges = before[before.row_type.eq("edge")].copy()
final_edges = final[final.row_type.eq("edge")].copy()

restore = []
for dataset, grp in before_edges.groupby("dataset"):
    counts = grp.source_id.astype(int).value_counts()
    div_sources = set(counts[counts >= 2].index.astype(int))
    restore.append(grp[grp.source_id.astype(int).isin(div_sources)])

restore_edges = pd.concat(restore, ignore_index=True) if restore else before_edges.iloc[:0].copy()
edges = pd.concat([final_edges, restore_edges], ignore_index=True)
edges = edges.drop_duplicates(subset=["dataset", "source_id", "target_id"])

out = pd.concat([nodes, edges], ignore_index=True)
out = out.sort_values(["dataset", "row_type", "node_id", "source_id", "target_id"]).reset_index(drop=True)
out["id"] = range(len(out))
out = out[final.columns]

dest = root / "experiments" / "edge_mix" / "nolinefit_mode1_reconstructed.csv"
dest.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(dest, index=False)
print("before_edges", len(before_edges), "mode2_edges", len(final_edges), "mode1_edges", len(edges))
print(dest)
