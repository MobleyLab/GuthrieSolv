#!/usr/bin/env python3
"""Load harmonized Guthrie observations, join to FreeSolv, report per-route
residuals and per-molecule raw-consensus accuracy. Diagnostic gate."""
import pandas as pd, numpy as np, json, sys
from rdkit import Chem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
from pathlib import Path
HERE = Path(__file__).resolve().parent

def freesolv():
    fs = json.load(open("/Users/tgg/Github/FreeSolv/database.json"))
    rows = []
    for cid, d in fs.items():
        m = Chem.MolFromSmiles(d["smiles"])
        if m:
            rows.append((Chem.MolToInchiKey(m), float(d["expt"]), float(d.get("d_expt", np.nan))))
    return pd.DataFrame(rows, columns=["ikey", "expt", "d_expt"]).drop_duplicates("ikey")

def metrics(pred, truth):
    d = np.asarray(pred) - np.asarray(truth)
    return dict(n=len(d), MAE=float(np.mean(np.abs(d))), RMSE=float(np.sqrt(np.mean(d**2))),
                bias=float(d.mean()), R=float(np.corrcoef(pred, truth)[0, 1]) if len(d) > 2 else np.nan)

def main():
    obs = pd.read_csv(HERE / "outputs/guthrie_dg_observations.csv")
    fsdf = freesolv()
    j = obs.merge(fsdf, on="ikey", how="inner")
    j["resid"] = j["dg_hyd"] - j["expt"]
    print("=== per-OBSERVATION residual by route ===")
    print(j.groupby("route")["resid"].agg(["size", "mean", "median", "std"]).round(3).to_string())
    print("\n=== per-OBSERVATION residual by process ===")
    print(j.groupby("process")["resid"].agg(["size", "mean", "median", "std"]).round(3).to_string())
    g = obs.groupby("ikey").agg(n=("dg_hyd", "size"), mean=("dg_hyd", "mean"),
                                median=("dg_hyd", "median"), std=("dg_hyd", "std")).reset_index()
    m = g.merge(fsdf, on="ikey", how="inner")
    print(f"\n=== per-MOLECULE raw consensus vs FreeSolv (overlap={len(m)}) ===")
    for col in ["mean", "median"]:
        print(f"  {col:7s}", {k: round(v, 3) for k, v in metrics(m[col], m["expt"]).items()})
    mm = m[m["n"] >= 2]
    print(f"  --- molecules with >=2 obs ({len(mm)}) ---")
    for col in ["mean", "median"]:
        print(f"  {col:7s}", {k: round(v, 3) for k, v in metrics(mm[col], mm["expt"]).items()})

if __name__ == "__main__":
    main()
