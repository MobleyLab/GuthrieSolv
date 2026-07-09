#!/usr/bin/env python3
"""Cross-validate GuthrieSolv's solubility (AQSOL) and vapour-pressure (VP)
extractions against independent references: AqSolDB (mcsorkun) and unified_VP."""
import sys, json
import numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harmonize_guthrie import to_molL, to_pascal, parse_val
from rdkit import Chem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
OUT = Path(__file__).resolve().parent / "outputs"

def canon(s):
    m = Chem.MolFromSmiles(str(s)); return Chem.MolToSmiles(m) if m else None

def stats(a, b):
    d = np.asarray(a) - np.asarray(b)
    return dict(n=len(d), MAE=round(float(np.mean(np.abs(d))), 3),
                RMSE=round(float(np.sqrt(np.mean(d**2))), 3), bias=round(float(d.mean()), 3),
                R=round(float(np.corrcoef(a, b)[0, 1]), 3))

g = pd.read_csv(Path(__file__).resolve().parent.parent / "guthrie_database.csv", encoding="latin1", low_memory=False)
g["value1"] = g["value1"].map(parse_val); g["MW"] = pd.to_numeric(g["MW"], errors="coerce")
g["proc"] = g["process"].astype(str).str.strip()

# solubility
aq = pd.read_csv("/Users/tgg/Github/AqSolDB/results/data_curated.csv", low_memory=False)
aq["cs"] = aq["SMILES"].map(canon); aqS = aq.dropna(subset=["cs"]).groupby("cs")["Solubility"].median()
gs = g[g["proc"] == "AQSOL"].copy()
gs["molL"] = [to_molL(str(u).strip(), v, mw) for u, v, mw in zip(gs["dimension1"], gs["value1"], gs["MW"])]
gs = gs[np.isfinite(gs["molL"]) & (gs["molL"] > 0)]; gs["cs"] = gs["mol"].map(canon)
gS = gs.groupby("cs")["molL"].median().apply(np.log10)
cs = sorted(set(aqS.index) & set(gS.index))
sol = stats(gS.loc[cs].values, aqS.loc[cs].values)

# VP
gv = g[g["proc"] == "VP"].copy(); gv["Pa"] = [to_pascal(str(u).strip(), v) for u, v in zip(gv["dimension1"], gv["value1"])]
gv = gv[np.isfinite(gv["Pa"]) & (gv["Pa"] > 0)]; gv["cs"] = gv["mol"].map(canon)
gVP = gv.groupby("cs")["Pa"].median().apply(np.log10)
uv = pd.read_csv("/Users/tgg/Downloads/physchem-v36/physchem_v36/data/physchemprops_gcs_cache/unified_VP_logPa_25C.csv", low_memory=False)
uv["cs"] = uv["smiles"].map(canon); uVP = uv.dropna(subset=["logVP_Pa"]).groupby("cs")["logVP_Pa"].median()
cv = sorted(set(gVP.index) & set(uVP.index)); vp = stats(gVP.loc[cv].values, uVP.loc[cv].values)

rep = {"GuthrieSolv_AQSOL_vs_AqSolDB": sol, "GuthrieSolv_VP_vs_unifiedVP": vp}
(OUT / "guthrie_vs_aqsoldb_vp.json").write_text(json.dumps(rep, indent=2))
print(json.dumps(rep, indent=2))

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(12, 5.3))
for a, (x, y, name, ref, s, c) in zip(ax, [
    (aqS.loc[cs].values, gS.loc[cs].values, "solubility (log mol/L)", "AqSolDB", sol, "#0d6b74"),
    (uVP.loc[cv].values, gVP.loc[cv].values, "vapour pressure (log Pa)", "unified_VP", vp, "#a8620a")]):
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    a.plot([lo, hi], [lo, hi], "--", color="#999", lw=1)
    a.scatter(x, y, s=12, c=c, alpha=.4, edgecolor="none")
    a.set_xlabel(f"{ref}  {name}"); a.set_ylabel(f"GuthrieSolv  {name}")
    a.set_title(f"GuthrieSolv vs {ref}\nMAE={s['MAE']}  R={s['R']}  bias={s['bias']:+}  n={s['n']}", fontsize=11)
    a.set_aspect("equal")
    for sp in ["top", "right"]: a.spines[sp].set_visible(False)
fig.suptitle("GuthrieSolv extraction validated against independent solubility / VP references", fontsize=12, y=1.02)
fig.tight_layout(); fig.savefig(OUT / "guthrie_vs_aqsoldb_vp.png", dpi=130, bbox_inches="tight")
print("saved outputs/guthrie_vs_aqsoldb_vp.png")
