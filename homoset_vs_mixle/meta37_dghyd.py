#!/usr/bin/env python3
"""Generate dG_hyd from meta37 (Henry, and VP x WS pairing) + unified_VP, then
compare to FreeSolv (truth) and to the GuthrieSolv Homoset/mixle reconciliation.

dG_hyd(Ben-Naim) = +1.364 * logKaw at 298 K (log K_aw = log C_air/C_water).
meta37 logHenrycc sign is calibrated to FreeSolv (a ~ +/-1.364)."""
import json
import numpy as np, pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
HERE = Path(__file__).resolve().parent; OUT = HERE / "outputs"
P36 = Path("/Users/tgg/Downloads/physchem-v36/physchem_v36/data/physchemprops_gcs_cache")

def ik(s):
    m = Chem.MolFromSmiles(str(s)); return Chem.MolToInchiKey(m) if m else None
def stats(p, t):
    p, t = np.asarray(p, float), np.asarray(t, float); d = p - t
    return dict(n=len(d), MAE=round(float(np.mean(np.abs(d))), 3),
                RMSE=round(float(np.sqrt(np.mean(d**2))), 3), R=round(float(np.corrcoef(p, t)[0, 1]), 3))

# FreeSolv truth
fs = json.load(open("/Users/tgg/Github/FreeSolv/database.json")); exd = {}
for cid, d in fs.items():
    m = Chem.MolFromSmiles(d["smiles"])
    if m: exd[Chem.MolToInchiKey(m)] = float(d["expt"])

# meta37 -> ikey with Henry / VP / WS
m = pd.read_csv(P36 / "meta37v2025curated.csv", low_memory=False)
for c in ["logHenrycc", "logVP", "logWS"]:
    m[c] = pd.to_numeric(m[c], errors="coerce")
m = m[m[["logHenrycc", "logVP", "logWS"]].notna().any(axis=1)].copy()
m["ikey"] = m["smiles_canonical"].map(ik); m = m.dropna(subset=["ikey"])
mH = m.dropna(subset=["logHenrycc"]).groupby("ikey")["logHenrycc"].median()
mVP = m.dropna(subset=["logVP"]).groupby("ikey")["logVP"].median()
mWS = m.dropna(subset=["logWS"]).groupby("ikey")["logWS"].median()
# unified_VP as extra VP
uv = pd.read_csv(P36 / "unified_VP_logPa_25C.csv", low_memory=False); uv["ikey"] = uv["smiles"].map(ik)
uVP = uv.dropna(subset=["ikey", "logVP_Pa"]).groupby("ikey")["logVP_Pa"].median()
VP = uVP.combine_first(mVP)                       # prefer unified_VP, fall back to meta37 VP

# --- calibrate the two Henry proxies to FreeSolv (sign + offset) ---
def calib_and_gen(series, name):
    c = [i for i in series.index if i in exd]
    x = np.array([series[i] for i in c]); y = np.array([exd[i] for i in c])
    a, b = np.polyfit(x, y, 1)                     # dG ~ a*proxy + b
    gen = pd.Series(a * series + b)                # generated dG_hyd for ALL molecules
    return gen, {"n_molecules_generated": int(len(gen)), "calib_slope": round(float(a), 3),
                 "calib_intercept": round(float(b), 3), "freesolv_fit": stats(a * x + b, y)}

dg_H, repH = calib_and_gen(mH, "logHenrycc")
# VP x WS Henry (dimensionless): logKaw ~ logVP - logWS (const absorbed by calibration)
vw_idx = sorted(set(VP.index) & set(mWS.index))
HVW = pd.Series({i: VP[i] - mWS[i] for i in vw_idx})
dg_VW, repVW = calib_and_gen(HVW, "VPxWS")

# union of generated dG (prefer Henry, fall back to VPxWS)
dg_all = dg_H.combine_first(dg_VW)

# GuthrieSolv reconciled (Homoset L=0.6 + mixle)
gu = pd.read_csv(OUT / "per_molecule_estimates.csv").set_index("ikey")
hs, mx = gu["homoset_L0.6_noise25"], gu["mixle_robust"]

rep = {
    "dG_from_meta37_Henry": repH,
    "dG_from_VPxWS_pairing": repVW,
    "dG_total_union_molecules": int(len(dg_all)),
    "overlap_with_FreeSolv": None, "overlap_with_GuthrieSolv": None,
}
fs_c = [i for i in dg_all.index if i in exd]
rep["overlap_with_FreeSolv"] = {"n": len(fs_c),
    "meta37_dG_vs_FreeSolv": stats([dg_all[i] for i in fs_c], [exd[i] for i in fs_c])}
gu_c = [i for i in dg_all.index if i in gu.index and np.isfinite(hs.get(i, np.nan))]
rep["overlap_with_GuthrieSolv"] = {
    "n": len(gu_c),
    "meta37_vs_GuthrieSolv_Homoset": stats([dg_all[i] for i in gu_c], [hs[i] for i in gu_c]),
    "meta37_vs_GuthrieSolv_mixle": stats([dg_all[i] for i in gu_c], [mx[i] for i in gu_c]),
}
# and: which is closer to FreeSolv on the 3-way overlap (meta37 vs Guthrie-HS vs Guthrie-mixle)?
tri = [i for i in fs_c if i in gu.index and np.isfinite(hs.get(i, np.nan))]
if tri:
    rep["three_way_vs_FreeSolv"] = {"n": len(tri),
        "meta37_Henry": stats([dg_all[i] for i in tri], [exd[i] for i in tri]),
        "GuthrieSolv_Homoset": stats([hs[i] for i in tri], [exd[i] for i in tri]),
        "GuthrieSolv_mixle": stats([mx[i] for i in tri], [exd[i] for i in tri])}
pd.DataFrame({"ikey": dg_all.index, "dG_hyd_meta37": dg_all.values}).to_csv(OUT / "meta37_dghyd.csv", index=False)
(OUT / "meta37_dghyd_report.json").write_text(json.dumps(rep, indent=2))
print(json.dumps(rep, indent=2))
