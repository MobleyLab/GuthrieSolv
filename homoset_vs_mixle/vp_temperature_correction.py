#!/usr/bin/env python3
"""Clausius-Clapeyron temperature-correct GuthrieSolv VP to 25 C using meta37 dHvap.

  log10 P(298.15) = log10 P(T_meas) - (dHvap/R)/ln10 * (1/298.15 - 1/T_meas)

dHvap from meta37 (deltaHvap_kJmol). Then re-check agreement vs unified_VP (25 C)
and the Henry pairing (VP - WS)."""
import sys, json
import numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harmonize_guthrie import to_pascal, to_kelvin, parse_val, to_molL
from rdkit import Chem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

OUT = Path(__file__).resolve().parent / "outputs"
P36 = Path("/Users/tgg/Downloads/physchem-v36/physchem_v36/data/physchemprops_gcs_cache")
R = 8.314462; LN10 = np.log(10); T0 = 298.15

def canon(s):
    m = Chem.MolFromSmiles(str(s)); return Chem.MolToSmiles(m) if m else None

def stats(a, b):
    d = np.asarray(a) - np.asarray(b)
    return dict(n=len(d), MAE=round(float(np.mean(np.abs(d))), 3), R=round(float(np.corrcoef(a, b)[0, 1]), 3))

g = pd.read_csv(Path(__file__).resolve().parent.parent / "guthrie_database.csv", encoding="latin1", low_memory=False)
g["value1"] = g["value1"].map(parse_val); g["MW"] = pd.to_numeric(g["MW"], errors="coerce")
g["proc"] = g["process"].astype(str).str.strip()

# meta37 dHvap (kJ/mol) per molecule
m = pd.read_csv(P36 / "meta37v2025curated.csv", low_memory=False)
m["dHv"] = pd.to_numeric(m["deltaHvap_kJmol"], errors="coerce") * 1000.0   # J/mol
m["cs"] = m["smiles_canonical"].map(canon)
dHv = m.dropna(subset=["dHv"]).groupby("cs")["dHv"].median()

v = g[g["proc"] == "VP"].copy()
v["Pa"] = [to_pascal(str(u).strip(), val) for u, val in zip(v["dimension1"], v["value1"])]
v = v[np.isfinite(v["Pa"]) & (v["Pa"] > 0)].copy()
v["logPa"] = np.log10(v["Pa"]); v["T"] = v["te53p"].map(to_kelvin); v["cs"] = v["mol"].map(canon)
v["dHv"] = v["cs"].map(dHv)
# Clausius-Clapeyron correction to 25 C (only where dHvap known)
corr = (v["dHv"] / R / LN10) * (1.0 / T0 - 1.0 / v["T"])
v["logPa_25"] = np.where(v["dHv"].notna(), v["logPa"] - corr, np.nan)

uv = pd.read_csv(P36 / "unified_VP_logPa_25C.csv", low_memory=False)
uv["cs"] = uv["smiles"].map(canon); uVP = uv.dropna(subset=["logVP_Pa"]).groupby("cs")["logVP_Pa"].median()

raw = v.groupby("cs")["logPa"].median()
cor = v.dropna(subset=["logPa_25"]).groupby("cs")["logPa_25"].median()
cset = sorted(set(cor.index) & set(uVP.index))                 # molecules we could correct
rep = {"n_VP_rows": int(len(v)), "n_rows_with_dHvap": int(v["dHv"].notna().sum()),
       "coverage_pct": round(100 * v["dHv"].notna().mean(), 1),
       "vs_unifiedVP_on_correctable_molecules": {
           "raw": stats(raw.loc[cset].values, uVP.loc[cset].values),
           "CC_corrected_to_25C": stats(cor.loc[cset].values, uVP.loc[cset].values)}}
(OUT / "vp_cc_correction_report.json").write_text(json.dumps(rep, indent=2))
print(json.dumps(rep, indent=2))
v[["mol", "cs", "logPa", "T", "dHv", "logPa_25"]].to_csv(OUT / "guthrie_vp_cc_corrected.csv", index=False)
print("\nsaved outputs/guthrie_vp_cc_corrected.csv")
