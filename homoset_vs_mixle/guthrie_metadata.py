#!/usr/bin/env python3
"""Guthrie's own metadata (Excel columns dropped from the flat pipeline):
  - `final`   : his own converted dG_hyd (kcal/mol, 737 rows) -> a THIRD reconciled
                estimate to score head-to-head against our Homoset/mixle vs FreeSolv.
  - `error1`  : his per-measurement TRUST FLAG -- the value 1.93 kcal/mol (and its kJ
                twin 8.08) marks data he judged "not particularly trustworthy" -> use as
                an observation weight / filter in the curation.
  - pH in `comments` : ionizable-compound conditions -> tag AQSOL rows.
"""
from __future__ import annotations
import sys, json, math
import numpy as np, pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from harmonize_guthrie import to_kelvin, parse_val, inchikey
from unit_rules import convert

OUT = HERE / "outputs"
SRC = HERE.parent / "guthrie_database.csv"
EXCLUDED = {"DGT", "DGTX", "DGB", "DGV", "PKA", "KA", "LAQACTCO"}
UNTRUSTED = {1.93, 8.08}          # Guthrie's "not trustworthy" flag (kcal and kJ twin)


def freesolv():
    fs = json.load(open("/Users/tgg/Github/FreeSolv/database.json"))
    d = {}
    for cid, v in fs.items():
        m = Chem.MolFromSmiles(v["smiles"])
        if m:
            d[Chem.MolToInchiKey(m)] = float(v["expt"])
    return d


def metrics(pred, truth):
    d = np.asarray(pred, float) - np.asarray(truth, float)
    ok = np.isfinite(d); d = d[ok]
    return dict(n=int(ok.sum()), MAE=round(float(np.mean(np.abs(d))), 3),
                RMSE=round(float(np.sqrt(np.mean(d**2))), 3),
                R=round(float(np.corrcoef(np.asarray(pred)[ok], np.asarray(truth)[ok])[0, 1]), 3))


def main():
    df = pd.read_csv(SRC, encoding="latin1", low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    df["value1"] = df["value1"].map(parse_val)
    df["error1"] = pd.to_numeric(df["error1"], errors="coerce")
    df["final"] = pd.to_numeric(df["final"], errors="coerce")
    df["MW"] = pd.to_numeric(df["MW"], errors="coerce")
    df["T"] = df["te53p"].map(to_kelvin)
    df["ikey"] = df["mol"].map(inchikey)
    df["proc"] = df["process"].astype(str).str.strip()
    df["unit"] = df["dimension1"].astype(str).str.strip()
    df["pH"] = df["comments"].astype(str).str.extract(r"pH\s*=?\s*([\d.]+)", expand=False).astype(float)
    df["untrusted"] = df["error1"].round(2).isin(UNTRUSTED)
    exd = freesolv()

    # ---- Guthrie's own dG (final, kcal/mol) per molecule ----
    gfin = df[(df["dimension3"].astype(str).str.strip() == "kcal/mol")
              & df["final"].notna() & df["final"].between(-40, 10) & df["ikey"].notna()]
    guthrie = gfin.groupby("ikey")["final"].median()

    # ---- our reconciliations (rule engine) + trust-weighted variant ----
    rows_ok, rows_trust = [], []
    for r in df[df["ikey"].notna() & df["value1"].notna() & ~df["proc"].isin(EXCLUDED)].itertuples(index=False):
        kind, val, _ = convert(r.unit, r.value1, r.proc, r.T, r.MW)
        if kind == "dg" and math.isfinite(val) and -60 < val < 12:
            rows_ok.append((r.ikey, val, r.untrusted))
    obs = pd.DataFrame(rows_ok, columns=["ikey", "dg", "untrusted"])
    ours_all = obs.groupby("ikey")["dg"].median()
    # trusted-only consensus (fall back to all if a molecule has no trusted obs)
    tr = obs[~obs["untrusted"]].groupby("ikey")["dg"].median()
    ours_trust = ours_all.copy(); ours_trust.update(tr)

    # ---- three-way vs FreeSolv on the common overlap ----
    common = sorted(set(guthrie.index) & set(ours_all.index) & set(exd))
    truth = np.array([exd[ik] for ik in common])
    rep = {
        "n_guthrie_final_molecules": int(len(guthrie)),
        "three_way_overlap": len(common),
        "Guthrie_final": metrics([guthrie[ik] for ik in common], truth),
        "ours_rule_engine": metrics([ours_all[ik] for ik in common], truth),
    }
    # trust-weighting effect on the FULL FreeSolv overlap (not just the 3-way)
    fov = sorted(set(ours_all.index) & set(exd))
    tv = np.array([exd[ik] for ik in fov])
    rep["trust_effect_all_overlap"] = {
        "n": len(fov),
        "all_obs": metrics([ours_all[ik] for ik in fov], tv),
        "trusted_only": metrics([ours_trust[ik] for ik in fov], tv),
        "n_molecules_with_untrusted_obs": int(obs.groupby("ikey")["untrusted"].any().sum()),
        "n_untrusted_observations": int(obs["untrusted"].sum()),
    }
    rep["pH_tagged_rows"] = int(df["pH"].notna().sum())
    rep["pH_values"] = {str(k): int(v) for k, v in df["pH"].dropna().value_counts().head(8).items()}

    (OUT / "guthrie_metadata_report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    print("\n=== THREE-WAY vs FreeSolv (overlap = %d molecules) ===" % len(common))
    print(f"  {'Guthrie (his final ΔG)':26s} MAE {rep['Guthrie_final']['MAE']:.3f}  R {rep['Guthrie_final']['R']:.3f}")
    print(f"  {'ours (rule engine)':26s} MAE {rep['ours_rule_engine']['MAE']:.3f}  R {rep['ours_rule_engine']['R']:.3f}")
    te = rep["trust_effect_all_overlap"]
    print(f"\n=== TRUST FLAG effect (all {te['n']} overlap molecules; {te['n_untrusted_observations']} untrusted obs) ===")
    print(f"  all observations   MAE {te['all_obs']['MAE']:.3f}  R {te['all_obs']['R']:.3f}")
    print(f"  trusted-only       MAE {te['trusted_only']['MAE']:.3f}  R {te['trusted_only']['R']:.3f}")


if __name__ == "__main__":
    main()
