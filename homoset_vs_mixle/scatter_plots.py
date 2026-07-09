#!/usr/bin/env python3
"""Scatter comparisons (NOT committed): Homoset vs mixle, for both the GuthrieSolv
hydration set (labelled by error vs FreeSolv) and the ODT odour-threshold set."""
from __future__ import annotations
import numpy as np, pandas as pd, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
TEAL, OCHRE, GREY = "#0d6b74", "#a8620a", "#8a9698"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.splines.top": False, "axes.spines.right": False} if False else
                    {"font.size": 10})


def mixle_em(y, mol_idx, src_idx, ref_src, n_iter=400):
    """Gaussian partial pooling with per-source bias (ref anchored 0). Returns per-mol posterior."""
    M, S = mol_idx.max() + 1, src_idx.max() + 1
    mu0, tau2 = float(np.median(y)), float(y.var()) + 1e-3
    sig2 = np.full(S, max(y.var() * .5, 1e-2)); b = np.zeros(S)
    Emu = np.full(M, mu0)
    for _ in range(n_iter):
        r = y - b[src_idx]; prec = 1.0 / sig2[src_idx]
        S1 = np.bincount(mol_idx, weights=prec, minlength=M)
        Sr = np.bincount(mol_idx, weights=r * prec, minlength=M)
        Pi = 1.0 / tau2 + S1; Emu = (mu0 / tau2 + Sr) / Pi; Vmu = 1.0 / Pi
        mu0 = float(Emu.mean()); tau2 = max(float(np.mean((Emu - mu0) ** 2 + Vmu)), 1e-3)
        for s in range(S):
            m = src_idx == s
            b[s] = 0.0 if s == ref_src else float(np.mean(y[m] - Emu[mol_idx][m]))
            sig2[s] = max(float(np.mean((y[m] - Emu[mol_idx][m] - b[s]) ** 2) + Vmu[mol_idx][m].mean()), 1e-3)
    return Emu, b, np.sqrt(sig2)


def mae(a, b): return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))
def rmse(a, b): return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))
def r(a, b): return float(np.corrcoef(a, b)[0, 1])


# ======================================================================
# 1. GUTHRIE : Homoset vs mixle, error vs FreeSolv
# ======================================================================
def guthrie():
    ev = pd.read_csv(OUT / "overlap_freesolv_estimates.csv")
    hs, mx, tr = ev["homoset_L0.6_noise25"], ev["mixle_gauss"], ev["expt"]
    e_hs = (hs - tr).abs(); e_mx = (mx - tr).abs()
    lo, hi = min(tr.min(), hs.min(), mx.min()) - 1, max(tr.max(), hs.max(), mx.max()) + 1

    fig, ax = plt.subplots(1, 3, figsize=(15, 5.2))
    for a, pred, col, name, mae_ in [(ax[0], hs, OCHRE, "Homoset (L=0.6)", mae(hs, tr)),
                                     (ax[1], mx, TEAL, "mixle (gaussian)", mae(mx, tr))]:
        a.plot([lo, hi], [lo, hi], "--", color=GREY, lw=1, zorder=1)
        a.scatter(tr, pred, s=16, c=col, alpha=.55, edgecolor="none", zorder=2)
        a.set_xlim(lo, hi); a.set_ylim(lo, hi); a.set_aspect("equal")
        a.set_xlabel("FreeSolv experimental  ΔG$_{hyd}$  (kcal/mol)")
        a.set_ylabel(f"{name} estimate")
        a.set_title(f"{name} vs FreeSolv\nMAE={mae_:.3f}  RMSE={rmse(pred,tr):.3f}  R={r(pred,tr):.3f}  n={len(tr)}",
                    fontsize=10)

    # panel 3: method agreement, coloured by who is closer to FreeSolv
    win = e_hs - e_mx        # >0 => mixle closer (better); <0 => homoset closer
    sc = ax[2].scatter(hs, mx, c=win, cmap="RdYlGn", vmin=-1.5, vmax=1.5, s=18,
                       alpha=.8, edgecolor="none")
    ax[2].plot([lo, hi], [lo, hi], "--", color=GREY, lw=1)
    ax[2].set_xlim(lo, hi); ax[2].set_ylim(lo, hi); ax[2].set_aspect("equal")
    ax[2].set_xlabel("Homoset (L=0.6) estimate"); ax[2].set_ylabel("mixle (gaussian) estimate")
    ax[2].set_title(f"Homoset vs mixle  (agree along diagonal)\n"
                    f"colour = |Homoset err| − |mixle err|;  mixle wins {int((win>0).sum())}/{len(win)}",
                    fontsize=10)
    cb = fig.colorbar(sc, ax=ax[2], fraction=.046, pad=.04)
    cb.set_label("green → mixle closer to FreeSolv", fontsize=8)
    fig.suptitle("GuthrieSolv hydration free energies — reconciliation vs FreeSolv truth "
                 f"(556 molecules)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "guthrie_homoset_vs_mixle.png", dpi=130, bbox_inches="tight")
    print(f"Guthrie: Homoset MAE {mae(hs,tr):.3f} | mixle MAE {mae(mx,tr):.3f} | "
          f"mixle closer on {int((win>0).sum())}/{len(win)} molecules")


# ======================================================================
# 2. ODT : Homoset (logODT_final) vs mixle on the raw per-source values
# ======================================================================
def odt():
    D = Path("/Users/tgg/Downloads/physchem-v36/physchem_v36/data/physchemprops_gcs_cache")
    df = pd.read_csv(D / "odt_combined_homoset_aligned.csv", low_memory=False)
    for c in ["neg_log_ppmv", "logODT_final", "odt_true_outlier"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["neg_log_ppmv", "logODT_final", "smiles", "source"]).copy()
    mols = [Chem.MolFromSmiles(str(s)) for s in df["smiles"]]
    df["MW"] = [Descriptors.MolWt(m) if m else np.nan for m in mols]
    df["ik"] = [Chem.MolToInchiKey(m) if m else None for m in mols]
    df = df.dropna(subset=["MW", "ik"])
    # raw per-source observation in the logODT (-log10 ug/L, 20C) frame that homoset uses
    #   ppmv -> ug/L:  y = -log10(ppmv * MW / 24.04e-3) = neg_log_ppmv - log10(MW) + log10(24040)
    df["y_raw"] = df["neg_log_ppmv"] - np.log10(df["MW"]) + np.log10(24040.0)

    uniq, mol_idx = np.unique(df["ik"].to_numpy(), return_inverse=True)
    suniq, src_idx = np.unique(df["source"].to_numpy(), return_inverse=True)
    ref_src = int(np.where(suniq == "LargerefODT2023")[0][0])   # largest air source = reference
    Emu, b, sig = mixle_em(df["y_raw"].to_numpy(float), mol_idx, src_idx, ref_src)
    mix = pd.Series(Emu, index=uniq)
    # homoset consensus per molecule
    hs = df.groupby("ik")["logODT_final"].median()
    common = hs.index.intersection(mix.index)
    hsv, mxv = hs.loc[common].to_numpy(), mix.loc[common].to_numpy()
    # remove the constant frame offset from the ppmv->ug/L reconstruction (homoset is itself
    # offset-invariant), so the scatter isolates the reconciliation/shrinkage difference.
    frame_offset = float(np.median(mxv - hsv))
    mxv = mxv - frame_offset

    print("\nODT sources & mixle bias vs reference (LargerefODT2023, air):")
    for s, bb, ss in zip(suniq, b, sig):
        print(f"    {s:28s} bias={bb:+.3f}  sigma={ss:.3f}")
    print(f"ODT: {len(common)} molecules; removed frame offset {frame_offset:+.2f} (unit reconstruction);"
          f"  Homoset vs mixle  MAE={mae(hsv,mxv):.3f}  RMSE={rmse(hsv,mxv):.3f}  R={r(hsv,mxv):.3f}")

    lo, hi = min(hsv.min(), mxv.min()) - .5, max(hsv.max(), mxv.max()) + .5
    fig, ax = plt.subplots(1, 2, figsize=(11, 5.2))
    typ = df.groupby("ik")["type"].first().loc[common]
    cmap = {"air": OCHRE, "water_thermo": TEAL, "recovered": "#b23b3b"}
    for t, cc in cmap.items():
        m = (typ == t).to_numpy()
        if m.any():
            ax[0].scatter(hsv[m], mxv[m], s=16, c=cc, alpha=.5, edgecolor="none", label=t)
    ax[0].plot([lo, hi], [lo, hi], "--", color=GREY, lw=1)
    ax[0].set_xlim(lo, hi); ax[0].set_ylim(lo, hi); ax[0].set_aspect("equal")
    ax[0].set_xlabel("Homoset  logODT$_{final}$  (−log₁₀ µg/L)")
    ax[0].set_ylabel("mixle posterior  (same frame)")
    ax[0].set_title(f"ODT: Homoset vs mixle\nMAE={mae(hsv,mxv):.3f}  R={r(hsv,mxv):.3f}  n={len(common)}",
                    fontsize=10)
    ax[0].legend(frameon=False, fontsize=8, loc="upper left")
    # residual vs homoset value
    ax[1].axhline(0, color=GREY, ls="--", lw=1)
    ax[1].scatter(hsv, mxv - hsv, s=14, c=TEAL, alpha=.45, edgecolor="none")
    ax[1].set_xlabel("Homoset  logODT$_{final}$")
    ax[1].set_ylabel("mixle − Homoset")
    ax[1].set_title("Where the two methods diverge\n(shrinkage pulls extremes toward the mean)",
                    fontsize=10)
    fig.suptitle("ODT odour-detection thresholds — Homoset vs mixle reconciliation", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "odt_homoset_vs_mixle.png", dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    guthrie()
    odt()
    print("\nsaved outputs/guthrie_homoset_vs_mixle.png and outputs/odt_homoset_vs_mixle.png")
