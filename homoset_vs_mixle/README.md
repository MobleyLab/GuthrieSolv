# Homoset vs. mixle — reconciling GuthrieSolv into ΔG_hyd

Turning the heterogeneous GuthrieSolv literature dump (53,895 records, 172 unit
strings, no reconciliation code) into a single trustworthy hydration free energy
per molecule, and comparing two curation philosophies against FreeSolv.

## Contents
| file | what |
|---|---|
| `unit_rules.py` | **declarative dimensional-analysis rule engine** for units (de-wrap → tokenise → net dimensions → classify → convert); adding a unit = one table row. Converts 65 combos, FreeSolv MAE 0.275 |
| `harmonize_rules.py` | runs the rule engine over GuthrieSolv, compares coverage/accuracy vs the switchboard |
| `guthrie_metadata.py` | uses Guthrie's Excel-only columns (`final` ΔG, `error1` trust flag, pH): three-way Guthrie-vs-ours-vs-FreeSolv + trust-flag effect |
| `harmonize_guthrie.py` | original regex switchboard (172 units → Ben-Naim ΔG_hyd, kcal/mol), T-corrected, keyed by InChIKey |
| `calibrate_units_loop.py` | active-learning loop: calibrate long-tail units against anchors, admit via Homoset noise gate, feed back, loop until dry (single pass) |
| `calibrate_units_iterative.py` | outer fixed-point / co-training wrapper: reconcile → re-anchor → anneal gate strict→loose → repeat until dry (19 units, +135 molecules on the bug-fixed base) |
| `pair_vp_solubility.py` | pairs VP×solubility per molecule (the only route for those observables) + reports what stays truly unconvertible (single-observable molecules) |
| `compare_methods.py` | Homoset PS-gate consensus (L sweep) vs mixle hierarchical partial-pooling (EM, ±robust) |
| `validate_vs_freesolv.py` | per-route + per-molecule accuracy gate against FreeSolv |
| `PAPER.tex` | arXiv-style preprint (compile with `pdflatex`/`tectonic`) |
| `REPORT.md` | full technical writeup |
| `report.html` | visual comparison |
| `outputs/` | harmonized observations, per-molecule estimates, metrics JSON |

## Reproduce
```bash
python harmonize_guthrie.py        # -> outputs/guthrie_dg_observations.csv  (12,027 obs / 2,473 mols)
python compare_methods.py          # -> outputs/comparison_report.json + per_molecule_estimates.csv
python validate_vs_freesolv.py     # sanity gate vs FreeSolv
```
Requires `rdkit`, `numpy`, `pandas`, `scipy`, and a local FreeSolv `database.json`.

## Headline (561-molecule FreeSolv overlap, kcal/mol; bug-fixed base)
| method | all MAE | all RMSE | conflict MAE | bias |
|---|---|---|---|---|
| raw median | 0.299 | 1.020 | 0.365 | +0.16 |
| Homoset (L=0.6, any η) | 0.273 | 0.951 | 0.314 | +0.14 |
| mixle (gaussian) | 0.315 | 0.902 | 0.362 | +0.09 |
| **mixle (robust-t)** | **0.249** | **0.873** | **0.276** | **+0.08** |

The **Homoset gate has two parameters** — the noise level η (tolerated RMS/L) and the dimension
value L; a source joins the homogeneous set iff `RMS/L ≤ η`. Homoset is the better transparent
**curator** (anchor L to real experimental noise, L=0.6, and it filters bad sources); mixle is the better
**reconciler** (lowest RMSE/bias, corrects per-source bias, rescues no-reference molecules).
Use both. See `REPORT.md` / `PAPER.tex` for details.

## References
- Homoset PS gate — the "homogeneous set" / proportional-similarity standardization procedure from
  Devos, Patte, Rouault, Laffort & Van Gemert, *Standardized Human Olfactory Thresholds* (IRL Press /
  Oxford Univ. Press, 1990), originally developed in the 1970s–80s to harmonize human odor detection
  thresholds; ported here from `merge_bbb_homoset_v2.py` / `build_delta_hvap_v2_homoset.py`.
- mixle hierarchical partial pooling — [gmboquet/mixle](https://github.com/gmboquet/mixle) (`Normal(Normal(mu0,tau),sigma)`), reproduced in numpy.
- Ground truth — [MobleyLab/FreeSolv](https://github.com/MobleyLab/FreeSolv).
