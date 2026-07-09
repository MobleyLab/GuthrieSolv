#!/usr/bin/env python3
"""Inline PNG figures into report.html as base64 data URIs (the artifact CSP blocks
external images). Idempotent: replaces the figures section if already present."""
import base64, re
from pathlib import Path
HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
HTML = HERE / "report.html"

FIGS = [
    ("unit_coverage.png",
     "Disposition of all 53,895 GuthrieSolv measurements: 38,539 (71%) convert to ΔG_hyd "
     "(free-energy · Henry · VP×solubility pairing); only 186 rows are unparsed. The rest are "
     "single-observable VP/solubility (physically un-pairable) or non-hydration. Right: the rule "
     "engine converts 65 (unit,process) combos vs the switchboard's 57."),
    ("unit_calibration_trajectory.png",
     "Active-learning calibration trajectory — FreeSolv MAE (held-out) stays flat while coverage "
     "grows across annealed outer iterations, then the loop self-terminates."),
    ("guthrie_homoset_vs_mixle.png",
     "GuthrieSolv reconciliation vs FreeSolv truth: Homoset and mixle both hug the diagonal; the "
     "third panel colours each molecule green where mixle beats Homoset."),
    ("guthrie_vs_aqsoldb_vp.png",
     "GuthrieSolv's inputs vs independent references: solubility matches AqSolDB tightly (R 0.98); "
     "vapour pressure is noisier (R 0.82) with a boiling-point artifact, fixed by Clausius–Clapeyron."),
    ("dghyd_sources_compared.png",
     "ΔG_hyd sources — coverage vs accuracy: meta37's Henry-derived ΔG covers 5,020 molecules but is "
     "model-based (MAE 0.37 vs FreeSolv); the GuthrieSolv reconciliation covers 2,675 but is ~2× more "
     "accurate (mixle 0.19), with mixle edging Homoset."),
]


def datauri(p: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()


def block() -> str:
    cards = []
    for name, cap in FIGS:
        f = OUT / name
        if not f.exists():
            continue
        cards.append(
            f'<figure style="margin:0 0 22px">'
            f'<img src="{datauri(f)}" alt="{cap}" '
            f'style="width:100%;max-width:100%;border:1px solid var(--hair);border-radius:10px"/>'
            f'<figcaption class="foot" style="margin-top:8px">{cap}</figcaption></figure>')
    return ('<section id="figures">\n<h2>Figures</h2>\n'
            '<p class="sub">Held-out calibration trajectory and the method-vs-truth scatter.</p>\n'
            + "\n".join(cards) + "\n</section>\n")


def main():
    html = HTML.read_text()
    new = block()
    if '<section id="figures">' in html:
        html = re.sub(r'<section id="figures">.*?</section>\n', new, html, flags=re.S)
    else:
        # insert before the Verdict section
        html = html.replace('<section>\n  <h2>Verdict</h2>', new + '\n<section>\n  <h2>Verdict</h2>', 1)
    HTML.write_text(html)
    print(f"embedded {len(FIGS)} figures; report.html now {len(html)//1024} KB")


if __name__ == "__main__":
    main()
