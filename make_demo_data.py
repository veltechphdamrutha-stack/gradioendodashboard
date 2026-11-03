#!/usr/bin/env python3
"""
Create a small synthetic dataset so you can try the pipeline and the dashboard
without downloading TCGA. The data are random numbers, NOT real patient data;
do not draw any biological conclusion from the demo results.

    python make_demo_data.py            # writes ./demo/
    python surrogate_test.py --expr demo/expr.csv --signatures demo/signatures.csv \
        --labels demo/labels.csv --survival demo/survival.csv --out results
"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
out = Path("demo")
out.mkdir(exist_ok=True)

n_genes, n_tumor, n_normal = 3000, 130, 20
samples = [f"T{i:03d}" for i in range(n_tumor)] + [f"N{i:03d}" for i in range(n_normal)]
genes = [f"GENE{i:04d}" for i in range(n_genes)]

base = rng.normal(5, 2, size=n_genes).clip(0.5, 12)
expr = base[:, None] + rng.normal(0, 0.8, size=(n_genes, len(samples)))

# 300 genes differ between tumor and normal, so tumor-vs-normal is easy
diff_genes = rng.choice(n_genes, 300, replace=False)
expr[np.ix_(diff_genes, np.arange(n_tumor, len(samples)))] += rng.normal(1.5, 0.4, (300, n_normal))

# a real "tumor-vs-normal" signature: 8 strongly different genes
real_sig = list(rng.choice(diff_genes, 8, replace=False))
# a "random-like" signature: 8 genes that do not differ
non_diff = np.setdiff1d(np.arange(n_genes), diff_genes)
null_sig = list(rng.choice(non_diff, 8, replace=False))
# survival signature: 6 genes tied weakly to survival time
os_sig = list(rng.choice(non_diff, 6, replace=False))
risk = expr[os_sig].mean(axis=0)
risk = (risk - risk.mean()) / risk.std()
time = rng.exponential(1000 * np.exp(-0.4 * risk))
event = (rng.random(len(samples)) < 0.6).astype(int)

pd.DataFrame(expr, index=genes, columns=samples).round(4).to_csv(out / "expr.csv")
pd.DataFrame({"label": ["tumor"] * n_tumor + ["normal"] * n_normal}, index=samples).to_csv(out / "labels.csv")
pd.DataFrame({"time": time.round(1), "event": event}, index=samples).to_csv(out / "survival.csv")

g = lambda idx: ";".join(genes[i] for i in idx)  # noqa: E731
pd.DataFrame({
    "Signature": ["Demo real signature", "Demo random-like signature", "Demo OS signature"],
    "Endpoint": ["tumor_normal", "tumor_normal", "os"],
    "Genes": [g(real_sig), g(null_sig), g(os_sig)],
    "Note": ["", "", ""],
}).to_csv(out / "signatures.csv", index=False)
print("Wrote demo/expr.csv, labels.csv, survival.csv, signatures.csv")
