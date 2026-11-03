#!/usr/bin/env python3
"""
Surrogate-gene-set testing for transcriptomic signatures.

Question answered: does a gene signature score better than random gene sets
of the same size and similar expression level?

For each signature the script
  1. finds the signature genes in your expression matrix (optionally using an
     alias table for renamed genes),
  2. scores the real signature (tumor-vs-normal AUC, or overall-survival
     concordance index of the mean expression),
  3. draws N expression-matched random gene sets (surrogates) and scores them,
  4. reports the empirical p-value, z-score and a verdict.

Usage (command line)
--------------------
    python surrogate_test.py \
        --expr expr.csv --signatures signatures.csv \
        --labels labels.csv --survival survival.csv --out results

Input files (all CSV, first column = sample or gene ID)
  expr.csv        expression matrix, log-scale (e.g. log2(CPM+1)).
                  Genes x samples or samples x genes (auto-detected).
  signatures.csv  columns: Signature, Endpoint, Genes [, Note]
                  Endpoint is "tumor_normal" or "os".
                  Genes are separated by ; , or spaces.
  labels.csv      sample ID + a "label" column (tumor / normal, or 1 / 0).
  survival.csv    sample ID + "time" and "event" (1 = death) columns.
  aliases.csv     optional, columns "old" and "new" (renamed gene symbols).

Outputs (in --out)
  summary.csv                    one row per signature
  surrogate_scores/<name>.csv    the raw surrogate scores (used by app.py)
  run_config.json                parameters and versions, for reproducibility

Notes on the method
  * p = (1 + #{surrogate >= real}) / (B + 1); the smallest value is 1/(B+1).
  * z = (real - mean(surrogates)) / sd(surrogates).
  * Tumor-vs-normal score: AUC of the gene-set mean, folded as max(AUC, 1-AUC).
  * Survival score: Harrell's C of the gene-set mean as a risk score, folded
    as max(C, 1-C). This equals the C-index of a univariate Cox model, whose
    fitted coefficient can take either sign.
  * Matching "window" (default): surrogate genes come from genes whose mean
    expression lies within +/- window_sd standard deviations (of the
    gene-level means) of the signature's mean expression.
    Matching "per_gene" (stricter): every surrogate gene is drawn from genes
    whose mean expression is close to that of the corresponding signature gene.
  * No multiple-testing correction is applied across signatures. Apply your
    own (e.g. Bonferroni over the number of signatures you test).
"""
from __future__ import annotations

import argparse
import json
import platform
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ENDPOINTS = ("tumor_normal", "os")
SUMMARY_COLUMNS = [
    "Signature", "Endpoint", "Genes_Recovered", "Alias_Recovered",
    "Real_Score", "Surr_Mean", "Surr_SD", "p_value", "z_score",
    "Verdict", "Note",
]


# --------------------------------------------------------------------------
# Loading helpers
# --------------------------------------------------------------------------
def slugify(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(name)).strip("_").lower()


def parse_genes(text: str) -> list[str]:
    """Split a gene list on semicolons, commas, tabs, spaces or newlines."""
    genes = [g.strip() for g in re.split(r"[;,\s]+", str(text)) if g.strip()]
    seen, unique = set(), []
    for g in genes:
        if g not in seen:
            seen.add(g)
            unique.append(g)
    return unique


def load_expression(path, signature_genes=None, orientation="auto") -> pd.DataFrame:
    """Return a samples x genes DataFrame of floats."""
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)

    if orientation == "genes_by_samples":
        df = df.T
    elif orientation == "auto":
        sig = set(signature_genes or [])
        if sig:
            in_rows, in_cols = len(sig & set(df.index)), len(sig & set(df.columns))
            if in_rows > in_cols:  # genes are rows
                df = df.T
        elif df.shape[0] > df.shape[1]:  # many more rows than columns -> genes
            df = df.T

    df = df.apply(pd.to_numeric, errors="coerce")
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]  # keep first copy of a symbol
    df = df.loc[:, df.notna().all(axis=0)]  # drop genes with missing values
    return df


def load_labels(path, positive="tumor") -> pd.Series:
    """Binary Series (1 = positive class) indexed by sample ID."""
    meta = pd.read_csv(path, index_col=0)
    col = meta["label"] if "label" in meta.columns else meta.iloc[:, 0]
    col = col.astype(str).str.strip().str.lower()
    meta_index = meta.index.astype(str)
    if set(col.unique()) <= {"0", "1", "0.0", "1.0"}:
        y = col.astype(float).astype(int)
    else:
        y = (col == str(positive).strip().lower()).astype(int)
    y.index = meta_index
    if y.nunique() < 2:
        raise ValueError(
            f"Labels contain a single class; check that '{positive}' matches "
            f"your label values {sorted(col.unique())[:5]}."
        )
    return y


def load_survival(path) -> pd.DataFrame:
    meta = pd.read_csv(path, index_col=0)
    meta.index = meta.index.astype(str)
    missing = {"time", "event"} - set(meta.columns)
    if missing:
        raise ValueError(f"survival file is missing column(s): {sorted(missing)}")
    meta = meta[["time", "event"]].apply(pd.to_numeric, errors="coerce").dropna()
    meta["event"] = (meta["event"] > 0).astype(int)
    return meta


def load_aliases(path) -> dict[str, str]:
    if not path:
        return {}
    a = pd.read_csv(path)
    return dict(zip(a["old"].astype(str), a["new"].astype(str)))


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def auc_folded(scores: np.ndarray, y: np.ndarray) -> np.ndarray:
    """AUC of every column of `scores` (n_samples x m) vs binary y, folded."""
    scores = np.atleast_2d(scores.T).T if scores.ndim == 1 else scores
    n1 = int(y.sum())
    n0 = len(y) - n1
    ranks = rankdata(scores, axis=0)
    auc = (ranks[y == 1].sum(axis=0) - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    return np.maximum(auc, 1.0 - auc)


class ConcordanceScorer:
    """Harrell's C for many risk scores sharing one survival dataset."""

    def __init__(self, time: np.ndarray, event: np.ndarray):
        time = np.asarray(time, float)
        event = np.asarray(event).astype(bool)
        comparable = (time[:, None] < time[None, :]) & event[:, None]
        self.i, self.j = np.nonzero(comparable)  # i dies before j
        if len(self.i) == 0:
            raise ValueError("No comparable survival pairs (no events?).")

    def folded(self, scores: np.ndarray) -> np.ndarray:
        scores = scores.reshape(len(scores), -1)
        out = np.empty(scores.shape[1])
        n_pairs = len(self.i)
        for k in range(scores.shape[1]):
            d = scores[self.i, k] - scores[self.j, k]  # higher risk should die first
            c = ((d > 0).sum() + 0.5 * (d == 0).sum()) / n_pairs
            out[k] = max(c, 1.0 - c)
        return out


# --------------------------------------------------------------------------
# Surrogate sampling
# --------------------------------------------------------------------------
def draw_surrogates(gene_means, sig_idx, n_surrogates, rng,
                    match="window", window_sd=2.0, per_gene_tol_sd=0.1,
                    min_candidates=25):
    """Return an (n_surrogates x k) integer array of gene column indices."""
    k = len(sig_idx)
    sd = float(gene_means.std())
    excluded = np.zeros(len(gene_means), bool)
    excluded[sig_idx] = True

    if match == "window":
        center = gene_means[sig_idx].mean()
        pool = np.flatnonzero((np.abs(gene_means - center) <= window_sd * sd) & ~excluded)
        if len(pool) < k:
            raise ValueError("Candidate pool is smaller than the signature; widen --window-sd.")
        return np.array([rng.choice(pool, size=k, replace=False)
                         for _ in range(n_surrogates)])

    if match == "per_gene":
        non_sig = np.flatnonzero(~excluded)
        pools = []
        for g in sig_idx:
            dist = np.abs(gene_means[non_sig] - gene_means[g])
            near = non_sig[dist <= per_gene_tol_sd * sd]
            if len(near) < min_candidates:  # widen to the nearest genes
                near = non_sig[np.argsort(dist)[:min_candidates]]
            pools.append(near)
        draws = np.empty((n_surrogates, k), int)
        for b in range(n_surrogates):
            used = set()
            for j, pool in enumerate(pools):
                choice = int(rng.choice(pool))
                tries = 0
                while choice in used and tries < 50:
                    choice = int(rng.choice(pool))
                    tries += 1
                used.add(choice)
                draws[b, j] = choice
        return draws

    raise ValueError("match must be 'window' or 'per_gene'")


# --------------------------------------------------------------------------
# Main test
# --------------------------------------------------------------------------
def run_signature(expr: pd.DataFrame, name: str, genes: list[str], endpoint: str,
                  y: pd.Series | None = None, survival: pd.DataFrame | None = None,
                  n_surrogates: int = 1000, seed: int = 42, match: str = "window",
                  window_sd: float = 2.0, aliases: dict | None = None,
                  alpha: float = 0.05, note: str = "", progress=None):
    """Test one signature. Returns (summary_row dict, surrogate_scores array)."""
    if endpoint not in ENDPOINTS:
        raise ValueError(f"Endpoint must be one of {ENDPOINTS}, got '{endpoint}'.")
    aliases = aliases or {}

    # --- align samples with the outcome data
    if endpoint == "tumor_normal":
        if y is None:
            raise ValueError("A labels file is required for tumor_normal signatures.")
        samples = expr.index.intersection(y.index)
    else:
        if survival is None:
            raise ValueError("A survival file is required for OS signatures.")
        samples = expr.index.intersection(survival.index)
    if len(samples) < 10:
        raise ValueError(f"Only {len(samples)} samples overlap between expression and outcome files.")
    X = expr.loc[samples]

    # --- find signature genes (direct hit first, then alias)
    cols = set(X.columns)
    found, aliased = [], 0
    for g in genes:
        if g in cols:
            found.append(g)
        elif aliases.get(g) in cols:
            found.append(aliases[g])
            aliased += 1
    found = list(dict.fromkeys(found))
    recovered = f"{len(found)}/{len(genes)}"

    row = {
        "Signature": name, "Endpoint": endpoint, "Genes_Recovered": recovered,
        "Alias_Recovered": aliased, "Real_Score": np.nan, "Surr_Mean": np.nan,
        "Surr_SD": np.nan, "p_value": np.nan, "z_score": np.nan,
        "Verdict": "Not testable", "Note": note,
    }
    if len(found) == 0:
        return row, np.array([])

    gene_index = {g: i for i, g in enumerate(X.columns)}
    sig_idx = np.array([gene_index[g] for g in found])
    values = X.to_numpy(float)
    gene_means = values.mean(axis=0)
    rng = np.random.default_rng(seed)

    if endpoint == "tumor_normal":
        yy = y.loc[samples].to_numpy(int)
        score = lambda S: auc_folded(S, yy)  # noqa: E731
    else:
        sv = survival.loc[samples]
        scorer = ConcordanceScorer(sv["time"].to_numpy(), sv["event"].to_numpy())
        score = scorer.folded

    real = float(score(values[:, sig_idx].mean(axis=1, keepdims=True))[0])

    draws = draw_surrogates(gene_means, sig_idx, n_surrogates, rng,
                            match=match, window_sd=window_sd)
    chunk = 100
    surr = np.empty(n_surrogates)
    for start in range(0, n_surrogates, chunk):
        block = draws[start:start + chunk]
        S = values[:, block].mean(axis=2)  # samples x chunk
        surr[start:start + chunk] = score(S)
        if progress is not None:
            progress((start + chunk) / n_surrogates)

    sd = float(surr.std(ddof=1))
    p = (1 + int((surr >= real).sum())) / (n_surrogates + 1)
    z = (real - float(surr.mean())) / sd if sd > 0 else np.nan
    row.update({
        "Real_Score": round(real, 4), "Surr_Mean": round(float(surr.mean()), 4),
        "Surr_SD": round(sd, 4), "p_value": round(p, 4),
        "z_score": round(z, 2),
        "Verdict": "Beats surrogates" if (p < alpha and z > 0) else "Random-like",
    })
    return row, surr


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--expr", required=True, help="expression CSV (log scale)")
    ap.add_argument("--signatures", required=True, help="signatures CSV")
    ap.add_argument("--labels", help="tumor/normal labels CSV")
    ap.add_argument("--survival", help="survival CSV (time, event)")
    ap.add_argument("--aliases", help="optional alias CSV (old,new)")
    ap.add_argument("--out", default="results", help="output folder")
    ap.add_argument("--positive", default="tumor", help="label value treated as positive class")
    ap.add_argument("--orientation", default="auto",
                    choices=["auto", "genes_by_samples", "samples_by_genes"])
    ap.add_argument("--n-surrogates", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--match", default="window", choices=["window", "per_gene"])
    ap.add_argument("--window-sd", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args(argv)

    sigs = pd.read_csv(args.signatures)
    for col in ("Signature", "Endpoint", "Genes"):
        if col not in sigs.columns:
            sys.exit(f"signatures file needs a '{col}' column")
    if "Note" not in sigs.columns:
        sigs["Note"] = ""
    sigs["Note"] = sigs["Note"].fillna("")

    all_genes = [g for text in sigs["Genes"] for g in parse_genes(text)]
    expr = load_expression(args.expr, all_genes, args.orientation)
    y = load_labels(args.labels, args.positive) if args.labels else None
    surv = load_survival(args.survival) if args.survival else None
    aliases = load_aliases(args.aliases)
    print(f"Expression matrix: {expr.shape[0]} samples x {expr.shape[1]} genes")

    out = Path(args.out)
    (out / "surrogate_scores").mkdir(parents=True, exist_ok=True)
    rows = []
    for _, s in sigs.iterrows():
        row, surr = run_signature(
            expr, str(s["Signature"]), parse_genes(s["Genes"]), str(s["Endpoint"]).strip(),
            y=y, survival=surv, n_surrogates=args.n_surrogates, seed=args.seed,
            match=args.match, window_sd=args.window_sd, aliases=aliases,
            alpha=args.alpha, note=str(s["Note"]))
        rows.append(row)
        if len(surr):
            pd.DataFrame({"score": surr}).to_csv(
                out / "surrogate_scores" / f"{slugify(row['Signature'])}.csv", index=False)
        print(f"{row['Signature']:<35} {row['Genes_Recovered']:>7}  "
              f"real={row['Real_Score']}  p={row['p_value']}  z={row['z_score']}  {row['Verdict']}")

    new = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    summary_path = out / "summary.csv"
    if summary_path.exists():  # update rows with the same name, keep the others
        old = pd.read_csv(summary_path)
        old = old[~old["Signature"].isin(new["Signature"])]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(summary_path, index=False)

    (out / "run_config.json").write_text(json.dumps({
        "n_surrogates": args.n_surrogates, "seed": args.seed, "match": args.match,
        "window_sd": args.window_sd, "alpha": args.alpha,
        "n_samples": int(expr.shape[0]), "n_genes": int(expr.shape[1]),
        "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__,
    }, indent=2))
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
