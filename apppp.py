"""
Gradio dashboard for surrogate-gene-set testing of UCEC transcriptomic signatures.

Run locally:   python app.py
Deploy:        push this folder to a Hugging Face Space (SDK: Gradio).

Data sources (in order of preference)
  1. results/summary.csv and results/surrogate_scores/*.csv
     produced by `surrogate_test.py` -> real surrogate distributions.
  2. Values reported in the paper (built in) -> the Surrogate Explorer then
     shows an APPROXIMATION drawn from the reported mean and SD, and says so.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import surrogate_test as st

HERE = Path(__file__).parent
RESULTS = Path(os.environ.get("UCEC_RESULTS", HERE / "results"))

VERDICT_COLORS = {
    "Beats surrogates": "#2ecc71",
    "Beats surrogates (source retracted)": "#f39c12",
    "Random-like": "#e74c3c",
    "Not testable": "#95a5a6",
}

# --------------------------------------------------------------------------
# Built-in values from the paper (used when no results/ folder is present)
# --------------------------------------------------------------------------
PAPER_SUMMARY = pd.DataFrame({
    "Signature": ["This study (8 transcripts)", "Zhou 2022 (ferroptosis lncRNA)",
                  "Tang & Zhuge (pseudogene)", "Zhou 2018 (lncRNA)", "Mello (lncRNA)",
                  "Anoikis lncRNA", "Ahsen (miRNA)"],
    "Endpoint": ["tumor_normal", "os", "os", "os", "tumor_normal", "os", "lymph_node_met"],
    "Genes_Recovered": ["8/8", "5/7", "8/9", "7/11", "15/17", "0/5", "0/18"],
    "Alias_Recovered": [0, 0, 0, 4, 0, 0, 0],  # Zhou 2018: 3 -> 7 genes after alias lookup
    "Real_Score": [0.9986, 0.6280, 0.5709, 0.5661, 0.6276, np.nan, np.nan],
    "Surr_Mean": [0.6637, 0.5426, 0.5454, 0.5431, 0.6977, np.nan, np.nan],
    "Surr_SD": [0.1139, 0.0353, 0.0372, 0.0350, 0.1256, np.nan, np.nan],
    "p_value": [0.001, 0.012, 0.243, 0.245, 0.645, np.nan, np.nan],
    "z_score": [2.94, 2.42, 0.69, 0.66, -0.56, np.nan, np.nan],
    "Verdict": ["Beats surrogates", "Beats surrogates", "Random-like", "Random-like",
                "Random-like", "Not testable", "Not testable"],
    "Note": ["", "Source article retracted", "", "", "", "", "miRNA signature; no miRNA matrix"],
})

PAPER_CV = pd.DataFrame({
    "Classifier": ["Logistic Regression", "SVM (RBF)", "XGBoost", "Random Forest"],
    "Mean_AUC": [1.0000, 1.0000, 0.9995, 0.9990],
    "SD": [0.0000, 0.0000, 0.0010, 0.0021],
})

PAPER_CORR = pd.DataFrame({
    "Transcript": ["ENSG00000286039", "PGM5P4", "KLHL4", "ASPA", "MEF2C-AS1",
                   "ZNF300P1", "FRMD6-AS2", "MSL3-DT", "Signature score"],
    "WNT4_rho": [0.413, 0.381, 0.378, 0.368, 0.331, 0.317, 0.258, 0.150, -0.431],
    "BUB1_rho": [-0.302, -0.125, -0.130, -0.223, -0.224, -0.195, -0.159, -0.063, 0.217],
})


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def load_summary() -> tuple[pd.DataFrame, bool]:
    """Return (summary, from_results_folder)."""
    path = RESULTS / "summary.csv"
    if path.exists():
        df = pd.read_csv(path)
        for col in st.SUMMARY_COLUMNS:
            if col not in df.columns:
                df[col] = "" if col in ("Note", "Verdict") else np.nan
        df["Note"] = df["Note"].fillna("")
        return df, True
    return PAPER_SUMMARY.copy(), False


def display_verdict(df: pd.DataFrame) -> pd.Series:
    """Verdict label with the retraction flag folded in for colouring."""
    return df.apply(
        lambda r: "Beats surrogates (source retracted)"
        if r["Verdict"] == "Beats surrogates" and "retract" in str(r["Note"]).lower()
        else r["Verdict"], axis=1)


def fmt_p(p: float, n_surr: int = 1000) -> str:
    """'p ≤ 0.001' when p sits at the smallest attainable value, else 'p = 0.012'."""
    if p <= 1.5 / (n_surr + 1):
        return f"p ≤ {1 / (n_surr + 1):.3f}"
    return f"p = {p:.3f}"


def pretty_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["p_value"] = out["p_value"].apply(
        lambda p: "" if pd.isna(p) else ("≤ 0.001" if p <= 0.0011 else f"{p:.3f}"))
    out["z_score"] = out["z_score"].apply(lambda z: "" if pd.isna(z) else f"{z:+.2f}")
    for c in ("Real_Score", "Surr_Mean", "Surr_SD"):
        out[c] = out[c].apply(lambda v: "" if pd.isna(v) else f"{v:.4f}")
    out["Verdict"] = display_verdict(df)
    return out.fillna("")


def load_surrogates(name: str):
    path = RESULTS / "surrogate_scores" / f"{st.slugify(name)}.csv"
    if path.exists():
        return pd.read_csv(path)["score"].to_numpy()
    return None


def load_cv() -> pd.DataFrame:
    p = RESULTS / "cv_results.csv"
    return pd.read_csv(p) if p.exists() else PAPER_CV


def load_corr() -> pd.DataFrame:
    p = RESULTS / "correlations.csv"
    return pd.read_csv(p) if p.exists() else PAPER_CORR


# --------------------------------------------------------------------------
# Tab functions
# --------------------------------------------------------------------------
def overview():
    df, from_results = load_summary()
    shown = display_verdict(df)
    counts = shown.value_counts().reindex(list(VERDICT_COLORS), fill_value=0)
    tested = int((df["Verdict"] != "Not testable").sum())
    md = (
        "| Signatures | Testable | Beat surrogates | Random-like | Not testable |\n"
        "|---|---|---|---|---|\n"
        f"| {len(df)} | {tested} | "
        f"{counts['Beats surrogates'] + counts['Beats surrogates (source retracted)']} | "
        f"{counts['Random-like']} | {counts['Not testable']} |"
    )
    source = ("Showing **your results folder** (`results/summary.csv`)." if from_results
              else "Showing the **values reported in the paper** (no `results/` folder found).")
    fig = px.pie(names=counts.index, values=counts.values, hole=0.45,
                 color=counts.index, color_discrete_map=VERDICT_COLORS,
                 title="Signature verdicts")
    fig.update_traces(textinfo="value+label")
    return source + "\n\n" + md, pretty_table(df), fig


def surrogate_view(sig_name):
    df, _ = load_summary()
    row = df[df["Signature"] == sig_name]
    if row.empty or pd.isna(row.iloc[0]["Real_Score"]):
        return go.Figure(), "This signature was not testable."
    row = row.iloc[0]
    real, mu, sd = row["Real_Score"], row["Surr_Mean"], row["Surr_SD"]

    scores = load_surrogates(sig_name)
    if scores is not None:
        source = f"Real surrogate scores (n = {len(scores)})."
        p_check = (1 + (scores >= real).sum()) / (len(scores) + 1)
        extra = f" | p recomputed from raw scores: {p_check:.3f}"
    else:
        rng = np.random.default_rng(42)
        scores = rng.normal(mu, sd, 1000)
        source = ("APPROXIMATION: drawn from a normal distribution with the reported "
                  "mean and SD. Raw surrogate scores were not found, and real AUC "
                  "distributions are usually skewed.")
        extra = ""

    fig = go.Figure(go.Histogram(x=scores, nbinsx=40,
                                 marker_color="rgba(52,152,219,0.6)"))
    fig.add_vline(x=real, line_dash="dash", line_color="red",
                  annotation_text=f"Real: {real:.4f}", annotation_position="top")
    fig.add_vline(x=float(np.mean(scores)), line_dash="dot", line_color="gray",
                  annotation_text="Surrogate mean", annotation_position="bottom")
    fig.update_layout(title=f"Null distribution: {sig_name}",
                      xaxis_title="Score (AUC or C-index)", yaxis_title="Count",
                      showlegend=False, height=480)
    p_txt = fmt_p(row["p_value"], len(scores))
    text = (f"Real {real:.4f} | Surrogates {mu:.4f} ± {sd:.4f} | {p_txt} | "
            f"z = {row['z_score']:+.2f} | {row['Verdict']}{extra}\n{source}")
    return fig, text


def ceiling_view():
    cv = load_cv()
    fig = px.bar(cv, x="Classifier", y="Mean_AUC", error_y="SD", color="Classifier",
                 title="5-fold cross-validated ROC-AUC (feature selection inside folds)")
    fig.update_yaxes(range=[0.99, 1.001], title="Mean AUC (axis truncated at 0.99)")
    fig.update_layout(showlegend=False)
    return cv, fig


def drift_view():
    df, _ = load_summary()
    rec = df["Genes_Recovered"].astype(str).str.split("/", expand=True).astype(float)
    t = pd.DataFrame({
        "Signature": df["Signature"], "Recovered": rec[0], "Original": rec[1],
        "Via alias": df["Alias_Recovered"].fillna(0).astype(int) if "Alias_Recovered" in df else 0,
    })
    t["Recovery %"] = (100 * t["Recovered"] / t["Original"]).round(0)
    fig = px.bar(t, x="Signature", y="Recovery %", text="Recovery %",
                 title="Share of published genes found in the expression matrix")
    fig.update_yaxes(range=[0, 105])
    fig.update_layout(height=450)
    return fig, t.astype({"Recovered": int, "Original": int})


def biology_view():
    c = load_corr()
    genes = c[c["Transcript"] != "Signature score"]
    fig = go.Figure()
    fig.add_bar(x=genes["Transcript"], y=genes["WNT4_rho"], name="WNT4 ρ", marker_color="#3498db")
    fig.add_bar(x=genes["Transcript"], y=genes["BUB1_rho"], name="BUB1 ρ", marker_color="#e74c3c")
    fig.update_layout(title="Per-transcript Spearman correlations", barmode="group", height=450)
    return fig, c


def run_custom(expr_file, label_file, surv_file, genes_text, endpoint, positive,
               n_surr, seed, match, progress=gr.Progress()):
    if expr_file is None:
        raise gr.Error("Upload an expression CSV first.")
    genes = st.parse_genes(genes_text)
    if len(genes) < 2:
        raise gr.Error("Enter at least two gene symbols (separated by commas, spaces or new lines).")
    n_surr = int(min(max(n_surr, 100), 5000))
    try:
        expr = st.load_expression(expr_file, genes)
        y = st.load_labels(label_file, positive) if (endpoint == "tumor_normal" and label_file) else None
        surv = st.load_survival(surv_file) if (endpoint == "os" and surv_file) else None
        row, surr = st.run_signature(
            expr, "Your signature", genes, endpoint, y=y, survival=surv,
            n_surrogates=n_surr, seed=int(seed), match=match, progress=progress)
    except (ValueError, KeyError) as e:
        raise gr.Error(str(e))
    if len(surr) == 0:
        return go.Figure(), "None of your genes were found in the expression matrix.", None

    fig = go.Figure(go.Histogram(x=surr, nbinsx=40, marker_color="rgba(52,152,219,0.6)"))
    fig.add_vline(x=row["Real_Score"], line_dash="dash", line_color="red",
                  annotation_text=f"Real: {row['Real_Score']:.4f}")
    fig.update_layout(title="Your signature vs surrogates", showlegend=False,
                      xaxis_title="Score (AUC or C-index)", yaxis_title="Count", height=450)
    text = (f"Genes found: {row['Genes_Recovered']} | Real {row['Real_Score']:.4f} | "
            f"Surrogates {row['Surr_Mean']:.4f} ± {row['Surr_SD']:.4f} | "
            f"{fmt_p(row['p_value'], len(surr))} | "
            f"z = {row['z_score']:+.2f} | {row['Verdict']}")
    tmp = Path(tempfile.mkdtemp()) / "surrogate_scores.csv"
    pd.DataFrame({"score": surr}).to_csv(tmp, index=False)
    return fig, text, str(tmp)


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
def build_demo() -> gr.Blocks:
    summary, _ = load_summary()
    testable = summary.dropna(subset=["Real_Score"])["Signature"].tolist()

    with gr.Blocks(title="UCEC Surrogate-Gene-Set Dashboard") as demo:
        gr.Markdown("# Surrogate-gene-set testing of endometrial carcinoma signatures")
        gr.Markdown("Does a published signature beat random gene sets of the same size "
                    "and expression level? Companion to the paper of the same title.")

        with gr.Tabs():
            with gr.TabItem("Overview"):
                ov_md = gr.Markdown()
                ov_table = gr.Dataframe(interactive=False, wrap=True)
                ov_plot = gr.Plot()

            with gr.TabItem("Surrogate Explorer"):
                dd = gr.Dropdown(testable, value=testable[0] if testable else None,
                                 label="Signature")
                ex_plot = gr.Plot()
                ex_text = gr.Textbox(label="Metrics", interactive=False, lines=3)
                dd.change(surrogate_view, dd, [ex_plot, ex_text])

            with gr.TabItem("Ceiling Effect"):
                gr.Markdown("Removing feature-selection leakage did not change tumor-vs-normal "
                            "AUC, because the task is already at ceiling. The y-axis is "
                            "truncated to make the tiny differences visible.")
                ce_table = gr.Dataframe(interactive=False)
                ce_plot = gr.Plot()

            with gr.TabItem("Identifier Drift"):
                gr.Markdown("Published gene lists often no longer match current annotations. "
                            "Genes that could not be found cannot be tested.")
                dr_plot = gr.Plot()
                dr_table = gr.Dataframe(interactive=False)

            with gr.TabItem("WNT4 / BUB1"):
                gr.Markdown("All eight transcripts correlate positively with WNT4 and negatively "
                            "with BUB1. The composite score is a negative sum, so its signs flip. "
                            "Correlations are not adjusted for tumor purity.")
                bi_plot = gr.Plot()
                bi_table = gr.Dataframe(interactive=False)

            with gr.TabItem("Test your own signature"):
                gr.Markdown(
                    "Upload an expression matrix (CSV, log scale, first column = gene or sample ID) "
                    "and an outcome file, paste your genes, and compare against random gene sets. "
                    "Large matrices are slow in the browser; for full runs use `surrogate_test.py`.")
                with gr.Row():
                    f_expr = gr.File(label="Expression CSV", file_types=[".csv"], type="filepath")
                    f_lab = gr.File(label="Labels CSV (tumor/normal)", file_types=[".csv"], type="filepath")
                    f_surv = gr.File(label="Survival CSV (time, event)", file_types=[".csv"], type="filepath")
                genes_in = gr.Textbox(label="Signature genes", lines=3,
                                      placeholder="GENE1, GENE2, GENE3 ...")
                with gr.Row():
                    ep = gr.Radio(["tumor_normal", "os"], value="tumor_normal", label="Endpoint")
                    pos = gr.Textbox(value="tumor", label="Positive class label")
                    nsur = gr.Number(value=1000, label="Surrogates (100–5000)", precision=0)
                    seed = gr.Number(value=42, label="Seed", precision=0)
                    match = gr.Radio(["window", "per_gene"], value="window", label="Matching")
                go_btn = gr.Button("Run test", variant="primary")
                cu_plot = gr.Plot()
                cu_text = gr.Textbox(label="Result", interactive=False, lines=2)
                cu_file = gr.File(label="Download surrogate scores")
                go_btn.click(run_custom,
                             [f_expr, f_lab, f_surv, genes_in, ep, pos, nsur, seed, match],
                             [cu_plot, cu_text, cu_file])

        gr.Markdown("---\n*Results are descriptive. No multiple-testing correction is applied "
                    "across signatures.*")

        def load_all():
            md, table, pie = overview()
            ce_t, ce_f = ceiling_view()
            dr_f, dr_t = drift_view()
            bi_f, bi_t = biology_view()
            ex_f, ex_t = surrogate_view(testable[0]) if testable else (go.Figure(), "")
            return (md, table, pie, ex_f, ex_t, ce_t, ce_f, dr_f, dr_t, bi_f, bi_t)

        demo.load(load_all, None, [ov_md, ov_table, ov_plot, ex_plot, ex_text,
                                   ce_table, ce_plot, dr_plot, dr_table, bi_plot, bi_table])
    return demo


demo = build_demo()

if __name__ == "__main__":
    demo.launch()
