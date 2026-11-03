import gradio as gr
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# DATA (Hardcoded from your final paper results)
# ============================================================
data = {
    "Signature": [
        "This study (8 transcripts)", "Zhou 2022 (ferroptosis lncRNA)",
        "Tang & Zhuge (pseudogene)", "Zhou 2018 (lncRNA)",
        "Mello (lncRNA)", "Anoikis lncRNA", "Ahsen (miRNA)"
    ],
    "Endpoint": ["Tumor vs Normal", "Overall Survival", "Overall Survival",
                 "Overall Survival", "Tumor vs Normal", "Overall Survival", "Lymph Node Met."],
    "Genes_Recovered": ["8/8", "5/7", "8/9", "7/11", "15/17", "0/5", "0/18"],
    "Real_Score": [0.9986, 0.6280, 0.5709, 0.5661, 0.6276, np.nan, np.nan],
    "Surr_Mean": [0.6637, 0.5426, 0.5454, 0.5431, 0.6977, np.nan, np.nan],
    "Surr_SD": [0.1139, 0.0353, 0.0372, 0.0350, 0.1256, np.nan, np.nan],
    "p_value": [0.001, 0.012, 0.243, 0.245, 0.645, np.nan, np.nan],
    "z_score": [2.94, 2.42, 0.69, 0.66, -0.56, np.nan, np.nan],
    "Verdict": ["Beats surrogates", "Beats surrogates (source retracted)",
                "Random-like", "Random-like", "Random-like", "Not testable", "Not testable"]
}
df = pd.DataFrame(data)

cv_data = pd.DataFrame({
    "Classifier": ["Logistic Regression", "SVM (RBF)", "XGBoost", "Random Forest"],
    "Mean_AUC": [1.0000, 1.0000, 0.9995, 0.9990],
    "SD": [0.0000, 0.0000, 0.0010, 0.0021]
})

wnt4_data = pd.DataFrame({
    "Transcript": ["ENSG00000286039", "PGM5P4", "KLHL4", "ASPA", "MEF2C-AS1", "ZNF300P1", "FRMD6-AS2", "MSL3-DT", "Signature Score"],
    "WNT4 rho": [0.413, 0.381, 0.378, 0.368, 0.331, 0.317, 0.258, 0.150, -0.431],
    "BUB1 rho": [-0.302, -0.125, -0.130, -0.223, -0.224, -0.195, -0.159, -0.063, 0.217]
})

# ============================================================
# FUNCTIONS
# ============================================================

def get_overview():
    fig = px.pie(
        df, names="Verdict", hole=0.4,
        color="Verdict",
        color_discrete_map={
            "Beats surrogates": "#2ecc71",
            "Beats surrogates (source retracted)": "#f39c12",
            "Random-like": "#e74c3c",
            "Not testable": "#95a5a6"
        },
        title="Signature Verdicts"
    )
    return df, fig

def plot_surrogate(sig_name):
    testable = df.dropna(subset=["Real_Score"])
    row = testable[testable["Signature"] == sig_name].iloc[0]

    real = row["Real_Score"]
    mu = row["Surr_Mean"]
    sd = row["Surr_SD"]

    # Generate synthetic null distribution
    np.random.seed(42)
    surrogate_dist = np.random.normal(mu, sd, 1000)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=surrogate_dist, nbinsx=40, name="Surrogate Distribution",
        marker_color="rgba(52, 152, 219, 0.6)"
    ))
    fig.add_vline(
        x=real, line_dash="dash", line_color="red",
        annotation_text=f"Real: {real:.4f}", annotation_position="top"
    )
    fig.update_layout(
        title=f"Null Distribution: {sig_name}",
        xaxis_title="Score (AUC or C-index)",
        yaxis_title="Frequency",
        showlegend=False,
        height=500
    )

    metrics_text = f"Real Score: {real:.4f} | Surrogate Mean: {mu:.4f} ± {sd:.4f} | p-value: {row['p_value']:.3f} | z-score: {row['z_score']:+.2f} | Verdict: {row['Verdict']}"
    return fig, metrics_text

def plot_ceiling():
    fig = px.bar(
        cv_data, x="Classifier", y="Mean_AUC", error_y="SD",
        title="5-Fold Cross-Validated ROC-AUC (Selection Inside Folds)",
        color="Classifier"
    )
    fig.update_yaxes(range=[0.99, 1.001])
    return fig

def plot_identifier_drift():
    mapping_data = pd.DataFrame({
        "Signature": ["This study", "Zhou 2022", "Tang & Zhuge", "Zhou 2018", "Mello", "Anoikis", "Ahsen"],
        "Genes Original": [8, 7, 9, 11, 17, 5, 18],
        "Genes Recovered": [8, 5, 8, 7, 15, 0, 0],
        "Recovery Rate": ["100%", "71%", "89%", "64%", "88%", "0%", "0%"]
    })
    fig = px.bar(
        mapping_data, x="Signature", y="Genes Recovered",
        title="Genes Recovered vs Original",
        color="Recovery Rate",
        text="Recovery Rate"
    )
    return fig, mapping_data

def plot_biology():
    fig = go.Figure()
    fig.add_trace(go.Bar(x=wnt4_data["Transcript"], y=wnt4_data["WNT4 rho"], name="WNT4 ρ", marker_color="#3498db"))
    fig.add_trace(go.Bar(x=wnt4_data["Transcript"], y=wnt4_data["BUB1 rho"], name="BUB1 ρ", marker_color="#e74c3c"))

    fig.update_layout(
        title="Per-Transcript Spearman Correlations (n=582)",
        barmode="group",
        height=500
    )
    return fig, wnt4_data

# ============================================================
# GRADIO UI
# ============================================================

with gr.Blocks(title="UCEC Surrogate Dashboard", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧬 Leakage-Controlled Surrogate-Gene-Set Testing")
    gr.Markdown("**Companion to:** *Leakage-Controlled Surrogate-Gene-Set Testing of Published Endometrial Carcinoma Transcriptomic Signatures*")

    with gr.Tabs():
        # TAB 1: OVERVIEW
        with gr.TabItem("Overview"):
            gr.Markdown("### Executive Summary")
            gr.Markdown("| Signatures Tested | Beats Surrogates | Random-like | Not Testable |")
            gr.Markdown("|---|---|---|---|")
            gr.Markdown("| 7 | 2 | 3 | 2 |")

            table_out = gr.Dataframe(df, label="Results Table")
            pie_out = gr.Plot(label="Verdict Distribution")
            demo.load(get_overview, inputs=None, outputs=[table_out, pie_out])

        # TAB 2: SURROGATE EXPLORER
        with gr.TabItem("Surrogate Explorer"):
            gr.Markdown("### Null Distribution Explorer")
            testable_sigs = df.dropna(subset=["Real_Score"])["Signature"].tolist()
            sig_dropdown = gr.Dropdown(choices=testable_sigs, label="Choose a Signature", value=testable_sigs[0])

            plot_out = gr.Plot()
            metrics_out = gr.Textbox(label="Metrics", interactive=False)

            sig_dropdown.change(plot_surrogate, inputs=sig_dropdown, outputs=[plot_out, metrics_out])
            demo.load(plot_surrogate, inputs=sig_dropdown, outputs=[plot_out, metrics_out])

        # TAB 3: CEILING EFFECT
        with gr.TabItem("Ceiling Effect"):
            gr.Markdown("### Leakage & The Ceiling Effect")
            gr.Markdown("**The core methodological finding:** Correcting for feature-selection leakage did not change the ROC-AUC because tumor-normal classification in TCGA-UCEC is already at ceiling.")
            gr.Dataframe(cv_data, label="Cross-Validation Results")
            ceiling_out = gr.Plot()
            demo.load(plot_ceiling, inputs=None, outputs=ceiling_out)

        # TAB 4: IDENTIFIER DRIFT
        with gr.TabItem("Identifier Drift"):
            gr.Markdown("### Identifier Drift")
            gr.Markdown("Gene identifiers frequently change. When a published signature's IDs no longer match current TCGA annotations, replication becomes impossible without manual mapping.")
            drift_out = gr.Plot()
            drift_table = gr.Dataframe()
            demo.load(plot_identifier_drift, inputs=None, outputs=[drift_out, drift_table])

        # TAB 5: BIOLOGY
        with gr.TabItem("WNT4/BUB1 Biology"):
            gr.Markdown("### WNT4 and BUB1 Correlations")
            gr.Markdown("All eight transcripts correlate **positively** with WNT4 and **negatively** with BUB1. However, the composite signature score (negative sum) flips the sign.")
            bio_out = gr.Plot()
            bio_table = gr.Dataframe()
            demo.load(plot_biology, inputs=None, outputs=[bio_out, bio_table])

    gr.Markdown("---")
    gr.Markdown("*Built with Gradio · 2026*")

# Launch
if __name__ == "__main__":
    demo.launch()
