# Surrogate-gene-set testing for UCEC transcriptomic signatures

Companion code to the paper *Leakage-Controlled Surrogate-Gene-Set Testing of
Published Endometrial Carcinoma Transcriptomic Signatures*.

It answers one question: **does a gene signature score better than random gene
sets of the same size and similar expression level?**

## What is in this folder

| File | Purpose |
|---|---|
| `surrogate_test.py` | The method. Command-line tool and importable module. |
| `app.py` | Gradio dashboard (overview, null-distribution explorer, test-your-own-signature). |
| `make_demo_data.py` | Makes a small **synthetic** dataset to try everything without TCGA. |
| `results/` | Output folder. The dashboard reads from here. |
| `requirements.txt` | Python packages. |

## Quick start (2 minutes, no TCGA needed)

```bash
pip install -r requirements.txt
python make_demo_data.py
python surrogate_test.py --expr demo/expr.csv --signatures demo/signatures.csv \
    --labels demo/labels.csv --survival demo/survival.csv --out results
python app.py            # open http://127.0.0.1:7860
```

The demo data are random numbers. Do not draw biological conclusions from them.
If `results/` is empty the dashboard shows the values reported in the paper
instead (and says so).

## Run it on your own data

1. **Expression matrix** `expr.csv`: log-scale values (e.g. log2(CPM+1)), first
   column = gene or sample ID. Genes × samples or samples × genes (auto-detected).
   Filter low-count genes first.
2. **Labels** `labels.csv`: sample ID plus a `label` column (`tumor`/`normal` or `1`/`0`).
3. **Survival** `survival.csv` (only for OS signatures): sample ID plus `time` and `event` (1 = death).
4. **Signatures** `signatures.csv`:

   | Signature | Endpoint | Genes | Note |
   |---|---|---|---|
   | My panel | tumor_normal | ZNF300P1;ASPA;MEF2C-AS1;ENSG00000286039;KLHL4;MSL3-DT;PGM5P4;FRMD6-AS2 | |
   | Published X | os | GENE1;GENE2;GENE3 | source retracted |

   `Endpoint` is `tumor_normal` or `os`. Genes can be separated by `;`, `,` or spaces.
5. *(Optional)* **Aliases** `aliases.csv` with columns `old,new`, for gene symbols that
   were renamed since the signature was published. The summary reports how many
   genes were rescued this way (`Alias_Recovered`).

```bash
python surrogate_test.py --expr expr.csv --signatures signatures.csv \
    --labels labels.csv --survival survival.csv --aliases aliases.csv --out results
```

Useful options: `--n-surrogates 1000`, `--seed 42`, `--match window|per_gene`,
`--window-sd 2.0`, `--alpha 0.05`, `--positive tumor`.

Outputs in `results/`: `summary.csv`, `surrogate_scores/<name>.csv` (raw null
scores), and `run_config.json` (parameters and versions). Re-running updates
rows with the same signature name and keeps the others.

## How to read the results

- **p** = (1 + number of surrogates ≥ real) / (B + 1). With 1,000 surrogates the
  smallest possible value is about 0.001.
- **z** = (real − surrogate mean) / surrogate SD.
- **Verdict**: "Beats surrogates" if p < `--alpha` and z > 0, otherwise
  "Random-like". "Not testable" means none of the genes were found.
- No multiple-testing correction is applied across signatures. Apply your own.
- **Ceiling warning:** when the task is nearly perfectly separable (as
  tumor-vs-normal in TCGA-UCEC is), many random sets score high too, and a
  "beats surrogates" result carries little weight.

## Matching modes

- `window` (default): surrogate genes come from genes whose mean expression is
  within ±`window-sd` standard deviations (of all gene means) of the signature's
  mean expression.
- `per_gene` (stricter): each surrogate gene is drawn from genes with mean
  expression close to that of the corresponding signature gene.

Check that the mode you use matches the description in your Methods section.

## Deploy the dashboard on Hugging Face Spaces

1. Create a new Space, choose SDK **Gradio**.
2. Upload `app.py`, `surrogate_test.py`, `requirements.txt`, and your `results/` folder.
3. Add this header to the top of the Space's `README.md`:

```yaml
---
title: UCEC Surrogate Dashboard
sdk: gradio
app_file: app.py
---
```

Do **not** upload raw TCGA expression files. Share the derived `results/` tables
and cite the GDC project instead.

## Archive for citation (Zenodo)

Link this GitHub repository to Zenodo and publish a release to get a DOI.
Include `results/` (derived tables only) and `run_config.json`.

## Data

TCGA-UCEC RNA-seq data are public from the Genomic Data Commons (GDC). Cite the
project and list the file manifest you used.
