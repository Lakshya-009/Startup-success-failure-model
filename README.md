# Startup Success Model

Scores how closely a startup resembles **failed** vs **successful** startups (from two public datasets), and finds the most
similar failed / successful startups for a text description. Built to be called by the backend; it has no web code of its own.

## Quick start
```bash
python -m venv .venv && .venv\Scripts\Activate.ps1      # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v                  # 11 tests
```
```python
from startup_model import StartupRiskModel

model = StartupRiskModel.load()                          # load once at server start-up
model.predict({"funding_usd_millions": 2.5, "country": "India", "industry": "edu"})
model.predict_many([...])                                # batch
model.similar_startups("quick grocery delivery app", industry="commerce_logistics", k_each=4)   # for the paid analysis
```
Install into the backend project instead of copying files: `pip install git+https://github.com/<org>/<repo>.git`

## Input contract (`predict`)
| field | type | notes |
|---|---|---|
| `funding_usd_millions` | number or null | total raised, **in millions of USD** (the LLM extractor must convert currencies) |
| `country` | string or null | free text ("USA", "India"); unlisted countries become `Other` |
| `industry` | string or null | one of `fintech, health, edu, mobility_hw, commerce_logistics, software_ai, media_social, other`, or free text (mapped by keywords). `industry_bucket` is accepted as an alias |

Missing values are allowed (use `null`); unknown extra keys are ignored. Bad values never raise; they are ignored and listed in `notes`.

## Output contract
```json
{
  "success_score": 0.013,            // 0-1, null if no usable input at all
  "band": "failure_like",            // failure_like | mixed | success_like | insufficient_data
  "band_label": "closer to failed startups",
  "confidence": "low",               // none | low | medium | moderate   (never "high" on purpose)
  "dataset_success_rate": 0.038,     // share of startups with a comparable score that succeeded in the training data
  "missing_fields": [],
  "out_of_range": true,              // funding below the 10th percentile of training data: weak comparison
  "funding_percentile": 0.016,
  "normalized_input": {"funding_usd_millions": 0.2, "country": "India", "industry": "edu"},
  "notes": ["..."],                  // human-readable caveats, safe to pass to the explainer LLM
  "model_version": "0.1.0"
}
```
**Ask the founder for missing fields before scoring** (each missing field lowers confidence), and show `notes` / `confidence` to the user.
`similar_startups` returns `{"failed": [...], "succeeded": [...]}`, each item with name, industry, country, funding, failure reason / lesson (failures) or exit reason / valuation (successes), and a similarity value.

### Minimal FastAPI wiring (reference)
```python
from fastapi import FastAPI
from startup_model import StartupRiskModel
app, model = FastAPI(), StartupRiskModel.load()

@app.post("/predict")
def predict(features: dict):  return model.predict(features)
```

## How good is it? (details in `reports/report.md`, plots in `reports/evaluation.png`)
Repeated 5-fold CV on 1,134 startups (298 succeeded / 836 failed):

| metric | value |
|---|---|
| **AUC** | **0.83** (±0.03) |
| **Balanced accuracy** | **0.75** |
| Accuracy | 0.74 (always guessing "failed" also scores 0.74, so use AUC / balanced accuracy) |
| Recall of successes / precision | ~0.79 / ~0.50 at the 0.5 threshold |

- Scores below 0.35 were truly failures **93%** of the time; scores above 0.65 were truly successes **61%** of the time (base rate 26%).
- No over/underfitting: train AUC 0.85 vs CV 0.83; the learning curve flattens, so more tuning will not help. Better **data** will.
- All three inputs matter (CV AUC: funding 0.69, industry 0.69, country 0.60, all together 0.83).

## Read this before relying on it
1. `success_score` is a **similarity score from a class-balanced model, not the real-world probability** that a startup succeeds.
2. The data is *notable* companies: unicorns vs. published failures. A typical early-stage startup with little funding is outside the range
   (flagged by `out_of_range`) and will usually score "closer to failed" simply because of its size.
3. "Success" means became a unicorn / exited, not "profitable small business". Part of the country/industry signal reflects which kinds of companies appear in each source list.
4. **Why not 95% accuracy?** Using all rows (including blanks) reaches ~95%, but the two files have very different missing-data patterns, and a model
   that only sees *which cells are empty* already scores ~90%. That is the model recognising the file, not the startup. Only rows with every input observed are used.
5. To improve for real: add a dataset of ordinary startups (survived / stalled / shut down), then retrain.

## Retrain / update
```bash
python -m startup_model.train      # rewrites startup_model/artifacts/*, reports/*
```
Data cleaning (`startup_model/cleaning.py`): re-parses the failures file's funding (it has unit errors), drops duplicate names, treats the 57 companies
listed in both files as failed, relabels defunct/bankrupt unicorns as failed, drops ambiguous "devalued" ones, and harmonises country and industry vocabularies.
If the installed scikit-learn differs from the one used to save the model, `load()` rebuilds it in memory from the bundled data (about a second), so a version mismatch never breaks the backend.

## Layout
```
startup_model/
  model.py       StartupRiskModel (predict, predict_many, similar_startups, load, retrain)
  cleaning.py    data cleaning + input normalisers        similar.py   TF-IDF similar-startup search
  train.py       training + evaluation + report           config.py    thresholds, paths, version
  data/          the two source CSVs                      artifacts/   model.joblib, model_card.json
reports/         report.md, metrics.json, evaluation.png   tests/       unit tests     examples/   sample input
```
