# Targeting Caregiver Support Under Resource Constraints

A case study of peripartum anxiety among mothers of children with disabilities in rural South India, and a demonstrative screening tool derived from it.

## Repository structure

- **`index.html`** — the live scorecard demo (see below for the hosted link). A self-contained, offline-capable implementation of the five-item risk scorecard described in the paper. No build step, no dependencies — open the file directly in a browser or visit the hosted version.
- **`analysis/pipeline.py`** — the complete, reproducible analysis pipeline behind the paper's numbers: data cleaning, iterative imputation, K-Prototypes household profiling, LASSO + bootstrap correlate selection, scorecard construction, and the cross-outcome robustness check.
- **`analysis/requirements.txt`** — Python dependencies for running the pipeline.

## Running the analysis

```
pip install -r analysis/requirements.txt
python analysis/pipeline.py <path_to_survey_data.xlsx> outputs/
```

Household-level survey data is not included in this repository (see Data availability below). The pipeline expects a raw Qualtrics-format export with the same column structure as the source survey.

## Live demo

The scorecard tool is hosted via GitHub Pages at: *(add link here once Pages is enabled)*

## Data availability

This study uses survey data collected by Satya Special School from 416 caregiving households in Puducherry and rural Tamil Nadu, under informed consent and institutional oversight. The underlying household-level data is not published in this repository. Researchers interested in the data should contact the author directly.

## Citation

Vadapalli, P. S. *Targeting Caregiver Support Under Resource Constraints: A Case Study of Peripartum Anxiety Among Mothers of Children with Disabilities in Rural South India.* Clinton Fellowship for Service.
