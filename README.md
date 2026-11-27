# CSCI 6330 Project — Benign-Only Anomaly Detection vs. RF-IDS

Comparing a benign-only anomaly detector (autoencoder) against a Random
Forest IDS (RF-IDS) on CICIDS2017, at a matched false-positive rate.

## Setup

1. Clone this repo.
2. Download `MachineLearningCSV.zip` from
   http://cicresearch.ca/CICDataset/CIC-IDS-2017/
3. Unzip it and copy all eight CSV files into `data/raw/` (create the folder
   if it doesn't exist).
4. Install dependencies: `pip install -r requirements.txt`
   (pandas, numpy, scikit-learn, torch, joblib, matplotlib, psutil, jupyter)
5. Open `Milestone2.ipynb` and run cells 1–4. For the final test results,
   run `m2.reanalyze_saved()` or `python -m src.final_figures`.
   Do not run `m2.score_test()` or `m2.save_official()`; the test set was
   already scored once.

`data/` is gitignored — everyone keeps their own local copy of the raw
CSVs and regenerates `data/processed/{train,val,test}.csv` by running the
notebook (the split is deterministic, so everyone's output is identical).

## Project structure

```
src/data_prep.py          shared data-loading and train/val/test split logic
src/m2_experiments.py     Milestone 2 training, validation, and test scoring
notebooks/                analysis notebooks
data/raw/                 (gitignored) raw CICIDS2017 CSVs — download yourself
data/processed/           (gitignored) generated train/val/test splits
```