# CSCI 6330 Project — Benign-Only Anomaly Detection vs. RF-IDS

Comparing a benign-only anomaly detector (autoencoder) against a Random
Forest IDS (RF-IDS) on CICIDS2017, at a matched false-positive rate.

## Setup

1. Clone this repo.
2. Download `MachineLearningCSV.zip` from
   http://cicresearch.ca/CICDataset/CIC-IDS-2017/
3. Unzip it and copy these two files into `data/raw/` (create the folder
   if it doesn't exist):
   - `Monday-WorkingHours_pcap_ISCX.csv`
   - `Wednesday-workingHours_pcap_ISCX.csv`

   (The other five days are needed for later milestones — see the project
   proposal. Not required yet.)
4. Install dependencies: `pip install pandas numpy`
5. Open `notebooks/Milestone1_Draft.ipynb` in PyCharm and run the cells
   top to bottom.

`data/` is gitignored — everyone keeps their own local copy of the raw
CSVs and regenerates `data/processed/{train,val,test}.csv` by running the
notebook (the split is deterministic, so everyone's output is identical).

## Project structure

```
src/data_prep.py          shared data-loading and train/val/test split logic
notebooks/                analysis notebooks
data/raw/                 (gitignored) raw CICIDS2017 CSVs — download yourself
data/processed/           (gitignored) generated train/val/test splits
```
