# Identifying Resistant Parasite Genes

Machine learning and bioinformatics pipeline to identify genes associated with drug resistance in parasites (e.g., Plasmodium spp.).
This repository provides data preprocessing, feature extraction from genomic sequences, simple variant handling, ML models, and notebooks for exploration.

## Quickstart

1. Create virtual env and install:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. Prepare data:
   - Put raw sequences in FASTA format under `data/raw/`.
   - Provide a metadata CSV `data/metadata.csv` with columns: `sample_id, label` where label is 0 (sensitive) or 1 (resistant).
3. Preprocess and extract features:
   ```bash
   python src/features.py --fasta-dir data/raw --meta data/metadata.csv --out data/processed/features.csv
   ```
4. Train model:
   ```bash
   python src/train_model.py --features data/processed/features.csv --out experiments/model.joblib
   ```
5. Evaluate:
   ```bash
   python src/evaluate.py --model experiments/model.joblib --features data/processed/features.csv
   ```

## Structure
- `src/` : scripts (feature extraction, training, evaluation, utils)
- `notebooks/` : EDA and modeling notebooks
- `data/` : raw and processed data (do not commit large raw files)
- `experiments/` : model artifacts and reports (gitignored)
- `Dockerfile`, `requirements.txt`, `.github/` for CI

## Notes
- This is a research pipeline. For variant calling or rigorous genomics analysis, integrate established tools (bwa, samtools, GATK) and consult a bioinformatician.
- The feature-engineering here (k-mer counts, GC%) is meant for quick prototyping and ML baseline.
