# Data instructions

- Place FASTA files (one per sample or multi-sequence FASTA) in `data/raw/`.
- Create `data/metadata.csv` with at least these columns:
  - sample_id : filename (without extension) matching FASTA file
  - label : 0 (sensitive) or 1 (resistant)
- After running `src/features.py`, a features CSV will be saved to `data/processed/features.csv`.
