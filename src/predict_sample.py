import argparse, joblib, pandas as pd
from Bio import SeqIO
from src.features import extract_features_from_fasta
import os
def predict(model_path, fasta_path, model_features_csv=None):
    art = joblib.load(model_path)
    model = art['model']
    feats = extract_features_from_fasta(fasta_path, k=3)
    # If trained with top-kmers vocabulary, align features to that vocab using model_features_csv
    if model_features_csv is not None:
        df = pd.read_csv(model_features_csv)
        topcols = [c for c in df.columns if c.startswith('k_')]
        row = []
        for c in topcols:
            kmer = c[2:]
            row.append(feats.get(kmer, 0.0))
        import numpy as np
        X = np.array(row).reshape(1,-1)
        prob = model.predict_proba(X)[0,1]
        pred = model.predict(X)[0]
        print('Predicted:', pred, 'prob:', prob)
    else:
        print('Provide model_features_csv used during training for correct feature alignment.')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--fasta', required=True)
    parser.add_argument('--model-features', default=None)
    args = parser.parse_args()
    predict(args.model, args.fasta, args.model_features)
