"""Extract features from FASTA sequences for ML.
Features included:
- k-mer counts (k=3 by default, normalized)
- GC content
- sequence length
"""
import os, argparse, pandas as pd
from Bio import SeqIO
from collections import Counter
import numpy as np
from tqdm import tqdm

def kmer_counts(seq, k=3):
    seq = seq.upper()
    counts = Counter()
    for i in range(len(seq)-k+1):
        kmer = seq[i:i+k]
        if 'N' in kmer: continue
        counts[kmer] += 1
    return counts

def extract_features_from_fasta(fasta_path, k=3):
    # concatenate sequences in fasta (if multiple records)
    seqs = [str(rec.seq) for rec in SeqIO.parse(fasta_path, 'fasta')]
    full = ''.join(seqs).upper()
    total = len(full)
    if total == 0:
        return {}
    kc = kmer_counts(full, k)
    # normalize kmer frequencies
    norm = {kmer: v/ (total - len(kmer) + 1) for kmer,v in kc.items()}
    gc = (full.count('G') + full.count('C')) / total
    return {'seq_len': total, 'gc': gc, **norm}

def build_feature_table(fasta_dir, meta_csv, out_csv, k=3, top_kmers=200):
    meta = pd.read_csv(meta_csv)
    # Collect kmer vocabulary across samples (top N)
    vocab = Counter()
    sample_kmers = {}
    for sid in tqdm(meta['sample_id']):
        fpath = os.path.join(fasta_dir, sid + '.fasta')
        if not os.path.exists(fpath):
            fpath = os.path.join(fasta_dir, sid + '.fa')
        if not os.path.exists(fpath):
            print('Missing', fpath); continue
        feats = extract_features_from_fasta(fpath, k=k)
        sample_kmers[sid] = feats
        # accumulate
        for kk,v in feats.items():
            if kk not in ['seq_len','gc']:
                vocab[kk] += v
    top = [k for k,_ in vocab.most_common(top_kmers)]
    rows = []
    for idx, row in meta.iterrows():
        sid = row['sample_id']
        feats = sample_kmers.get(sid, {})
        r = {'sample_id': sid, 'label': row['label'], 'seq_len': feats.get('seq_len',0), 'gc': feats.get('gc',0)}
        for kmer in top:
            r[f'k_{kmer}'] = feats.get(kmer, 0.0)
        rows.append(r)
    df = pd.DataFrame(rows)
    df.fillna(0, inplace=True)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print('Saved features to', out_csv)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fasta-dir', required=True)
    parser.add_argument('--meta', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--top-kmers', type=int, default=200)
    args = parser.parse_args()
    build_feature_table(args.fasta_dir, args.meta, args.out, k=args.k, top_kmers=args.top_kmers)
