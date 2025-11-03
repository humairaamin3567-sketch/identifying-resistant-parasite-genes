import argparse, joblib, os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from src.utils import set_seed

def train(features_csv, out_model, test_size=0.2, seed=42):
    set_seed(seed)
    df = pd.read_csv(features_csv)
    X = df.drop(columns=['sample_id','label']).values
    y = df['label'].values
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size, random_state=seed, stratify=y)
    clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:,1]
    print(classification_report(y_test, preds))
    print('AUC:', roc_auc_score(y_test, probs))
    os.makedirs(os.path.dirname(out_model), exist_ok=True)
    joblib.dump({'model': clf}, out_model)
    print('Saved model to', out_model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    train(args.features, args.out)
