import argparse, joblib, pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from src.utils import set_seed
def evaluate(model_path, features_csv):
    set_seed(42)
    art = joblib.load(model_path)
    model = art['model']
    df = pd.read_csv(features_csv)
    X = df.drop(columns=['sample_id','label']).values
    y = df['label'].values
    preds = model.predict(X)
    probs = model.predict_proba(X)[:,1]
    print(classification_report(y, preds))
    print('AUC:', roc_auc_score(y, probs))
    print('Confusion matrix:\n', confusion_matrix(y, preds))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--features', required=True)
    args = parser.parse_args()
    evaluate(args.model, args.features)
