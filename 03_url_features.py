"""
03_url_features.py - XGBoost URL Feature Classifier with 5-Fold CV
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import pickle
import tldextract
from config import PROC_DIR, MODEL_DIR, RESULTS_DIR, XGB_PARAMS, SEED

warnings.filterwarnings("ignore")

def extract_url_features(url_str):
    """Extract features from URL string"""
    if not url_str or len(str(url_str).strip()) == 0:
        return [0] * 12
    
    urls = str(url_str).split('|')
    features = []
    
    for url in urls[:3]:  # Use first 3 URLs max
        url = url.strip()
        if not url:
            continue
        
        extracted = tldextract.extract(url)
        
        f = [
            len(url),
            url.count('http'),
            url.count('https'),
            url.count('-'),
            url.count('_'),
            url.count('.'),
            url.count('@'),
            len(extracted.domain),
            len(extracted.subdomain),
            1 if extracted.suffix else 0,
            url.count('/'),
            url.count('?'),
        ]
        features.extend(f)
    
    # Pad to fixed length
    while len(features) < 36:
        features.append(0)
    
    return features[:36]

def main():
    print("\n" + "="*70)
    print("  PHASE 3: XGBoost URL Feature Classifier")
    print("="*70 + "\n")
    
    # Load URL dataset
    url_csv = os.path.join(PROC_DIR, "url_dataset.csv")
    if not os.path.exists(url_csv):
        print("[ERR] url_dataset.csv not found. Run Phase 1 first.")
        return
    
    print("[>] Loading URL dataset...")
    df = pd.read_csv(url_csv)
    df = df.dropna(subset=['urls'])
    df = df[df['urls'].str.len() > 0]
    
    print(f"[OK] Loaded {len(df):,} URL samples")
    print(f"     Phishing: {(df['label']==1).sum():,}")
    print(f"     Legitimate: {(df['label']==0).sum():,}")
    
    # Extract features
    print("\n[>] Extracting URL features...")
    X = np.array([extract_url_features(url) for url in df['urls']])
    y = df['label'].values
    
    print(f"[OK] Feature matrix: {X.shape}")
    
    # Normalize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # 5-Fold CV
    print("\n[>] Starting 5-fold cross-validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    fold_scores = []
    fold_auc = []
    
    from sklearn.metrics import roc_auc_score
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"   Fold {fold}/5...", end=" ")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model = xgb.XGBClassifier(**XGB_PARAMS, verbose=0)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        
        score = model.score(X_test, y_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        
        fold_scores.append(score)
        fold_auc.append(auc)
        
        print(f"Acc: {score:.3f}, AUC: {auc:.3f}")
    
    print(f"\n[OK] Cross-Validation Results:")
    print(f"     Mean Accuracy: {np.mean(fold_scores):.4f} +/- {np.std(fold_scores):.4f}")
    print(f"     Mean AUC: {np.mean(fold_auc):.4f} +/- {np.std(fold_auc):.4f}")
    
    # Train final model on all data
    print("\n[>] Training final model on all data...")
    final_model = xgb.XGBClassifier(**XGB_PARAMS, verbose=0)
    final_model.fit(X, y)
    
    # Feature importance
    importance = final_model.feature_importances_
    top_features = np.argsort(importance)[-10:][::-1]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(top_features)), importance[top_features])
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels([f"Feature {i}" for i in top_features])
    ax.set_xlabel("Importance")
    ax.set_title("Top 10 URL Feature Importance")
    plt.tight_layout()
    
    imp_path = os.path.join(RESULTS_DIR, "xgb_feature_importance.png")
    plt.savefig(imp_path, dpi=100)
    print(f"[OK] Feature importance saved to {imp_path}")
    plt.close()
    
    # Save model
    model_path = os.path.join(MODEL_DIR, "xgb_url_classifier.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump((final_model, scaler), f)
    print(f"[OK] Model saved to {model_path}")
    
    print("\n[OK] Phase 3 Complete!")

if __name__ == "__main__":
    main()
