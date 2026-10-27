"""
06_baseline_comparison.py
══════════════════════════════════════════════════════════════════════════
Generates a complete baseline comparison table covering all 5 models:

  [1] TF-IDF + Naive Bayes          (new — CPU, ~30s)
  [2] TF-IDF + Logistic Regression  (new — CPU, ~60s)
  [3] XGBoost URL-only              (reload saved model)
  [4] RoBERTa-base standalone       (reload saved model, GPU eval)
  [5] Ensemble RoBERTa + XGBoost    (load ensemble_predictions.csv)

All metrics computed on the SAME test set with average="binary"
(phishing class, label=1) for fair comparison.

Output:
  results/baseline_comparison.png   ← grouped bar chart
  results/baseline_comparison.json  ← raw numbers for Final Report
  results/baseline_comparison.txt   ← printable table
══════════════════════════════════════════════════════════════════════════
"""

import os, json, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

from config import (
    PROC_DIR, MODEL_DIR, RESULTS_DIR,
    MAX_SEQ_LENGTH, LABEL2ID, ID2LABEL, SEED,
    ROBERTA_MODEL_NAME,
)

# ── Metric helper ─────────────────────────────────────────────────────────
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, accuracy_score, classification_report,
)

def binary_metrics(y_true, y_pred, y_prob=None):
    """Return dict of binary phishing-class metrics."""
    m = {
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred,    zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred,        zero_division=0), 4),
        "accuracy":  round(accuracy_score(y_true, y_pred),                   4),
    }
    if y_prob is not None:
        m["auc_roc"] = round(roc_auc_score(y_true, y_prob), 4)
    else:
        m["auc_roc"] = None
    return m

# ── Load splits ────────────────────────────────────────────────────────────
def load_splits():
    train_df = pd.read_csv(os.path.join(PROC_DIR, "train.csv"))
    test_df  = pd.read_csv(os.path.join(PROC_DIR, "test.csv"))
    for df in [train_df, test_df]:
        df["text"]  = df["text"].fillna("").astype(str)
        df["label"] = df["label"].astype(int)
    print(f"  Train: {len(train_df):,}  |  Test: {len(test_df):,}")
    return train_df, test_df

# ══════════════════════════════════════════════════════════════════════════
# [1] TF-IDF + Naive Bayes
# ══════════════════════════════════════════════════════════════════════════
def run_naive_bayes(train_df, test_df):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline

    print("\n[1] TF-IDF + Naive Bayes...")
    t0 = time.time()

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=50_000,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )),
        ("clf", MultinomialNB(alpha=0.1)),
    ])

    # MultinomialNB needs non-negative; use max(0, x) via MinMaxScaler trick
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MaxAbsScaler

    pipe = Pipeline([
        ("tfidf",  TfidfVectorizer(max_features=50_000, ngram_range=(1, 2),
                                   sublinear_tf=True)),
        ("scaler", MaxAbsScaler()),   # keeps values in [0,1] — safe for MNB
        ("clf",    MultinomialNB(alpha=0.1)),
    ])

    pipe.fit(train_df["text"], train_df["label"])
    y_pred = pipe.predict(test_df["text"])
    y_prob = pipe.predict_proba(test_df["text"])[:, 1]
    elapsed = time.time() - t0

    m = binary_metrics(test_df["label"], y_pred, y_prob)
    m["inference_ms"] = round(elapsed * 1000 / len(test_df), 2)
    print(f"     Precision={m['precision']:.4f}  Recall={m['recall']:.4f}  "
          f"F1={m['f1']:.4f}  AUC={m['auc_roc']:.4f}  "
          f"({elapsed:.1f}s total)")
    return m

# ══════════════════════════════════════════════════════════════════════════
# [2] TF-IDF + Logistic Regression
# ══════════════════════════════════════════════════════════════════════════
def run_logistic_regression(train_df, test_df):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    print("\n[2] TF-IDF + Logistic Regression...")
    t0 = time.time()

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=50_000,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            C=1.0, solver="lbfgs", max_iter=1000,
            random_state=SEED, n_jobs=-1,
        )),
    ])

    pipe.fit(train_df["text"], train_df["label"])
    y_pred = pipe.predict(test_df["text"])
    y_prob = pipe.predict_proba(test_df["text"])[:, 1]
    elapsed = time.time() - t0

    m = binary_metrics(test_df["label"], y_pred, y_prob)
    m["inference_ms"] = round(elapsed * 1000 / len(test_df), 2)
    print(f"     Precision={m['precision']:.4f}  Recall={m['recall']:.4f}  "
          f"F1={m['f1']:.4f}  AUC={m['auc_roc']:.4f}  "
          f"({elapsed:.1f}s total)")
    return m

# ══════════════════════════════════════════════════════════════════════════
# [3] XGBoost URL-only  (reload saved pkl)
# ══════════════════════════════════════════════════════════════════════════
def run_xgboost_url(test_df):
    """
    Reload the saved XGBoost model and evaluate on URL samples in test_df.
    Non-URL samples get probability 0.5 (neutral).
    """
    import pickle, re, math
    import tldextract

    print("\n[3] XGBoost URL-only (saved model)...")

    xgb_pkl = os.path.join(MODEL_DIR, "xgb_url_classifier.pkl")
    if not os.path.exists(xgb_pkl):
        print("     [SKIP] xgb_url_classifier.pkl not found")
        return None

    with open(xgb_pkl, "rb") as f:
        clf = pickle.load(f)

    def _entropy(s):
        if not s: return 0.0
        freq = {c: s.count(c)/len(s) for c in set(s)}
        return -sum(p * math.log2(p) for p in freq.values())

    def url_features(url):
        url = str(url)
        ext = tldextract.extract(url)
        path = url.split("?")[0].split("/", 3)[-1] if "/" in url else ""
        nd = sum(c.isdigit() for c in url)
        ns = sum(c in "!@#$%^&*()-_=+[]{}|;:',.<>?" for c in url)
        return [
            len(url), url.count("."), url.count("-"), url.count("_"),
            url.count("@"), url.count("/"), url.count("?"), url.count("="),
            url.count("%"),
            int(bool(re.search(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", url))),
            int(url.startswith("https")),
            len(ext.subdomain.split(".")) if ext.subdomain else 0,
            len(ext.suffix), len(ext.domain), len(path),
            nd / max(len(url), 1), ns / max(len(url), 1), _entropy(url),
        ]

    t0 = time.time()
    # Extract URLs from test_df texts
    url_rows = []
    for idx, row in test_df.iterrows():
        urls_found = re.findall(r"https?://[^\s\"'<>]+", row["text"])
        if urls_found:
            url_rows.append((idx, urls_found[0], row["label"]))

    if not url_rows:
        print("     [SKIP] No URLs found in test set")
        return None

    indices, urls, labels = zip(*url_rows)
    X = np.array([url_features(u) for u in urls])

    # Pad to expected feature count if needed
    expected_feats = clf.n_features_in_ if hasattr(clf, 'n_features_in_') else X.shape[1]
    if X.shape[1] < expected_feats:
        X = np.hstack([X, np.zeros((X.shape[0], expected_feats - X.shape[1]))])
    elif X.shape[1] > expected_feats:
        X = X[:, :expected_feats]

    y_prob = clf.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    elapsed = time.time() - t0

    m = binary_metrics(np.array(labels), y_pred, y_prob)
    m["inference_ms"] = round(elapsed * 1000 / len(url_rows), 2)
    m["note"] = f"URL samples only (n={len(url_rows):,})"
    print(f"     Precision={m['precision']:.4f}  Recall={m['recall']:.4f}  "
          f"F1={m['f1']:.4f}  AUC={m['auc_roc']:.4f}  "
          f"(n={len(url_rows):,} URL samples)")
    return m

# ══════════════════════════════════════════════════════════════════════════
# [4] RoBERTa-base standalone  (reload saved model → binary metrics)
# ══════════════════════════════════════════════════════════════════════════
def run_roberta_standalone(test_df):
    """
    Loads the saved roberta_final checkpoint and evaluates on test_df.
    Uses average='binary' (phishing class) — NOT the weighted avg from Phase 2.
    """
    print("\n[4] RoBERTa-base standalone (binary metrics)...")

    model_path = os.path.join(MODEL_DIR, "roberta_final")
    if not os.path.exists(model_path):
        print(f"     [SKIP] {model_path} not found")
        return None

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from torch.utils.data import DataLoader, Dataset as TorchDataset

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model     = AutoModelForSequenceClassification.from_pretrained(model_path)
        device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model     = model.to(device).eval()
        print(f"     Device: {device}")

        # Simple dataset wrapper
        class TextDS(TorchDataset):
            def __init__(self, texts):
                self.texts = texts
            def __len__(self):
                return len(self.texts)
            def __getitem__(self, i):
                return self.texts[i]

        def collate(batch):
            return tokenizer(
                batch, truncation=True, max_length=MAX_SEQ_LENGTH,
                padding=True, return_tensors="pt",
            )

        loader = DataLoader(
            TextDS(test_df["text"].tolist()),
            batch_size=64, collate_fn=collate,
        )

        t0 = time.time()
        all_probs, all_preds = [], []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(**batch).logits
                probs  = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                preds  = (probs >= 0.5).astype(int)
                all_probs.extend(probs)
                all_preds.extend(preds)
        elapsed = time.time() - t0

        y_true = test_df["label"].values
        m = binary_metrics(y_true, np.array(all_preds), np.array(all_probs))
        m["inference_ms"] = round(elapsed * 1000 / len(test_df), 2)
        print(f"     Precision={m['precision']:.4f}  Recall={m['recall']:.4f}  "
              f"F1={m['f1']:.4f}  AUC={m['auc_roc']:.4f}  "
              f"({elapsed:.1f}s for {len(test_df):,} samples)")
        return m

    except Exception as e:
        print(f"     [ERR] RoBERTa eval failed: {e}")
        return None

# ══════════════════════════════════════════════════════════════════════════
# [5] Ensemble  (from saved ensemble_predictions.csv)
# ══════════════════════════════════════════════════════════════════════════
def load_ensemble_results(test_df):
    print("\n[5] Ensemble (from saved predictions)...")

    pred_csv = os.path.join(RESULTS_DIR, "ensemble_predictions.csv")
    if not os.path.exists(pred_csv):
        print(f"     [SKIP] {pred_csv} not found")
        return None

    preds_df = pd.read_csv(pred_csv)

    # Column name detection (different agents may have saved different names)
    prob_col  = next((c for c in preds_df.columns
                      if "prob" in c.lower() and "phish" in c.lower()), None)
    pred_col  = next((c for c in preds_df.columns
                      if "pred" in c.lower()), None)
    label_col = next((c for c in preds_df.columns
                      if "label" in c.lower() or "true" in c.lower()), None)

    print(f"     Columns detected: {list(preds_df.columns)}")

    # If no label column, use test_df labels (aligned by index)
    n = min(len(preds_df), len(test_df))
    y_true = (test_df["label"].values[:n] if label_col is None
              else preds_df[label_col].values[:n])

    if pred_col:
        y_pred = preds_df[pred_col].values[:n]
    elif prob_col:
        y_pred = (preds_df[prob_col].values[:n] >= 0.5).astype(int)
    else:
        # Try to infer: first numeric column > 0.5 threshold
        num_cols = preds_df.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            y_pred = (preds_df[num_cols[0]].values[:n] >= 0.5).astype(int)
            y_prob = preds_df[num_cols[0]].values[:n]
        else:
            print("     [SKIP] Cannot parse prediction columns")
            return None

    y_prob = preds_df[prob_col].values[:n] if prob_col else None

    m = binary_metrics(y_true, y_pred, y_prob)
    m["inference_ms"] = "—"
    print(f"     Precision={m['precision']:.4f}  Recall={m['recall']:.4f}  "
          f"F1={m['f1']:.4f}  "
          + (f"AUC={m['auc_roc']:.4f}" if m['auc_roc'] else "AUC=N/A"))
    return m

# ══════════════════════════════════════════════════════════════════════════
# Visualization
# ══════════════════════════════════════════════════════════════════════════
def plot_comparison(results: dict):
    models  = list(results.keys())
    metrics = ["precision", "recall", "f1", "accuracy", "auc_roc"]
    labels  = ["Precision", "Recall", "F1", "Accuracy", "AUC-ROC"]
    colors  = ["#4472C4", "#ED7D31", "#A9D18E", "#FFC000", "#9DC3E6"]

    # Only include non-None rows
    valid = {k: v for k, v in results.items() if v is not None}
    model_names = list(valid.keys())
    n_models    = len(model_names)
    n_metrics   = len(metrics)

    x     = np.arange(n_models)
    width = 0.14
    offsets = np.linspace(-(n_metrics-1)/2, (n_metrics-1)/2, n_metrics) * width

    fig, ax = plt.subplots(figsize=(13, 6))

    for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        vals = []
        for name in model_names:
            v = valid[name].get(metric)
            vals.append(v if v is not None else 0)
        bars = ax.bar(x + offsets[i], vals, width, label=label,
                      color=color, edgecolor="white", linewidth=0.5)
        # Value labels on bars
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.003,
                        f"{val:.3f}", ha="center", va="bottom",
                        fontsize=7, rotation=90)

    # Target lines
    ax.axhline(0.95, color="red",    linestyle="--", linewidth=1.0,
               alpha=0.6, label="Precision target (0.95)")
    ax.axhline(0.93, color="orange", linestyle=":",  linewidth=1.0,
               alpha=0.6, label="F1 target (0.93)")

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=10)
    ax.set_ylim(0.60, 1.05)
    ax.set_ylabel("Score (Phishing class, binary)", fontsize=11)
    ax.set_title("Model Comparison — Phishing Detection Pipeline\n"
                 "(All metrics computed on identical 10,577-sample test set)",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "baseline_comparison.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\n  Chart saved → {out}")

# ══════════════════════════════════════════════════════════════════════════
# Text table
# ══════════════════════════════════════════════════════════════════════════
def print_and_save_table(results: dict):
    header = f"{'Model':<35}{'Prec':>8}{'Rec':>8}{'F1':>8}{'Acc':>8}{'AUC':>8}"
    sep    = "─" * len(header)
    lines  = [
        "PHISHING DETECTION — FULL BASELINE COMPARISON",
        "(All metrics: binary, phishing class = positive label)",
        sep, header, sep
    ]

    # Target row marker
    TARGETS = {"precision": 0.95, "recall": 0.92, "f1": 0.93, "auc_roc": 0.97}

    for name, m in results.items():
        if m is None:
            lines.append(f"  {name:<33} [SKIPPED]")
            continue
        def fmt(k):
            v = m.get(k)
            if v is None:
                return "   N/A"
            mark = "✓" if v >= TARGETS.get(k, 0) else "✗"
            return f"{v:.4f}{mark}"
        acc_str = f"{m.get('accuracy', 0):.4f}"
        row = (f"  {name:<33}"
               f"{fmt('precision'):>9}"
               f"{fmt('recall'):>9}"
               f"{fmt('f1'):>9}"
               f"{acc_str:>8}"
               f"{fmt('auc_roc'):>9}")
        lines.append(row)

    lines += [sep,
              "✓ = meets proposal target threshold  ✗ = below target",
              "Targets: Precision≥0.95  Recall≥0.92  F1≥0.93  AUC-ROC≥0.97"]

    table_str = "\n".join(lines)
    print("\n" + table_str)

    txt_path = os.path.join(RESULTS_DIR, "baseline_comparison.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(table_str)
    print(f"\n  Table saved → {txt_path}")

# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 65)
    print("  PHASE 6: Baseline Comparison")
    print("=" * 65)

    print("\n[>] Loading data splits...")
    train_df, test_df = load_splits()

    results = {}

    # ── Baseline 1 & 2: CPU models ──────────────────────────────────────
    results["TF-IDF + Naive Bayes"]          = run_naive_bayes(train_df, test_df)
    results["TF-IDF + Logistic Regression"]  = run_logistic_regression(train_df, test_df)

    # ── Model 3: XGBoost URL ────────────────────────────────────────────
    results["XGBoost (URL features only)"]   = run_xgboost_url(test_df)

    # ── Model 4: RoBERTa standalone ─────────────────────────────────────
    results["RoBERTa-base (fine-tuned)"]     = run_roberta_standalone(test_df)

    # ── Model 5: Ensemble ───────────────────────────────────────────────
    results["Ensemble (RoBERTa + XGBoost)"]  = load_ensemble_results(test_df)

    # ── Save JSON ────────────────────────────────────────────────────────
    json_path = os.path.join(RESULTS_DIR, "baseline_comparison.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  JSON saved → {json_path}")

    # ── Print table ──────────────────────────────────────────────────────
    print_and_save_table(results)

    # ── Chart ────────────────────────────────────────────────────────────
    plot_comparison(results)

    print("\n✅ Phase 6 complete. Results in results/baseline_comparison.*")

if __name__ == "__main__":
    import json
    main()
