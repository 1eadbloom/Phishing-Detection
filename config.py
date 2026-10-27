"""
config.py - Phishing & Social Engineering Detection Pipeline
Central configuration, dataset URLs, library indices, hyperparameters
"""

import os

# Directory Structure
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
RAW_DIR     = os.path.join(DATA_DIR, "raw")
PROC_DIR    = os.path.join(DATA_DIR, "processed")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
LOGS_DIR    = os.path.join(BASE_DIR, "logs")

for _d in [RAW_DIR, PROC_DIR, MODEL_DIR, RESULTS_DIR, LOGS_DIR]:
    os.makedirs(_d, exist_ok=True)

# Dataset URLs

# [D1] HuggingFace phishing dataset - loaded via datasets API
HF_DATASET_NAME = "ealvaradob/phishing-dataset"

# [D2] SMS Spam Collection (UCI ML Repo)
SMS_SPAM_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases"
    "/00228/smsspamcollection.zip"
)

# [D3] SpamAssassin corpus tarballs
SPAMASSASSIN_URLS = [
    "https://spamassassin.apache.org/old/publiccorpus/20030228_spam.tar.bz2",
    "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham.tar.bz2",
    "https://spamassassin.apache.org/old/publiccorpus/20030228_hard_ham.tar.bz2",
]

# [D4] PhishTank verified phishing URL feed
PHISHTANK_URL = "http://data.phishtank.com/data/online-valid.json.bz2"

# RoBERTa Fine-Tuning Hyperparameters
# [L1] transformers - AutoModelForSequenceClassification
# Use "roberta-base" for Colab T4 (15 GB VRAM)
# Swap to "roberta-large" if A100 is available
ROBERTA_MODEL_NAME  = "roberta-base"
MAX_SEQ_LENGTH      = 256        # tokens
TRAIN_BATCH_SIZE    = 16
EVAL_BATCH_SIZE     = 32
GRAD_ACCUM_STEPS    = 4          # effective batch size = 64
LEARNING_RATE       = 2e-5
NUM_EPOCHS          = 3
WARMUP_RATIO        = 0.06
WEIGHT_DECAY        = 0.01
FP16                = True       # [L5] accelerate mixed-precision on T4/A100
SAVE_TOTAL_LIMIT    = 2          # keep only best 2 checkpoints

# URL XGBoost Classifier
# [L7] xgboost
XGB_PARAMS = {
    "n_estimators":     400,
    "max_depth":        6,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "eval_metric":      "logloss",
    "use_label_encoder": False,
    "random_state":     42,
}

# Ensemble Meta-Classifier
# [L6] scikit-learn - LogisticRegression as stacking meta-learner
ENSEMBLE_C       = 1.0
ENSEMBLE_MAX_ITER = 500

# huihui_ai/qwen3-coder-abliterated Explanation Module
# [L11] langchain  [L12] langchain-community  [L4] peft (optional LoRA)
Qwen3_Coder_MODEL      = "huihui_ai/qwen3-coder-abliterated"          # Ollama model tag
Qwen3_Coder_TEMP       = 0.1               # low temperature for deterministic analysis
Qwen3_Coder_MAX_TOKENS = 512
EXPLAIN_THRESHOLD = 0.35             # borderline zone: 0.35 < prob < 0.65

# Label Mapping
LABEL2ID = {"legitimate": 0, "phishing": 1}
ID2LABEL = {0: "legitimate", 1: "phishing"}

# Random Seeds
SEED = 42
