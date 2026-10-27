#!/usr/bin/env python3
"""
02_finetune_roberta.py - Stable RoBERTa Fine-tuning for Phishing Detection
FIXED VERSION:
- No manual device placement
- Optimized metrics loading
- Faster dataset mapping
- Safer Trainer config
- Better logging & debug visibility
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# -----------------------------
# Config imports
# -----------------------------
from config import (
    ROBERTA_MODEL_NAME, MAX_SEQ_LENGTH,
    TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE,
    GRAD_ACCUM_STEPS, LEARNING_RATE,
    NUM_EPOCHS, WARMUP_RATIO,
    WEIGHT_DECAY, FP16,
    SAVE_TOTAL_LIMIT,
    PROC_DIR, MODEL_DIR, RESULTS_DIR, LOGS_DIR,
    LABEL2ID, ID2LABEL, SEED,
)

# -----------------------------
# Global metric objects (FIXED)
# -----------------------------
import evaluate

acc_metric = evaluate.load("accuracy")
prec_metric = evaluate.load("precision")
rec_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")


# -----------------------------
# Data preparation
# -----------------------------
def prepare_data():
    print("[>] Loading dataset...")

    csv_path = os.path.join(PROC_DIR, "dataset_merged.csv")
    df = pd.read_csv(csv_path)[["text", "label"]].dropna()

    print(f"[OK] Samples: {len(df):,}")
    print(f"     Phishing: {(df['label'] == 1).sum():,}")
    print(f"     Legit:    {(df['label'] == 0).sum():,}")

    train, temp = train_test_split(
        df, test_size=0.3, stratify=df["label"], random_state=SEED
    )
    val, test = train_test_split(
        temp, test_size=0.5, stratify=temp["label"], random_state=SEED
    )

    print("\n[OK] Split:")
    print(f"  Train: {len(train):,}")
    print(f"  Val:   {len(val):,}")
    print(f"  Test:  {len(test):,}")

    train.to_csv(os.path.join(PROC_DIR, "train.csv"), index=False)
    val.to_csv(os.path.join(PROC_DIR, "val.csv"), index=False)
    test.to_csv(os.path.join(PROC_DIR, "test.csv"), index=False)

    return train, val, test


# -----------------------------
# Tokenization
# -----------------------------
def tokenize_function(batch, tokenizer):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
    )


# -----------------------------
# Metrics (FIXED - no reload)
# -----------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    # [Fix 7] average="binary" correctly measures phishing-class performance
    # "weighted" was masking per-class differences (all scores showed identical 0.9715)
    return {
        "accuracy":  acc_metric.compute(predictions=preds,  references=labels)["accuracy"],
        "precision": prec_metric.compute(predictions=preds, references=labels, average="binary")["precision"],
        "recall":    rec_metric.compute(predictions=preds,   references=labels, average="binary")["recall"],
        "f1":        f1_metric.compute(predictions=preds,    references=labels, average="binary")["f1"],
    }


# -----------------------------
# Main
# -----------------------------
def main():
    try:
        import torch
        from datasets import Dataset
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            TrainingArguments,
            Trainer,
            DataCollatorWithPadding,
            EarlyStoppingCallback,
            set_seed,
        )

        set_seed(SEED)

        print("\n" + "=" * 70)
        print("  PHASE 2: RoBERTa Fine-tuning (STABLE VERSION)")
        print("=" * 70 + "\n")

        # -------------------------
        # Data loading
        # -------------------------
        train_csv = os.path.join(PROC_DIR, "train.csv")

        if not os.path.exists(train_csv):
            train_df, val_df, test_df = prepare_data()
        else:
            print("[OK] Using cached splits")
            train_df = pd.read_csv(train_csv)
            val_df = pd.read_csv(os.path.join(PROC_DIR, "val.csv"))
            test_df = pd.read_csv(os.path.join(PROC_DIR, "test.csv"))

        # -------------------------
        # Tokenizer
        # -------------------------
        print(f"\n[>] Loading tokenizer: {ROBERTA_MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL_NAME)

        # -------------------------
        # Dataset conversion
        # -------------------------
        print("[>] Converting datasets...")

        train_ds = Dataset.from_pandas(train_df)
        val_ds = Dataset.from_pandas(val_df)
        test_ds = Dataset.from_pandas(test_df)

        # FAST MAP (FIXED)
        train_ds = train_ds.map(
            lambda x: tokenize_function(x, tokenizer),
            batched=True,
            num_proc=4,
            desc="Tokenizing train",
        )
        val_ds = val_ds.map(
            lambda x: tokenize_function(x, tokenizer),
            batched=True,
            num_proc=4,
            desc="Tokenizing val",
        )
        test_ds = test_ds.map(
            lambda x: tokenize_function(x, tokenizer),
            batched=True,
            num_proc=4,
            desc="Tokenizing test",
        )

        # REMOVE RAW TEXT (IMPORTANT STABILITY FIX)
        train_ds = train_ds.remove_columns(["text"])
        val_ds = val_ds.remove_columns(["text"])
        test_ds = test_ds.remove_columns(["text"])

        train_ds.set_format("torch")
        val_ds.set_format("torch")
        test_ds.set_format("torch")

        # -------------------------
        # Model (NO manual .to(device))
        # -------------------------
        print(f"\n[>] Loading model: {ROBERTA_MODEL_NAME}")

        model = AutoModelForSequenceClassification.from_pretrained(
            ROBERTA_MODEL_NAME,
            num_labels=2,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[OK] Device detected: {device}")
        print("Torch:", torch.__version__)
        print("CUDA:", torch.version.cuda)
        print("Device:", torch.cuda.get_device_name(0))
        print("Capability:", torch.cuda.get_device_capability(0))

        # -------------------------
        # Training args (STABLE)
        # -------------------------
        args = TrainingArguments(
            output_dir=os.path.join(MODEL_DIR, "roberta_phishing"),
            num_train_epochs=NUM_EPOCHS,

            per_device_train_batch_size=TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=EVAL_BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM_STEPS,

            learning_rate=LEARNING_RATE,
            warmup_ratio=WARMUP_RATIO,
            weight_decay=WEIGHT_DECAY,

            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=SAVE_TOTAL_LIMIT,

            load_best_model_at_end=True,
            metric_for_best_model="f1",

            fp16=False,
            bf16=True,
            gradient_checkpointing=False,
            
            logging_dir=LOGS_DIR,
            logging_steps=50,

            seed=SEED,
            report_to=[],

            dataloader_num_workers=2,  # stability boost
        )

        # -------------------------
        # Trainer
        # -------------------------
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=compute_metrics,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=2)
            ],
        )

        print("\n[>] START TRAINING...\n")

        train_result = trainer.train()

        # -------------------------
        # Evaluation
        # -------------------------
        print("\n[>] Final evaluation...")

        test_results = trainer.evaluate(test_ds)

        print("\n[OK] TEST RESULTS:")
        for k, v in test_results.items():
            if k.startswith("eval_"):
                print(f"  {k}: {v:.4f}")

        # -------------------------
        # Save model
        # -------------------------
        model_path = os.path.join(MODEL_DIR, "roberta_final")
        trainer.save_model(model_path)

        print(f"\n[OK] Model saved: {model_path}")

        # -------------------------
        # Save probability scores for ensemble / baseline comparison
        # -------------------------
        print("[>] Saving test-set probability scores...")
        import torch as _torch
        from torch.utils.data import DataLoader as _DL
        model.eval()
        all_probs = []
        loader = _DL(test_ds, batch_size=64)
        with _torch.no_grad():
            for batch in loader:
                batch = {k: v.to(model.device) for k, v in batch.items()
                         if k in ("input_ids", "attention_mask")}
                out = model(**batch)
                p = _torch.softmax(out.logits, dim=-1)[:, 1].cpu().numpy()
                all_probs.extend(p)
        y_test_labels = test_df["label"].values[:len(all_probs)]
        prob_df = pd.DataFrame({
            "prob_phishing":   all_probs,
            "prob_legitimate": [1 - p for p in all_probs],
            "true_label":      y_test_labels,
        })
        import os as _os
        prob_df.to_csv(_os.path.join(RESULTS_DIR, "roberta_probs.csv"), index=False)
        print(f"[OK] Probabilities saved → results/roberta_probs.csv")

        # -------------------------
        # Training curve
        # -------------------------
        print("[>] Plotting training curves...")

        logs = trainer.state.log_history
        loss, steps = [], []

        for log in logs:
            if "loss" in log:
                loss.append(log["loss"])
                steps.append(log.get("step", len(loss)))

        if loss:
            plt.figure()
            plt.plot(steps, loss)
            plt.xlabel("Steps")
            plt.ylabel("Loss")
            plt.title("Training Loss Curve")

            curve_path = os.path.join(RESULTS_DIR, "training_curves.png")
            plt.savefig(curve_path)
            plt.close()

            print(f"[OK] Saved: {curve_path}")

        print("\n[OK] PHASE 2 COMPLETE")

    except Exception as e:
        print(f"\n[ERR] Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()