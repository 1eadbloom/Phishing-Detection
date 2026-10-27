"""
02_finetune_roberta.py - RoBERTa Fine-tuning for Phishing Detection
Complete implementation with data splitting, training, and evaluation
"""

import os, json, warnings

# Fix PyTorch DLL loading issue on Windows
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from config import (
    ROBERTA_MODEL_NAME, MAX_SEQ_LENGTH, TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE,
    GRAD_ACCUM_STEPS, LEARNING_RATE, NUM_EPOCHS, WARMUP_RATIO,
    WEIGHT_DECAY, FP16, SAVE_TOTAL_LIMIT,
    PROC_DIR, MODEL_DIR, RESULTS_DIR, LOGS_DIR,
    LABEL2ID, ID2LABEL, SEED,
)

warnings.filterwarnings("ignore")

def prepare_data():
    """Load and split dataset into train/val/test"""
    print("[>] Loading merged dataset...")
    csv_path = os.path.join(PROC_DIR, "dataset_merged.csv")
    df = pd.read_csv(csv_path)
    df = df[['text', 'label']].dropna()
    
    print(f"[OK] Loaded {len(df):,} samples")
    print(f"     Phishing: {(df['label']==1).sum():,}")
    print(f"     Legitimate: {(df['label']==0).sum():,}")
    
    # Split: 70% train, 15% val, 15% test
    train, temp = train_test_split(
        df, test_size=0.3, stratify=df['label'], random_state=SEED
    )
    val, test = train_test_split(
        temp, test_size=0.5, stratify=temp['label'], random_state=SEED
    )
    
    print(f"\n[OK] Data split:")
    print(f"     Train: {len(train):,} samples")
    print(f"     Val:   {len(val):,} samples")
    print(f"     Test:  {len(test):,} samples")
    
    train.to_csv(os.path.join(PROC_DIR, "train.csv"), index=False)
    val.to_csv(os.path.join(PROC_DIR, "val.csv"), index=False)
    test.to_csv(os.path.join(PROC_DIR, "test.csv"), index=False)
    
    return train, val, test

def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
    )

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    acc = evaluate.load("accuracy")
    prec = evaluate.load("precision")
    rec = evaluate.load("recall")
    f1 = evaluate.load("f1")
    
    return {
        "accuracy": acc.compute(predictions=predictions, references=labels)["accuracy"],
        "precision": prec.compute(predictions=predictions, references=labels, average="weighted")["precision"],
        "recall": rec.compute(predictions=predictions, references=labels, average="weighted")["recall"],
        "f1": f1.compute(predictions=predictions, references=labels, average="weighted")["f1"],
    }

def main():
    try:
        import torch
        from datasets import Dataset
        import evaluate
        from transformers import (
            AutoTokenizer, AutoModelForSequenceClassification,
            TrainingArguments, Trainer, DataCollatorWithPadding,
            EarlyStoppingCallback, set_seed,
        )
        set_seed(SEED)
        
        print("\n" + "="*70)
        print("  PHASE 2: RoBERTa Fine-tuning")
        print("="*70 + "\n")
        
        # Prepare data
        train_csv = os.path.join(PROC_DIR, "train.csv")
        if not os.path.exists(train_csv):
            train_df, val_df, test_df = prepare_data()
        else:
            print("[OK] Using cached data splits")
            train_df = pd.read_csv(train_csv)
            val_df = pd.read_csv(os.path.join(PROC_DIR, "val.csv"))
            test_df = pd.read_csv(os.path.join(PROC_DIR, "test.csv"))
        
        # Tokenizer
        print(f"\n[>] Loading {ROBERTA_MODEL_NAME} tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL_NAME)
        
        # Datasets
        print("[>] Tokenizing datasets...")
        train_ds = Dataset.from_pandas(train_df).map(
            lambda ex: tokenize_function(ex, tokenizer), batched=True, desc="Train"
        )
        val_ds = Dataset.from_pandas(val_df).map(
            lambda ex: tokenize_function(ex, tokenizer), batched=True, desc="Val"
        )
        test_ds = Dataset.from_pandas(test_df).map(
            lambda ex: tokenize_function(ex, tokenizer), batched=True, desc="Test"
        )
        
        train_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        val_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        test_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        
        # Model
        print(f"\n[>] Loading {ROBERTA_MODEL_NAME} model...")
        model = AutoModelForSequenceClassification.from_pretrained(
            ROBERTA_MODEL_NAME,
            num_labels=2,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        print(f"[OK] Device: {device}")
        
        # Training
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
            fp16=FP16 and torch.cuda.is_available(),
            logging_dir=LOGS_DIR,
            logging_steps=100,
            seed=SEED,
            report_to=[],
        )
        
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )
        
        print("\n[>] Starting training (8-10 hours on GPU)...\n")
        train_result = trainer.train()
        
        # Eval on test
        print("\n[>] Evaluating on test set...")
        test_results = trainer.evaluate(test_ds)
        
        print("\n[OK] Test Results:")
        for key, val in test_results.items():
            if key.startswith("eval_"):
                print(f"     {key}: {val:.4f}")
        
        # Save
        model_path = os.path.join(MODEL_DIR, "roberta_final")
        trainer.save_model(model_path)
        print(f"\n[OK] Model saved to {model_path}")
        
        # Training curves
        print("[>] Generating training curves...")
        if hasattr(trainer.state, 'log_history'):
            logs = trainer.state.log_history
            epochs, train_loss, eval_f1 = [], [], []
            
            for log in logs:
                if 'epoch' in log:
                    epochs.append(log['epoch'])
                    if 'loss' in log:
                        train_loss.append(log['loss'])
                    if 'eval_f1' in log:
                        eval_f1.append(log['eval_f1'])
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(epochs[:len(train_loss)], train_loss, label='Training Loss', marker='o')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Training Loss over Epochs')
            ax.legend()
            ax.grid(True)
            
            curve_path = os.path.join(RESULTS_DIR, "training_curves.png")
            plt.savefig(curve_path, dpi=100)
            print(f"[OK] Curves saved to {curve_path}")
            plt.close()
        
        print("\n[OK] Phase 2 Complete!")
    
    except Exception as e:
        print(f"\n[ERR] Phase 2 failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
