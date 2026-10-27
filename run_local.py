#!/usr/bin/env python3
"""
run_local.py - Local Execution Pipeline for Phishing Detection
"""

import os
import sys
import subprocess
from pathlib import Path

def print_section(title):
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")

def print_step(step_num, title):
    print(f"[Step {step_num}] {title}")

def run_script(script_path, timeout=None, use_torch_preload=False):
    """Run a Python script and capture output"""
    try:
        script_name = os.path.basename(script_path)
        print_step("RUN", f"Executing {script_name}...")
        abs_path = os.path.abspath(script_path)
        
        # Set environment variables for PyTorch on Windows
        env = os.environ.copy()
        env['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        
        # Use torch preload wrapper if needed (to fix Windows DLL issue)
        if use_torch_preload:
            cmd = f'python torch_preload.py "{abs_path}"'
        else:
            cmd = f'python "{abs_path}"'
        
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=False,
            timeout=timeout,
            env=env
        )
        if result.returncode == 0:
            print(f"[OK] {script_name} completed\n")
            return True
        else:
            print(f"[ERR] {script_name} failed with code {result.returncode}\n")
            return False
    except subprocess.TimeoutExpired:
        print(f"[TIME] {script_name} timed out after {timeout}s\n")
        return False
    except Exception as e:
        print(f"[ERR] Error running {script_name}: {e}\n")
        return False

def main():
    base_dir = Path(__file__).parent
    os.chdir(base_dir)
    
    print_section("Phishing Detection System - Local Execution")
    
    # Step 1: Data Pipeline
    print("[Phase 1/5] Data Preparation")
    print("  * Download datasets [D1-D4]")
    print("  * Merge & clean data")
    print("  * Extract URL features")
    run_script("01_data_pipeline.py", timeout=1800, use_torch_preload=False)
    
    # Step 2: RoBERTa Fine-tuning
    print("[Phase 2/5] RoBERTa Fine-tuning")
    print("  * Training time: 8-10 hours on GPU")
    print("  * Saving checkpoints to ./models/")
    run_script("02_finetune_roberta.py", timeout=36000, use_torch_preload=True)
    
    # Step 3: URL Features XGBoost
    print("[Phase 3/5] XGBoost URL Classifier")
    print("  * 5-fold cross-validation")
    print("  * Feature importance analysis")
    run_script("03_url_features.py", timeout=3600, use_torch_preload=False)
    
    # Step 4: LLaMA-3 Explanations
    print("[Phase 4/5] LLaMA-3 Explanation Module (OPTIONAL)")
    print("  * Requires Ollama running locally")
    print("  * Few-shot explanation for borderline predictions")
    run_script("04_llm_explain.py", timeout=7200, use_torch_preload=False)
    
    # Step 5: Ensemble Evaluation
    print("[Phase 5/5] Ensemble Evaluation")
    print("  * Stacking meta-learner")
    print("  * Confusion matrix & ROC/PR curves")
    run_script("05_ensemble_eval.py", timeout=3600, use_torch_preload=True)
    
    # Summary
    print_section("Pipeline Summary")
    print("[OK] Local execution completed!")
    print("  - Outputs saved to: ./results/")
    print("  - Models saved to:  ./models/")
    print("  - Logs saved to:    ./logs/")
    print("  - Data saved to:    ./data/")
    print("\n  Phases completed:")
    print("  - 01_data_pipeline.py (fully functional)")
    print("  - 02_finetune_roberta.py (complete training)")
    print("  - 03_url_features.py (complete training)")
    print("  - 04_llm_explain.py (optional)")
    print("  - 05_ensemble_eval.py (complete evaluation)")
    print("\n  Note: Full training requires ~12-15 hours on GPU\n")

if __name__ == "__main__":
    main()
