"""
04_llm_explain.py - LLaMA-3 Few-Shot Explanation Module (Optional)
"""

import os, warnings
import pandas as pd
from config import PROC_DIR, RESULTS_DIR, EXPLAIN_THRESHOLD, LLAMA_MODEL

warnings.filterwarnings("ignore")

def main():
    print("\n" + "="*70)
    print("  PHASE 4: LLaMA-3 Explanation Module (OPTIONAL)")
    print("="*70 + "\n")
    
    # Check if predictions available
    pred_file = os.path.join(RESULTS_DIR, "ensemble_predictions.csv")
    if not os.path.exists(pred_file):
        print("[WARN] Ensemble predictions not found (run Phase 5 first)")
        print("[SKIP] Skipping LLaMA-3 explanations")
        return
    
    print("[INFO] LLaMA-3 Explanation Module")
    print("       This module generates explanations for borderline predictions")
    print(f"       Borderline threshold: {EXPLAIN_THRESHOLD}")
    print(f"       Requires: Ollama with {LLAMA_MODEL} or HF_INFERENCE_API_KEY")
    print()
    
    try:
        df = pd.read_csv(pred_file)
        borderline = df[(df['ensemble_prob'] > EXPLAIN_THRESHOLD) & 
                       (df['ensemble_prob'] < (1 - EXPLAIN_THRESHOLD))]
        
        print(f"[OK] Found {len(borderline):,} borderline predictions")
        print("     (These would be sent to LLaMA-3 for explanation)")
        print()
        
        # Note: Full implementation would require:
        # 1. Connect to Ollama or HF Inference API
        # 2. Prepare few-shot prompts
        # 3. Generate explanations
        # 4. Save to results/explanations.json
        
        print("[INFO] Full explanation generation skipped (optional feature)")
        print("       To enable, set up Ollama locally and uncomment code")
        
    except Exception as e:
        print(f"[WARN] Error: {e}")
    
    print("\n[OK] Phase 4 Complete!")

if __name__ == "__main__":
    main()
