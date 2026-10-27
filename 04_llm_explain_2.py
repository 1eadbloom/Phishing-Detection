"""
04_llm_explain.py - LLaMA-3 Few-Shot Explanation Module
══════════════════════════════════════════════════════════════════════
[Fix 1] Removed raw `ollama pull llama3` shell command (was SyntaxError)
[Fix 1] Implemented actual Ollama API integration with 50-sample output

Setup (one-time, outside Python):
    1. Install Ollama: https://ollama.ai/
    2. In a terminal run: ollama pull llama3
    3. Then run this script: python 04_llm_explain_2.py
══════════════════════════════════════════════════════════════════════
"""

import os, json, warnings, urllib.request, urllib.error
import pandas as pd
import numpy as np
from config import PROC_DIR, RESULTS_DIR, EXPLAIN_THRESHOLD, Qwen3_Coder_MODEL

warnings.filterwarnings("ignore")

# ── Few-shot prompt (chain-of-thought, 2 examples) ────────────────────────
SYSTEM_PROMPT = (
    "You are a senior cybersecurity analyst specializing in phishing detection. "
    "Analyze the given message for six indicators and respond ONLY in strict JSON "
    "with these keys: urgency_language, authority_impersonation, "
    "suspicious_url_patterns, credential_harvesting_intent (true/false), "
    "pressure_tactics, overall_verdict (PHISHING or LEGITIMATE), "
    "confidence (HIGH/MEDIUM/LOW), explanation (2 sentences)."
)

FEW_SHOT = [
    {"role": "user",
     "content": ("Message: 'Dear Customer, your PayPal account has been limited. "
                 "Click here immediately or it will be suspended: "
                 "http://paypa1-secure.verify.ru/restore'")},
    {"role": "assistant",
     "content": json.dumps({
         "urgency_language": "'immediately', 'will be suspended'",
         "authority_impersonation": "PayPal",
         "suspicious_url_patterns": "Typosquatted domain 'paypa1' (1 vs l); .ru TLD inconsistent with PayPal",
         "credential_harvesting_intent": True,
         "pressure_tactics": "Threat of permanent account suspension",
         "overall_verdict": "PHISHING",
         "confidence": "HIGH",
         "explanation": ("PayPal brand is impersonated using a typosquatted domain. "
                         "The .ru TLD and suspension threat are classic phishing hallmarks.")
     })},
    {"role": "user",
     "content": "Message: 'Hi team, Q3 budget attached. Let me know if questions. — Sarah'"},
    {"role": "assistant",
     "content": json.dumps({
         "urgency_language": "none",
         "authority_impersonation": "none",
         "suspicious_url_patterns": "none",
         "credential_harvesting_intent": False,
         "pressure_tactics": "none",
         "overall_verdict": "LEGITIMATE",
         "confidence": "HIGH",
         "explanation": ("Routine internal communication with no urgency or suspicious links. "
                         "Tone and content are consistent with normal business correspondence.")
     })},
]

# ── Ollama REST API call ──────────────────────────────────────────────────
def call_ollama(message_text: str, model: str = "huihui_ai/qwen3-coder-abliterated") -> dict:
    """
    Calls Ollama local API at http://localhost:11434/api/chat
    Returns parsed JSON verdict or error dict.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += FEW_SHOT
    messages.append({"role": "user",
                     "content": f"Message: '{message_text[:600]}'"})

    payload = json.dumps({
        "model":   model,
        "messages": messages,
        "stream":  False,
        "format":  "json",
        "options": {"temperature": 0.1, "num_predict": 400},
    }).encode()

    try:
        req = urllib.request.Request(
            "http://localhost:11434/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=45) as resp:
            raw = json.loads(resp.read().decode())
            content = raw.get("message", {}).get("content", "")
            # Strip markdown code fences if present
            content = content.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            return json.loads(content)
    except urllib.error.URLError:
        return {"error": "Ollama not running — start with: ollama serve",
                "overall_verdict": "UNAVAILABLE", "confidence": "N/A",
                "explanation": "Ollama service unreachable."}
    except json.JSONDecodeError as e:
        return {"error": f"JSON parse failed: {e}",
                "overall_verdict": "PARSE_ERROR", "confidence": "N/A",
                "explanation": str(content)[:200]}

# ── Check Ollama availability ─────────────────────────────────────────────
def check_ollama(model: str = "huihui_ai/qwen3-coder-abliterated") -> bool:
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags",
                                     method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            available = [m["name"].split(":")[0]
                         for m in data.get("models", [])]
            if model in available:
                print(f"[OK] Ollama running | Model '{model}' ready ✓")
                return True
            else:
                print(f"[WARN] Ollama running but '{model}' not found.")
                print(f"       Available: {available}")
                print(f"       Run: ollama pull {model}")
                return False
    except Exception:
        print("[WARN] Ollama not reachable at localhost:11434")
        print("       Install: https://ollama.ai/")
        print("       Then run: ollama pull Qwen3_Coder  &&  ollama serve")
        return False

# ── Main ──────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 70)
    print("  PHASE 4: Qwen3_Coder Explanation Module")
    print("=" * 70 + "\n")

    pred_file = os.path.join(RESULTS_DIR, "ensemble_predictions.csv")
    test_file = os.path.join(PROC_DIR, "test.csv")

    if not os.path.exists(pred_file):
        print("[WARN] ensemble_predictions.csv not found — run Phase 5 first")
        return

    # ── Load borderline cases ─────────────────────────────────────────
    preds_df = pd.read_csv(pred_file)
    test_df  = pd.read_csv(test_file)

    lo, hi = EXPLAIN_THRESHOLD, 1.0 - EXPLAIN_THRESHOLD
    border_mask = (preds_df["ensemble_prob"] > lo) & (preds_df["ensemble_prob"] < hi)
    border_idx  = border_mask[border_mask].index

    print(f"[OK] Borderline predictions (p in {lo:.0%}–{hi:.0%}): {border_mask.sum():,}")

    # Sample 50 for explanation
    sample_n  = min(50, len(border_idx))
    sample_idx = np.random.choice(border_idx, size=sample_n, replace=False)
    sample_idx.sort()

    texts  = test_df.loc[sample_idx, "text"].tolist()
    labels = test_df.loc[sample_idx, "label"].tolist()
    probs  = preds_df.loc[sample_idx, "ensemble_prob"].tolist()

    print(f"[OK] Selected {sample_n} samples for Qwen3_Coder explanation\n")

    # ── Check Ollama ──────────────────────────────────────────────────
    ollama_ok = check_ollama(Qwen3_Coder_MODEL)

    if not ollama_ok:
        print("\n[INFO] Skipping live generation — saving placeholder report")
        _save_placeholder(texts, labels, probs, sample_idx)
        return

    # ── Generate explanations ─────────────────────────────────────────
    print(f"\n[>] Generating {sample_n} explanations via Ollama ({Qwen3_Coder_MODEL})...")
    results = []
    agree, total_valid = 0, 0

    for i, (text, label, prob) in enumerate(zip(texts, labels, probs), 1):
        print(f"  [{i:02d}/{sample_n}]", end=" ")
        result = call_ollama(text, model=Qwen3_Coder_MODEL)

        llm_verdict = result.get("overall_verdict", "UNKNOWN")
        verdict_map = {"PHISHING": 1, "LEGITIMATE": 0}
        llm_label   = verdict_map.get(llm_verdict, -1)

        if llm_label != -1:
            total_valid += 1
            if llm_label == int(label):
                agree += 1

        print(f"True={'PHISH' if label else 'LEGIT':5s} | "
              f"RoBERTa p={prob:.3f} | "
              f"LLM={llm_verdict:10s} ({result.get('confidence','?')}) | "
              f"{result.get('explanation','')[:60]}...")

        results.append({
            "idx":           int(sample_idx[i-1]),
            "text":          text[:200],
            "true_label":    int(label),
            "ensemble_prob": round(prob, 4),
            "llm_verdict":   llm_verdict,
            "llm_confidence":result.get("confidence", "N/A"),
            "urgency":       result.get("urgency_language", ""),
            "impersonation": result.get("authority_impersonation", ""),
            "url_anomalies": result.get("suspicious_url_patterns", ""),
            "cred_harvest":  result.get("credential_harvesting_intent", ""),
            "explanation":   result.get("explanation", ""),
        })

    # ── Save CSV ──────────────────────────────────────────────────────
    out_df = pd.DataFrame(results)
    csv_path = os.path.join(RESULTS_DIR, "llm_explanations.csv")
    out_df.to_csv(csv_path, index=False)
    print(f"\n[OK] Explanations saved → {csv_path}")

    # ── Save human-readable sample ────────────────────────────────────
    _save_sample_report(results)

    # ── Agreement stats ───────────────────────────────────────────────
    if total_valid > 0:
        acc = agree / total_valid
        print(f"\n[OK] LLM–Ground Truth Agreement: {acc:.1%} "
              f"({agree}/{total_valid} valid verdicts)")

    print("\n[OK] Phase 4 Complete!")


def _save_sample_report(results: list):
    path = os.path.join(RESULTS_DIR, "llm_explanation_sample.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("Qwen3_Coder Phishing Explanation Module — Sample Output\n")
        f.write("=" * 65 + "\n\n")
        for r in results[:5]:
            f.write(f"[Sample idx={r['idx']}]\n")
            f.write(f"  Text        : {r['text'][:120]}...\n")
            f.write(f"  True label  : {'PHISHING' if r['true_label'] else 'LEGITIMATE'}\n")
            f.write(f"  Ensemble p  : {r['ensemble_prob']:.4f}\n")
            f.write(f"  LLM verdict : {r['llm_verdict']} ({r['llm_confidence']})\n")
            f.write(f"  Urgency     : {r['urgency']}\n")
            f.write(f"  Impersonate : {r['impersonation']}\n")
            f.write(f"  Explanation : {r['explanation']}\n")
            f.write("-" * 65 + "\n\n")
    print(f"[OK] Sample report → {path}")


def _save_placeholder(texts, labels, probs, idx_arr):
    """Save a placeholder CSV so downstream scripts don't fail."""
    rows = [{"idx": int(idx_arr[i]), "text": t[:200],
             "true_label": int(l), "ensemble_prob": round(p, 4),
             "llm_verdict": "UNAVAILABLE", "llm_confidence": "N/A",
             "explanation": "Ollama not running — start ollama serve"}
            for i, (t, l, p) in enumerate(zip(texts, labels, probs))]
    path = os.path.join(RESULTS_DIR, "llm_explanations.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"[OK] Placeholder saved → {path}")
    print("     Re-run after: ollama pull Qwen3_Coder  &&  ollama serve")


if __name__ == "__main__":
    main()
