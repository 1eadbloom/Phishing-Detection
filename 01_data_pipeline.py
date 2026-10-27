"""
01_data_pipeline.py
====================================================================
Downloads, parses, and merges all phishing detection datasets into a
single unified CSV ready for training.

Datasets consumed:
  [D1] ealvaradob/phishing-dataset  (HuggingFace, ~15K)
  [D2] SMS Spam Collection          (UCI ML Repo, ~5.5K)
  [D3] SpamAssassin Public Corpus   (Apache, ~6K)
  [D4] PhishTank URL Dataset        (PhishTank.org, up to 50K sample)

Libraries used:
  [L2] datasets   - HuggingFace dataset loader
  [L8] tldextract - URL domain parsing
  [L13] tqdm      - progress display

Output:
  data/processed/dataset_merged.csv     <- main training dataset
  data/processed/url_dataset.csv        <- URL-only subset for [L7] XGBoost
  data/processed/eda_report.txt         <- class balance / source stats
====================================================================
"""

import os, re, io, bz2, email, tarfile, zipfile, warnings, requests
import pandas as pd
import numpy as np
from tqdm import tqdm                    # [L13]
from datasets import load_dataset        # [L2]
import tldextract                        # [L8]
from config import (
    HF_DATASET_NAME, SMS_SPAM_URL, SPAMASSASSIN_URLS,
    PHISHTANK_URL, RAW_DIR, PROC_DIR, SEED
)

warnings.filterwarnings("ignore")
np.random.seed(SEED)

# -- Helpers ---------

def _clean_text(text: str) -> str:
    """Remove HTML tags, decode entities, collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)           # strip HTML
    text = re.sub(r"https?://\S+", " [URL] ", text) # mask URLs (keep token)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:2000]  # cap at 2000 chars

def _extract_urls(text: str) -> list:
    """Return list of all URLs found in text."""
    return re.findall(r"https?://[^\s\"'<>]+", text or "")

# -- [D1] HuggingFace phishing-dataset ----

def load_hf_dataset() -> pd.DataFrame:
    """
    [D1] ealvaradob/phishing-dataset
    Columns: text, label (0=legitimate, 1=phishing)
    Returns: DataFrame with [text, label, source, urls] columns
    """
    print("[>] [D1] Loading ealvaradob/phishing-dataset from HuggingFace Hub...")
    ds = load_dataset(HF_DATASET_NAME, trust_remote_code=True)
    rows = []
    for split_name, split in ds.items():
        for item in tqdm(split, desc=f"  Parsing {split_name}"):
            text = item.get("text", item.get("Email", ""))
            label = int(item.get("label", item.get("Label", 0)))
            rows.append({
                "text": _clean_text(text),
                "label": label,
                "source": "HuggingFace-D1",
                "urls": "|".join(_extract_urls(text)),
            })
    df = pd.DataFrame(rows)
    print(f"[OK] {len(df):,} samples | phishing: {df.label.sum():,} | legitimate: {(df.label==0).sum():,}")
    return df

# -- [D2] SMS Spam Collection ----

def load_sms_spam() -> pd.DataFrame:
    """
    [D2] SMS Spam Collection (UCI ML Repo)
    Tab-separated: label\\ttext (label='ham' or 'spam')
    Maps: spam->1 (phishing-like), ham->0 (legitimate)
    """
    print("[>] [D2] Downloading SMS Spam Collection from UCI...")
    zip_path = os.path.join(RAW_DIR, "sms_spam.zip")
    if not os.path.exists(zip_path):
        r = requests.get(SMS_SPAM_URL, timeout=30)
        with open(zip_path, 'wb') as f:
            f.write(r.content)
    rows = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if 'smsspamcollection' in name.lower():
                with zf.open(name) as f:
                    for line in tqdm(f, desc="  Parsing SMS Spam"):
                        parts = line.decode('utf-8', errors='replace').split('\t')
                        if len(parts) >= 2:
                            label = 1 if parts[0].lower() == 'spam' else 0
                            text = parts[1]
                            rows.append({
                                "text": _clean_text(text),
                                "label": label,
                                "source": "SMS-D2",
                                "urls": "|".join(_extract_urls(text)),
                            })
    df = pd.DataFrame(rows)
    print(f"[OK] {len(df):,} samples | phishing: {df.label.sum():,} | legitimate: {(df.label==0).sum():,}")
    return df

# -- [D3] SpamAssassin corpus ----

def load_spamassassin() -> pd.DataFrame:
    """
    [D3] SpamAssassin public corpus
    Expects tarballs with email files in spam/ and easy_ham/ and hard_ham/
    """
    print("[>] [D3] Downloading SpamAssassin corpus from Apache...")
    rows = []
    for url in SPAMASSASSIN_URLS:
        tar_path = os.path.join(RAW_DIR, url.split('/')[-1])
        if not os.path.exists(tar_path):
            r = requests.get(url, timeout=30)
            with open(tar_path, 'wb') as f:
                f.write(r.content)
        try:
            with tarfile.open(tar_path, 'r:bz2') as tf:
                for member in tqdm(tf.getmembers(), desc=f"  Parsing {url.split('/')[-1]}"):
                    if member.isfile() and not member.name.startswith('.'):
                        f = tf.extractfile(member)
                        try:
                            content = f.read().decode('utf-8', errors='replace')
                            msg = email.message_from_string(content)
                            body = msg.get_payload(decode=True)
                            if body:
                                body = body.decode('utf-8', errors='replace')
                            else:
                                body = str(msg)
                            
                            # Label from path
                            label = 1 if 'spam' in tar_path.lower() else 0
                            
                            rows.append({
                                "text": _clean_text(body),
                                "label": label,
                                "source": "SpamAssassin-D3",
                                "urls": "|".join(_extract_urls(body)),
                            })
                        except Exception:
                            pass
        except Exception as e:
            print(f"  [WARN] Error reading {tar_path}: {e}")
    
    df = pd.DataFrame(rows)
    if len(df) > 0:
        print(f"[OK] {len(df):,} samples | phishing: {df.label.sum():,} | legitimate: {(df.label==0).sum():,}")
    else:
        print("[SKIP] No samples extracted")
    return df

# -- [D4] PhishTank URL dataset ----

def load_phishtank() -> pd.DataFrame:
    """
    [D4] PhishTank verified phishing URLs (online-valid.json.bz2)
    Returns: DataFrame with [text=(URL string), label=1, ...]
    """
    print("[>] [D4] Downloading PhishTank verified phishing URLs...")
    bz2_path = os.path.join(RAW_DIR, "phishtank_urls.json.bz2")
    if not os.path.exists(bz2_path):
        r = requests.get(PHISHTANK_URL, timeout=30)
        with open(bz2_path, 'wb') as f:
            f.write(r.content)
    
    rows = []
    try:
        import json
        with bz2.open(bz2_path, 'rt', encoding='utf-8') as f:
            data = json.load(f)
            items = data if isinstance(data, list) else data.get('results', [])
            for item in tqdm(items[:50000], desc="  Parsing PhishTank"):  # cap at 50k
                url = item.get('url', '')
                if url:
                    rows.append({
                        "text": url,
                        "label": 1,
                        "source": "PhishTank-D4",
                        "urls": url,
                    })
    except Exception as e:
        print(f"  [WARN] Error reading PhishTank: {e}")
    
    df = pd.DataFrame(rows)
    if len(df) > 0:
        print(f"[OK] {len(df):,} samples (all phishing)")
    else:
        print("[SKIP] No URLs extracted")
    return df

# -- Main merge ----

def main():
    print("\n" + "="*70)
    print("  PHASE 1: DATA PIPELINE - Download & Merge")
    print("="*70 + "\n")
    
    dfs = []
    
    # Load all datasets
    try:
        dfs.append(load_hf_dataset())
    except Exception as e:
        print(f"[ERR] [D1] HF dataset failed: {e}\n")
    
    try:
        dfs.append(load_sms_spam())
    except Exception as e:
        print(f"[ERR] [D2] SMS Spam failed: {e}\n")
    
    try:
        dfs.append(load_spamassassin())
    except Exception as e:
        print(f"[ERR] [D3] SpamAssassin failed: {e}\n")
    
    try:
        dfs.append(load_phishtank())
    except Exception as e:
        print(f"[ERR] [D4] PhishTank failed: {e}\n")
    
    # Merge
    if not dfs:
        print("[ERR] No datasets loaded!")
        return
    
    df_merged = pd.concat(dfs, ignore_index=True)
    df_merged = df_merged.drop_duplicates(subset=['text'], keep='first')
    
    print(f"\n[OK] Total merged: {len(df_merged):,} samples")
    print(f"     Phishing: {df_merged.label.sum():,}")
    print(f"     Legitimate: {(df_merged.label==0).sum():,}")
    
    # Save
    merged_path = os.path.join(PROC_DIR, "dataset_merged.csv")
    df_merged.to_csv(merged_path, index=False)
    print(f"[OK] Saved to {merged_path}")
    
    # URL subset
    df_urls = df_merged[df_merged['urls'].str.len() > 0].copy()
    url_path = os.path.join(PROC_DIR, "url_dataset.csv")
    df_urls.to_csv(url_path, index=False)
    print(f"[OK] URL subset: {len(df_urls):,} (saved to {url_path})")

if __name__ == "__main__":
    main()
