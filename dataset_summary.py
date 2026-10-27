import pandas as pd
import os

csv_path = 'data/processed/dataset_merged.csv'
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    print('\n Dataset Summary:')
    print(f'  Total rows: {len(df):,}')
    print(f'  Columns: {list(df.columns)}')
    phishing_count = (df["label"]==1).sum()
    legit_count = (df["label"]==0).sum()
    print(f'  Phishing: {phishing_count:,} ({phishing_count/len(df)*100:.1f}%)')
    print(f'  Legitimate: {legit_count:,} ({legit_count/len(df)*100:.1f}%)')
    print(f'  Sources: {df["source"].unique().tolist()}')
    print(f'  File size: {os.path.getsize(csv_path)/1024/1024:.1f} MB')
    print()
