"""
Phase 1: Clean parallel data pipeline for English→Tunisian (Derja) MT.

Reads raw CSV, deduplicates, filters quality, splits into train/val/test.
No instruction wrappers, no T5 paraphrases, no template inflation.
Output: clean_en_tn_{train,val,test}.jsonl
"""
import json
import random
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
random.seed(SEED)

DATA_DIR = Path(__file__).parent


def normalize_arabic(text: str) -> str:
    text = re.sub(r'[إأآا]', 'ا', text)
    text = re.sub(r'[ىي]', 'ي', text)
    text = re.sub(r'[ة]', 'ه', text)
    text = re.sub(r'[ئ]', 'ي', text)
    text = re.sub(r'[ؤ]', 'و', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def is_garbage(text: str) -> bool:
    text = text.strip()
    if len(text) < 2:
        return True
    garbage_patterns = [
        r'^[+\-=\*/_\.\,\;\:\[\]\(\)\#\@\!\?\"\']+$',
        r'^\d+$',
        r'^[a-zA-Z]+$',
        r'ترجم\s+لهنا|ترجم\s+هذا|ترجم\s+ل|ترجم\s+هكا',
    ]
    for pat in garbage_patterns:
        if re.match(pat, text, re.IGNORECASE):
            return True
    return False


def is_likely_instruction(text: str) -> bool:
    instruction_markers = [
        'ترجم', 'اكتب', 'translat', 'write in derja',
        'how do you say', 'what is the word', 'what does',
        'in derja', 'can you give me', 'write in',
    ]
    t = text.lower().strip()
    for m in instruction_markers:
        if t.startswith(m):
            return True
    return False


def length_ratio_filter(en: str, tn: str) -> bool:
    en_words = len(en.split())
    tn_words = len(tn.split())
    if en_words == 0 or tn_words == 0:
        return False
    ratio = tn_words / max(en_words, 1)
    return 0.2 <= ratio <= 5.0


def main():
    print("=" * 60)
    print("Phase 1: Clean English→Tunisian Parallel Data Pipeline")
    print("=" * 60)

    # 1. Load raw data
    raw_path = DATA_DIR / "tunsi_english_data.csv"
    df = pd.read_csv(raw_path)
    df.columns = ['id', 'en', 'tn']
    print(f"\n[1] Loaded raw CSV: {len(df)} rows")

    # 2. Basic cleaning
    df = df.dropna(subset=['en', 'tn'])
    df['en'] = df['en'].str.strip()
    df['tn'] = df['tn'].str.strip()
    print(f"    After dropna + strip: {len(df)} rows")

    # 3. Filter out garbage
    mask_en = df['en'].apply(lambda x: not is_garbage(x))
    mask_tn = df['tn'].apply(lambda x: not is_garbage(x))
    df = df[mask_en & mask_tn]
    print(f"    After garbage filter: {len(df)} rows")

    # 4. Filter out instruction-like prompts
    mask_inst = df['en'].apply(lambda x: not is_likely_instruction(x))
    df = df[mask_inst]
    print(f"    After instruction filter: {len(df)} rows")

    # 5. Length ratio filter
    mask_len = df.apply(lambda r: length_ratio_filter(r['en'], r['tn']), axis=1)
    df = df[mask_len]
    print(f"    After length ratio filter: {len(df)} rows")

    # 6. Deduplicate on English side (keep first)
    before = len(df)
    df = df.drop_duplicates(subset='en', keep='first')
    print(f"    After EN dedup: {before} → {len(df)} rows")

    # 7. Also check for Arabic-side instructions (from bad data)
    mask_tn_clean = df['tn'].apply(lambda x: not is_likely_instruction(x))
    df = df[mask_tn_clean]
    print(f"    After TN instruction filter: {len(df)} rows")

    print(f"\n[2] Final clean dataset: {len(df)} rows")
    print(f"    EN words: mean={df['en'].str.split().str.len().mean():.1f}")
    print(f"    TN chars: mean={df['tn'].str.len().mean():.1f}")

    # 8. Stratified split by EN length bucket
    df['len_bucket'] = pd.cut(
        df['en'].str.split().str.len(),
        bins=[0, 3, 6, 10, 100],
        labels=['short', 'medium', 'long', 'very_long']
    )
    train_val, test = train_test_split(
        df, test_size=250, random_state=SEED,
        stratify=df['len_bucket']
    )
    train, val = train_test_split(
        train_val, test_size=250, random_state=SEED,
        stratify=train_val['len_bucket']
    )

    print(f"\n[3] Splits:")
    print(f"    Train: {len(train)}")
    print(f"    Val:   {len(val)}")
    print(f"    Test:  {len(test)}")

    # 9. Save as clean parallel jsonl
    for split_name, split_df in [('train', train), ('val', val), ('test', test)]:
        out_path = DATA_DIR / f"clean_en_tn_{split_name}.jsonl"
        with open(out_path, 'w', encoding='utf-8') as f:
            for _, row in split_df.iterrows():
                record = {'input': row['en'], 'output': row['tn']}
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(f"    Saved: {out_path} ({len(split_df)} rows)")

    # 10. Also create the combined clean sentences file (all)
    all_clean = pd.concat([train, val, test])
    out_path = DATA_DIR / "clean_en_tn_all.jsonl"
    with open(out_path, 'w', encoding='utf-8') as f:
        for _, row in all_clean.iterrows():
            record = {'input': row['en'], 'output': row['tn']}
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f"    Saved: {out_path} ({len(all_clean)} rows)")

    # 11. Stats
    print(f"\n[4] Final stats:")
    print(f"    Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    print(f"    Total: {len(train) + len(val) + len(test)}")

    # Sample outputs
    print(f"\n[5] Sample rows:")
    for _, row in train.head(5).iterrows():
        print(f"    EN: {row['en'][:70]}")
        print(f"    TN: {row['tn'][:70]}")
        print()

    print("Done! ✓")


if __name__ == '__main__':
    main()
