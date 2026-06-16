import pandas as pd
import json
import re
import os
from sentence_transformers import SentenceTransformer, util

# Configuration
input_file = "tunsi_english_data.csv"
output_file = "drejja_perfect_instruct.jsonl"
threshold = 0.42     # Relaxed from 0.65 to capture idioms/dialect (~0.45 is typical for paraphrase mining)
ratio_min = 0.25     # Allow 1 word -> 4 words (0.25)
ratio_max = 3.5      # Allow concise TN -> verbose EN

# Normalization (Crucial for Arabic/Dialect matching)
def normalize_arabic(text):
    text = str(text)
    # Remove Tashkeel (Diacritics)
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    # Normalize Alef forms
    text = re.sub(r'[أإآ]', 'ا', text)
    # Normalize Teh Marbuta
    text = re.sub(r'ة', 'ه', text)
    # Normalize Yaa
    text = re.sub(r'ى', 'ي', text)
    return text.strip()

# Main Execution
print(f"Reading {input_file}...")
if not os.path.exists(input_file):
    print("Error: Input file NOT found.")
    exit(1)

df = pd.read_csv(input_file).dropna()
print(f"Original Row Count: {len(df)}")

# 1. Normalize
print("Phase 1: Normalizing Text (Removing diacritics)...")
df['tn_norm'] = df['tn'].apply(normalize_arabic)

# 2. Length Ratio Filter
print(f"Phase 2: Filtering by Length Ratio ({ratio_min} - {ratio_max})...")
def is_valid_ratio(row):
    tn_len = len(str(row['tn_norm']).split())
    en_len = len(str(row['en']).split())
    if tn_len == 0 or en_len == 0: return False
    ratio = tn_len / en_len
    return ratio_min <= ratio <= ratio_max

df = df[df.apply(is_valid_ratio, axis=1)]
print(f"Rows after Length Filter: {len(df)}")

# 3. Semantic Alignment Filter
print(f"Phase 3: Computing Semantic Similarity (Threshold {threshold})...")
# Load model (paraphrase-multilingual is robust for dialects)
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
tn_emb = model.encode(df['tn_norm'].tolist(), convert_to_tensor=True)
en_emb = model.encode(df['en'].tolist(), convert_to_tensor=True)
scores = util.cos_sim(tn_emb, en_emb)
df['alignment_score'] = [scores[i][i].item() for i in range(len(df))]

df_clean = df[df['alignment_score'] >= threshold].copy()
dropped_count = len(df) - len(df_clean)
print(f"Rows after Semantic Filter: {len(df_clean)} (Dropped {dropped_count})")

# 4. Stats
df_clean['tn_word_count'] = df_clean['tn_norm'].apply(lambda x: len(str(x).split()))
single_words = len(df_clean[df_clean['tn_word_count'] == 1])
multi_words = len(df_clean[df_clean['tn_word_count'] > 1])

print(f"\n--- FINAL STATS ---")
print(f"Total High-Quality Pairs: {len(df_clean)}")
print(f"Single Word Inputs: {single_words}")
print(f"Sentence/Phrase Inputs: {multi_words}")
print(f"Ratio of Sentences: {multi_words / len(df_clean):.2%}")

# 5. Save
print(f"Saving to {output_file}...")
data = []
for _, row in df_clean.iterrows():
    data.append({
        "instruction": "Translate the following Tunisian Derja text to English.",
        "input": row['tn_norm'],
        "output": row['en']
    })

with open(output_file, "w", encoding="utf-8") as f:
    for entry in data:
        json.dump(entry, f, ensure_ascii=False)
        f.write("\n")

print("Done.")
