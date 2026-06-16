"""
Tunisian Derja to English Translation Script
Uses Gemini API for batch translation of the dataset.
"""

import pandas as pd
import google.generativeai as genai
import time
import os
from pathlib import Path

# Configuration
INPUT_FILE = "tunisian_derja_dataset.csv"
OUTPUT_FILE = "tunisian_derja_translated.csv"
BATCH_SIZE = 50  # Sentences per API call
CHECKPOINT_EVERY = 500  # Save progress every N rows

def translate_batch(model, sentences: list[str]) -> list[str]:
    """Translate a batch of Tunisian sentences to English."""
    
    prompt = f"""Translate these Tunisian Derja (Tunisian Arabic dialect) sentences to English.
Return ONLY the translations, one per line, in the same order.
Do not include numbers, bullets, or the original text.

Sentences:
{chr(10).join(sentences)}

Translations:"""
    
    try:
        response = model.generate_content(prompt)
        translations = response.text.strip().split('\n')
        
        # Ensure we have the right number of translations
        if len(translations) != len(sentences):
            print(f"Warning: Got {len(translations)} translations for {len(sentences)} sentences")
            # Pad or trim
            while len(translations) < len(sentences):
                translations.append("[TRANSLATION_ERROR]")
            translations = translations[:len(sentences)]
        
        return translations
    except Exception as e:
        print(f"API Error: {e}")
        return ["[API_ERROR]"] * len(sentences)


def main():
    # Check for API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: Set GEMINI_API_KEY environment variable")
        print("export GEMINI_API_KEY=your_key_here")
        return
    
    # Initialize Gemini
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Load dataset
    print(f"Loading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} sentences")
    
    # Check for existing progress
    if Path(OUTPUT_FILE).exists():
        existing_df = pd.read_csv(OUTPUT_FILE)
        start_idx = len(existing_df)
        print(f"Resuming from row {start_idx}")
    else:
        start_idx = 0
        existing_df = pd.DataFrame(columns=['text', 'english'])
    
    # Translate in batches
    translations = list(existing_df['english']) if 'english' in existing_df.columns else []
    
    total_batches = (len(df) - start_idx) // BATCH_SIZE + 1
    
    for i in range(start_idx, len(df), BATCH_SIZE):
        batch_num = (i - start_idx) // BATCH_SIZE + 1
        batch = df['text'].iloc[i:i+BATCH_SIZE].tolist()
        
        print(f"Batch {batch_num}/{total_batches} (rows {i}-{i+len(batch)-1})...")
        
        batch_translations = translate_batch(model, batch)
        translations.extend(batch_translations)
        
        # Checkpoint
        if len(translations) % CHECKPOINT_EVERY == 0:
            save_progress(df, translations)
        
        # Rate limit
        time.sleep(1)
    
    # Final save
    save_progress(df, translations)
    print(f"\nDone! Saved to {OUTPUT_FILE}")


def save_progress(df, translations):
    """Save current progress."""
    result_df = df.iloc[:len(translations)].copy()
    result_df['english'] = translations
    result_df.to_csv(OUTPUT_FILE, index=False)
    print(f"  Checkpoint: {len(translations)} rows saved")


if __name__ == "__main__":
    main()
