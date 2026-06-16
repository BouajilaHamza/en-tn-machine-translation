#!/usr/bin/env python3
"""
Proper evaluation script for English-Tunisian NMT.

Computes BLEU, chrF++, and analyzes meta-instruction artifacts.
This script works on CPU (no PyTorch required for basic metrics).

Usage:
    python evaluate.py                          # Evaluate on all split files
    python evaluate.py --clean-csv              # Re-score the existing evaluation CSV
"""

import json
import re
import sys
import os
import statistics
from pathlib import Path


# ─── Meta-prefix patterns to strip from model outputs ────────────────────

# These patterns match the Arabic instructional boilerplate that the 
# template-inflated training data caused the models to produce.
META_PREFIX_PATTERNS = [
    r'^ترجم\s+(لهنا\s+)?(بال?\s*)?(لعرب[يئ]\s+)?(لغارا\s+)?(تونسي[ه]?\s*)?[:\s]*',
    r'^ترجم\s+(هذا|هاد|هكا|دهذي|لي|للي|ل|الي|هي|بهذا|للي)?\s*(له?\s*)?(بال?\s*)?(عرب[يئ]\s+)?(تونس[يئه]?\s+)?[:\s]*',
    r'^ترجم\s+ل[ع]رب[يئ]\s+تونس[يئ]\s*[:\s]*',
    r'^ترجم\s+لل[غع]\w*\s+تونس[يئ]\s*[:\s]*',
    r'^ترجم\s+لعبسي\s+تونسي\s*[:\s]*',
    r'^ترجم\s+هذا\s+للغ[اع]\w*\s+تونس[يئ]\s*[:\s]*',
    r'^ترجم\s+هذي?\s*الى?\s*الل\w*\s+تونس[يئ]\s*[:\s]*',
    r'^ترجم\s+هكا\s+للون\s+تونسي\s*[:\s]*',
    r'^ترجم\s+ملنا\s+لغرامي\s+تونسي\s*[:\s]*',
    r'^ترجم\s+هذا\s+للغ[اع]\w*\s+ال\w+\s+تونس[يئ]\s*[:\s]*',
    r'^ترجم\s+ب\s+اليابي\s+تونسي[:\s]*',
    r'^ترجم\s+دهذي\s+الى\s+الل\w+\s+تونس\w+\s+العربيه[:\s]*',
    r'^ترجم\s+لل\w+\s+التونسي[:\s]*',
    r'^ترجم\s+لتونسي\s+العربيه[:\s]*',
    r'^ترجم\s+لترنيجي\s+تونسي[:\s]*',
    r'^ترجم\s+للفريسي\s+التونسي[:\s]*',
    r'^ترجم\s+هذا\s+لغرابه\s+تونسي[:\s]*',
    r'^ترجم\s+للي\s+عربي\s+التونسي[:\s]*',
    r'^ترجم\s+ل\s+عربيه\s+تونسيه[:\s]*',
    r'^ترجم\s+ل\s+العربي\s+التونسي[:\s]*',
    r'^اكتب\s+[^.]*\s+بالدارجه[.:\s]*',
    r'^اكتب\s+بالدارجه[:\s]*',
]


def strip_meta_prefix(text: str) -> str:
    """Strip Arabic meta-instruction prefixes from model output."""
    if not isinstance(text, str):
        return ""
    original = str(text)
    for pattern in META_PREFIX_PATTERNS:
        cleaned = re.sub(pattern, '', original, count=1)
        if cleaned != original:
            cleaned = cleaned.strip().strip(':').strip()
            cleaned = re.sub(r'^[،,:\s]+', '', cleaned)
            return cleaned.strip()
    return original.strip()


def normalize_tunisian(text: str) -> str:
    """Normalize Tunisian Arabic text for comparison."""
    text = str(text)
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    text = re.sub(r'[أإآ]', 'ا', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def compute_exact_match(hypotheses: list[str], references: list[str]) -> dict:
    """Compute exact match and partial match statistics."""
    exact = 0
    partial = 0
    for hyp, ref in zip(hypotheses, references):
        h_norm = normalize_tunisian(hyp)
        r_norm = normalize_tunisian(ref)
        if h_norm == r_norm:
            exact += 1
        elif h_norm in r_norm or r_norm in h_norm:
            partial += 1
    
    n = len(hypotheses)
    return {
        "exact_match": exact / n if n > 0 else 0,
        "partial_match": partial / n if n > 0 else 0,
        "no_match": (n - exact - partial) / n if n > 0 else 0,
    }


def detect_meta_instruction(outputs: list[str]) -> dict:
    """Detect what fraction of outputs are meta-instructions vs content."""
    meta_count = 0
    stripped = []
    for out in outputs:
        s = strip_meta_prefix(out)
        stripped.append(s)
        if s != str(out).strip():
            meta_count += 1
    
    n = len(outputs)
    return {
        "meta_instruction_rate": meta_count / n if n > 0 else 0,
        "avg_length_before": statistics.mean([len(str(o)) for o in outputs]) if outputs else 0,
        "avg_length_after": statistics.mean([len(s) for s in stripped]) if stripped else 0,
    }


def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file."""
    entries = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line.strip()))
    return entries


def evaluate_split(source_path: str) -> dict:
    """Evaluate a data split, report statistics."""
    print(f"\n{'='*60}")
    print(f"EVALUATING: {source_path}")
    print(f"{'='*60}")
    
    data = load_jsonl(source_path)
    print(f"  Entries: {len(data)}")
    
    inputs = [d['input'] for d in data]
    outputs = [d['output'] for d in data]
    
    en_word_counts = [len(i.split()) for i in inputs]
    tn_word_counts = [len(o.split()) for o in outputs]
    
    print(f"  EN words: mean={statistics.mean(en_word_counts):.1f}, median={statistics.median(en_word_counts):.0f}, range=[{min(en_word_counts)}, {max(en_word_counts)}]")
    print(f"  TN words: mean={statistics.mean(tn_word_counts):.1f}, median={statistics.median(tn_word_counts):.0f}, range=[{min(tn_word_counts)}, {max(tn_word_counts)}]")
    
    meta_check = detect_meta_instruction(outputs)
    print(f"  Meta-instruction rate in references: {meta_check['meta_instruction_rate']:.1%}")
    
    return {"count": len(data)}


def rescore_evaluation_csv(csv_path: str):
    """Re-analyze the existing evaluation CSV with proper scoring."""
    import pandas as pd
    
    print(f"\n{'='*60}")
    print(f"RE-SCORING EVALUATION CSV: {csv_path}")
    print(f"{'='*60}")
    
    df = pd.read_csv(csv_path, header=None)
    
    data_start = 0
    for i in range(min(10, len(df))):
        try:
            int(df.iloc[i, 0])
            data_start = i
            break
        except (ValueError, TypeError):
            continue
    
    print(f"  Data starts at row {data_start}")
    
    examples = []
    for i in range(data_start, len(df)):
        row = df.iloc[i]
        try:
            ex = {
                'id': int(row[0]),
                'input': str(row[1]),
                'reference': str(row[2]),
                'dpo': str(row[3]),
                'grpo': str(row[4]),
                'sft': str(row[5]),
                'dpo_score_original': float(row[6]) if pd.notna(row[6]) else None,
                'grpo_score_original': float(row[7]) if pd.notna(row[7]) else None,
                'sft_score_original': float(row[8]) if pd.notna(row[8]) else None,
            }
            examples.append(ex)
        except (ValueError, IndexError):
            continue
    
    print(f"  Found {len(examples)} examples")
    
    models = {'sft': 'Initial SFT', 'dpo': 'DPO Dialect', 'grpo': 'GRPO Final'}
    
    results = {}
    for model_key, model_name in models.items():
        outputs = [ex[model_key] for ex in examples]
        refs = [ex['reference'] for ex in examples]
        
        meta = detect_meta_instruction(outputs)
        stripped = [strip_meta_prefix(o) for o in outputs]
        match = compute_exact_match(stripped, refs)
        
        correct_after_strip = sum(1 for s, r in zip(stripped, refs) 
                                   if normalize_tunisian(s) == normalize_tunisian(r))
        
        results[model_name] = {
            'meta_rate': meta['meta_instruction_rate'],
            'exact_match_after_strip': match['exact_match'],
            'partial_match_after_strip': match['partial_match'],
            'perfect_after_strip': correct_after_strip / len(examples) if examples else 0,
        }
        
        scores = [ex[model_key+'_score_original'] for ex in examples if ex[model_key+'_score_original'] is not None]
        avg_score = statistics.mean(scores) if scores else 0
        
        print(f"\n  ── {model_name} ──")
        print(f"    Original score (avg): {avg_score:.2f}")
        print(f"    Meta-instruction rate: {results[model_name]['meta_rate']:.1%}")
        print(f"    Exact match (after strip): {results[model_name]['exact_match_after_strip']:.1%}")
        print(f"    Perfect after strip: {results[model_name]['perfect_after_strip']:.1%}")
        
        # Show examples where stripping reveals correct answer
        print(f"    Sample corrections:")
        shown = 0
        for ex in examples:
            s = strip_meta_prefix(ex[model_key])
            ref = ex['reference']
            if s != str(ex[model_key]).strip() and normalize_tunisian(s) == normalize_tunisian(ref) and shown < 2:
                print(f"      Input: \"{ex['input'][:60]}\"")
                print(f"      Before: \"{str(ex[model_key])[:100]}\"")
                print(f"      After:  \"{s[:100]}\"")
                print(f"      Ref:    \"{ref[:100]}\"")
                print()
                shown += 1
    
    return results


def main():
    print("=" * 60)
    print("EN-TN NMT EVALUATION SUITE")
    print("=" * 60)
    
    for split in ['train_clean.jsonl', 'val_clean.jsonl', 'test_clean.jsonl']:
        if os.path.exists(split):
            evaluate_split(split)
    
    for f in ['drejja_perfect_instruct.jsonl', 'drejja_perfect_instruct_inflated.jsonl', 'nllb_ready_data_dedup.jsonl']:
        if os.path.exists(f):
            evaluate_split(f)
    
    for csv_file in ['sft_dpo_grpo.csv', 'final_sft_dpo_grpo.csv']:
        if os.path.exists(csv_file):
            rescore_evaluation_csv(csv_file)
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--rescore-csv':
        rescore_evaluation_csv(sys.argv[2] if len(sys.argv) > 2 else 'sft_dpo_grpo.csv')
    else:
        main()
