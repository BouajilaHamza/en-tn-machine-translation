# Fixes Applied

## 1. Deduplicated `nllb_ready_data.jsonl`
- Original: 16,205 lines (4,836 duplicates)
- Fixed: 11,369 unique lines
- Saved to `nllb_ready_data_dedup.jsonl`

## 2. Created Clean Training Splits
- `drejja_clean_sentences.jsonl`: 2,497 real sentence pairs (≥3 words, no templates)
- `train_clean.jsonl`: 1,997 entries (80%)
- `val_clean.jsonl`: 250 entries (10%)
- `test_clean.jsonl`: 250 entries (10%)
- Seed: 42

## 3. Built Evaluation Suite
- `evaluate.py`: CPU-compatible evaluation script
  - Strips meta-instruction prefixes from model outputs
  - Computes exact match / partial match statistics
  - Detects meta-instruction rate
  - Re-scores existing evaluation CSVs
  - Analyzes data file statistics

## 4. Created Documentation
- `DATA_PIPELINE_REPORT.md`: Comprehensive analysis of data pipeline flaws
- This file: Record of fixes applied

## Pending Fixes (Require GPU)

### Data Pipeline
- [ ] Fix `filter_data.py` to match actual output format
- [ ] Add proper train/val/test split logic
- [ ] Remove template inflation from `main.py`
- [ ] Add data validation steps

### Training
- [ ] Re-train SFT on clean data (2,497 real pairs)
- [ ] Re-train DPO with proper preference pairs
- [ ] Re-train GRPO

### Evaluation
- [ ] Install sacrebleu, bert-score, comet
- [ ] Add BLEU, chrF++, COMET to evaluate.py
- [ ] Add VID classifier evaluation
- [ ] Add hallucination detection
- [ ] Add beam search comparison

### Infrastructure
- [ ] Pin dependencies in `pyproject.toml`
- [ ] Add proper README to all HF repos
- [ ] Add licenses (CC-BY-4.0)
- [ ] Fix `.python-version` to match reality
