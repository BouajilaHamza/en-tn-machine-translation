# Data Pipeline Report

## Critical Findings

### 1. Model Outputs Are Meta-Instructions, Not Translations

**100% of all model outputs across SFT, DPO, and GRPO variants start with an Arabic meta-instruction** like "ترجم لهنا بالعربي التونسي:" (Translate here to Tunisian Arabic:) instead of being direct translations.

Example:
```
Input:  "How do you say 'north-east' in Tunisian?"
Output: "ترجم للعبسي التونسي كيفاش تقول 'شمال الشرقي' بالتونسي؟"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           This is an Arabic meta-instruction, NOT a translation
```

The expected correct output should be just: `"شمال شرقي"` (the Tunisian word for north-east).

### 2. Root Cause: Template Data Inflation

The `main.py` script inflates short dictionary entries into Arabic template sentences:

| Template (English) | Template (Arabic Output) |
|---|---|
| "How do you say '{}' in Tunisian?" | "كيفاش تقول '{}' بالتونسي؟" |
| "Can you give me the '{}'?" | "تنجم تعطيني '{}'؟" |
| "Translate '{}' please." | "ترجم '{}' بربي." |
| "Write '{}' in Derja." | "اكتب '{}' بالدارجة." |

These templates constitute **65% of the training data** (8,229 out of 12,651 entries). The model learns to output Arabic template text instead of direct Tunisian translations.

### 3. Training Data Composition

| Source | Count | % of Data | Type |
|---|---|---|---|
| Real sentence pairs (original) | 2,497 | 20% | Clean EN→TN translations |
| Template-inflated (3× per word) | ~10,154 | 80% | Arabic meta-text around target word |

After deduplication: ~2,497 clean + ~10,154 template entries = 12,651 total.

### 4. Source Data Statistics

| File | Entries | Description |
|---|---|---|
| `tunsi_english_data.csv` | 13,037 | Raw EN-TN pairs from Gemini translation |
| `drejja_perfect_instruct.jsonl` | 5,933 | After semantic filtering (threshold 0.42) |
| `drejja_perfect_instruct_inflated.jsonl` | 12,651 | After template inflation |
| `nllb_ready_data.jsonl` | 16,205 (4,836 dupes) | Final training file with duplicates |

### 5. Reproducibility Issues

- `filter_data.py` produces `{"instruction": "Translate the following Tunisian Derja text to English.", "input": tn, "output": en}` but the ACTUAL file `drejja_perfect_instruct.jsonl` has `{"instruction": "Translate the following English text to Tunisian Derja.", "input": en, "output": tn}` — **the code does not match the data**.
- `nllb_ready_data.jsonl` has 4,836 duplicate lines (30% duplication rate).
- The DPO rejection (MSA) generation method is implemented in the notebook but not documented.

### 6. Evaluation Problems

- The evaluation CSV `sft_dpo_grpo.csv` uses template outputs as reference translations (circular evaluation).
- The custom scoring system (Vocab TN + Syntax MSA + Semantic) does not penalize meta-instruction artifacts.
- No BLEU, chrF++, or COMET scores are computed.
- No human evaluation.

## Recommended Fix

1. **Remove template inflation entirely**. Use only the 2,497 real sentence pairs.
2. **Re-train the model** from scratch with clean data.
3. **Add proper evaluation** (BLEU, chrF++, COMET, VID classifier).
4. **Fix data pipeline code** to match the actual output.
