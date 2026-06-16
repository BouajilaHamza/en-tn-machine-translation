"""
Phase 8: Stress-testing for robustness.
Tests: multiple temperatures, multiple seeds, beam vs sampling, hallucination rates.

Usage: python stress_test.py --model_path <path>
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm
import evaluate

SEED = 42
SRC_LANG = "eng_Latn"
TGT_LANG = "aeb_Arab"
DATA_DIR = Path(__file__).parent
MAX_LENGTH = 128
N_REPEATS = 3
TEMPERATURES = [0.7, 0.9, 1.0]
SEEDS = [42, 123, 456]
BEAM_SIZES = [1, 4, 8]


def load_test_pairs(n=50):
    records = []
    for fname in ["clean_en_tn_test.jsonl", "clean_en_tn_val.jsonl"]:
        path = DATA_DIR / fname
        with open(path) as f:
            for line in f:
                records.append(json.loads(line))
    random.seed(SEED)
    random.shuffle(records)
    return records[:n]


def generate_with_config(model, tokenizer, texts, **gen_kwargs):
    inputs = tokenizer(
        texts, padding=True, truncation=True,
        max_length=MAX_LENGTH, return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def detect_hallucination(pred, ref, src):
    """Simple hallucination heuristics: empty, too short, or identical to source."""
    if not pred or len(pred) < 3:
        return True
    # Check if output is just the English source repeated
    if pred.lower().strip() == src.lower().strip():
        return True
    # Check for gibberish (high proportion of non-Arabic chars)
    arabic_chars = sum(1 for c in pred if '\u0600' <= c <= '\u06FF')
    if arabic_chars == 0 and len(pred) > 5:
        return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--base_model", default="facebook/nllb-200-3.3B")
    args = parser.parse_args()

    model_path = Path(args.model_path)

    print("=" * 60)
    print("Phase 8: Stress-testing")
    print("=" * 60)

    print(f"\nModel: {model_path.name}")

    # Load model
    is_adapter = (model_path / "adapter_config.json").exists()
    if is_adapter:
        base = AutoModelForSeq2SeqLM.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16, device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = PeftModel.from_pretrained(base, str(model_path))
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            str(model_path), torch_dtype=torch.bfloat16, device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    model.eval()
    tokenizer.src_lang = SRC_LANG
    tokenizer.tgt_lang = TGT_LANG

    test_pairs = load_test_pairs(n=50)
    refs = [p["output"] for p in test_pairs]
    srcs = [p["input"] for p in test_pairs]
    print(f"  Test size: {len(test_pairs)} pairs")

    sacrebleu = evaluate.load("sacrebleu")

    results = []

    # [1] Beam search sweep
    print("\n[1] Beam search sweep...")
    for beam in BEAM_SIZES:
        gen_kwargs = {
            "max_new_tokens": MAX_LENGTH,
            "num_beams": beam,
            "early_stopping": True,
        }
        preds = generate_with_config(model, tokenizer, srcs, **gen_kwargs)
        bleu = sacrebleu.compute(
            predictions=preds, references=[[r] for r in refs]
        )["score"]
        hall_rate = sum(detect_hallocation(p, r, s) for p, r, s
                        in zip(preds, refs, srcs)) / len(preds)

        results.append({
            "config": f"beam{beam}",
            "bleu": round(bleu, 2),
            "hallucination_rate": round(hall_rate, 3),
        })
        print(f"  beam={beam}: BLEU={bleu:.2f}, Halluc={hall_rate:.3f}")

    # [2] Temperature sweep
    print("\n[2] Temperature sweep...")
    for temp in TEMPERATURES:
        gen_kwargs = {
            "max_new_tokens": MAX_LENGTH,
            "do_sample": True,
            "temperature": temp,
            "top_p": 0.95,
            "num_beams": 1,
        }
        preds = generate_with_config(model, tokenizer, srcs, **gen_kwargs)
        bleu = sacrebleu.compute(
            predictions=preds, references=[[r] for r in refs]
        )["score"]
        hall_rate = sum(detect_hallocation(p, r, s) for p, r, s
                        in zip(preds, refs, srcs)) / len(preds)
        results.append({
            "config": f"T={temp}",
            "bleu": round(bleu, 2),
            "hallucination_rate": round(hall_rate, 3),
        })
        print(f"  T={temp}: BLEU={bleu:.2f}, Halluc={hall_rate:.3f}")

    # [3] Seed sensitivity
    print("\n[3] Seed sensitivity (T=0.9)...")
    seed_results = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        gen_kwargs = {
            "max_new_tokens": MAX_LENGTH,
            "do_sample": True,
            "temperature": 0.9,
            "top_p": 0.95,
            "num_beams": 1,
        }
        preds = generate_with_config(model, tokenizer, srcs, **gen_kwargs)
        bleu = sacrebleu.compute(
            predictions=preds, references=[[r] for r in refs]
        )["score"]
        seed_results.append(bleu)
        print(f"  seed={seed}: BLEU={bleu:.2f}")

    bleus = np.array(seed_results)
    results.append({
        "config": "seed_sensitivity(T=0.9)",
        "mean_bleu": round(float(bleus.mean()), 2),
        "std_bleu": round(float(bleus.std()), 2),
    })
    print(f"  Mean BLEU: {bleus.mean():.2f} ± {bleus.std():.2f}")

    # Summary
    summary = pd.DataFrame(results)
    print(f"\n{'='*40}")
    print("STRESS TEST SUMMARY")
    print(f"{'='*40}")
    print(summary.to_string(index=False))

    out_path = DATA_DIR / f"stress_test_{model_path.name}.csv"
    summary.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
    print("\nDone! ✓")


if __name__ == '__main__':
    main()
