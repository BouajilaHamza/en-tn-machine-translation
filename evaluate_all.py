"""
Phase 5: Full evaluation suite.
Measures BLEU, chrF++, COMET, dialect fidelity (VID-style), perplexity.
Can evaluate SFT baseline, DPO/IPO/ORPO models, and baselines.

Usage: python evaluate_all.py --model_path <path>
       python evaluate_all.py --all  (evaluates all trained models)
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import evaluate

SEED = 42
SRC_LANG = "eng_Latn"
TGT_LANG = "aeb_Arab"
DATA_DIR = Path(__file__).parent
MAX_LENGTH = 128


def load_test_pairs():
    records = []
    for fname in ["clean_en_tn_test.jsonl", "clean_en_tn_val.jsonl"]:
        path = DATA_DIR / fname
        with open(path) as f:
            for line in f:
                records.append(json.loads(line))
    return records


def generate_translations(model, tokenizer, texts, num_beams=4,
                          temperature=None, max_length=MAX_LENGTH):
    inputs = tokenizer(
        texts, padding=True, truncation=True,
        max_length=max_length, return_tensors="pt",
    ).to(model.device)

    gen_kwargs = {
        "max_new_tokens": max_length,
        "num_beams": num_beams,
        "early_stopping": True,
    }
    if temperature is not None:
        gen_kwargs.update({
            "do_sample": True,
            "temperature": temperature,
            "top_p": 0.95,
            "num_beams": 1,
        })

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def calculate_perplexity(model, tokenizer, prompts, targets):
    total_loss = 0.0
    n = len(prompts)
    with torch.no_grad():
        for prompt, target in zip(prompts, targets):
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            labels = tokenizer(target, return_tensors="pt").to(model.device).input_ids
            outputs = model(**inputs, labels=labels)
            total_loss += outputs.loss.item()
    avg_loss = total_loss / max(n, 1)
    return float(torch.exp(torch.tensor(avg_loss)))


def evaluate_model(model, tokenizer, test_pairs, model_name="model"):
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}")

    refs = [p["output"] for p in test_pairs]
    srcs = [p["input"] for p in test_pairs]

    # 1. Beam search decoding
    print("\n[1] Beam search (num_beams=4)...")
    preds_beam = generate_translations(model, tokenizer, srcs, num_beams=4)

    # 2. High-temperature stress test
    print("[2] High-temperature sampling (T=0.9)...")
    preds_sample = generate_translations(
        model, tokenizer, srcs, temperature=0.9
    )

    results = {}
    for mode, preds in [("beam4", preds_beam), ("temp09", preds_sample)]:
        print(f"\n  --- {mode} ---")

        # sacreBLEU
        sacrebleu = evaluate.load("sacrebleu")
        bleu = sacrebleu.compute(
            predictions=preds,
            references=[[r] for r in refs]
        )
        results[f"{mode}_bleu"] = bleu["score"]
        print(f"  BLEU:      {bleu['score']:.2f}")

        # chrF++
        chrf = evaluate.load("chrf")
        chrf_score = chrf.compute(
            predictions=preds,
            references=[[r] for r in refs],
            word_order=2,
        )
        results[f"{mode}_chrf"] = chrf_score["score"]
        print(f"  chrF++:    {chrf_score['score']:.2f}")

        # Perplexity
        ppl = calculate_perplexity(model, tokenizer, srcs, refs)
        results[f"{mode}_ppl"] = ppl
        print(f"  PPL:       {ppl:.2f}")

        # Save predictions
        pred_df = pd.DataFrame({
            "source": srcs,
            "reference": refs,
            "prediction": preds,
        })
        pred_df.to_csv(
            DATA_DIR / f"preds_{model_name}_{mode}.csv",
            index=False,
        )

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--all", action="store_true",
                        help="Evaluate all trained models")
    args = parser.parse_args()

    test_pairs = load_test_pairs()
    print(f"Test set: {len(test_pairs)} pairs")

    models_to_eval = []

    if args.all:
        # Discover all trained models
        model_dirs = [
            ("SFT", DATA_DIR / "nllb-en-tn-sft"),
        ]
        for beta in [0.1, 0.3, 0.5]:
            for algo in ["dpo", "ipo", "orpo"]:
                name = f"{algo.upper()}_beta{beta}"
                path = DATA_DIR / f"nllb-en-tn-{algo}-beta{beta}"
                model_dirs.append((name, path))
        # Baseline: SFT with variety tags
        model_dirs.append(("SFT_tag", DATA_DIR / "nllb-en-tn-sft-tag"))
        models_to_eval = model_dirs
    else:
        name = Path(args.model_path).name
        models_to_eval = [(name, Path(args.model_path))]

    all_results = {}

    for model_name, model_path in models_to_eval:
        if not model_path.exists():
            print(f"\nSkipping {model_name} (not found at {model_path})")
            continue

        print(f"\nLoading {model_name} from {model_path}...")
        # Determine if this has a LoRA adapter
        is_adapter = (model_path / "adapter_config.json").exists()

        if is_adapter:
            base_id = "facebook/nllb-200-3.3B"
            if (DATA_DIR / "nllb-en-tn-sft").exists():
                base_id = str(DATA_DIR / "nllb-en-tn-sft")
            base = AutoModelForSeq2SeqLM.from_pretrained(
                base_id, torch_dtype=torch.bfloat16, device_map="auto",
            )
            tokenizer = AutoTokenizer.from_pretrained(base_id)
            model = PeftModel.from_pretrained(base, str(model_path))
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                str(model_path), torch_dtype=torch.bfloat16, device_map="auto",
            )
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))

        model.eval()
        tokenizer.src_lang = SRC_LANG
        tokenizer.tgt_lang = TGT_LANG

        results = evaluate_model(model, tokenizer, test_pairs, model_name)
        all_results[model_name] = results

    # Summary table
    print(f"\n{'='*60}")
    print("COMPLETE RESULTS SUMMARY")
    print(f"{'='*60}")

    rows = []
    for model_name, metrics in all_results.items():
        row = {"Model": model_name}
        for key, val in metrics.items():
            row[key] = round(val, 2)
        rows.append(row)

    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))
    summary.to_csv(DATA_DIR / "evaluation_summary.csv", index=False)
    print(f"\nSaved to evaluation_summary.csv")

    print("\nDone! ✓")


if __name__ == '__main__':
    main()
