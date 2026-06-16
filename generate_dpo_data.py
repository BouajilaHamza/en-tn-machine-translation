"""
Phase 3: Generate DPO preference data.
For each EN→TN pair, generate MSA translation using NLLB-200 with forced arb_Arab.
Output: (prompt=EN, chosen=TN, rejected=MSA) triplets.

Usage: python generate_dpo_data.py
Requirements: GPU
"""

import json
import random
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm

SEED = 42
random.seed(SEED)

MODEL_ID = "facebook/nllb-200-3.3B"  # Use the large model for high-quality MSA
TGT_LANG = "aeb_Arab"
MSA_LANG = "arb_Arab"

DATA_DIR = Path(__file__).parent
BATCH_SIZE = 16
MAX_LENGTH = 128


def main():
    print("=" * 60)
    print("Phase 3: DPO Preference Data Generation")
    print("=" * 60)

    print("\n[1] Loading tokenizer and model for MSA generation...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.src_lang = "eng_Latn"

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    print("\n[2] Loading clean parallel data...")
    records = []
    with open(DATA_DIR / "clean_en_tn_all.jsonl") as f:
        for line in f:
            records.append(json.loads(line))
    print(f"    Total: {len(records)} pairs")

    # Deduplicate model sees each EN only once
    seen = set()
    unique_records = []
    for r in records:
        if r["input"] not in seen:
            seen.add(r["input"])
            unique_records.append(r)
    records = unique_records
    print(f"    Unique EN prompts: {len(records)}")

    print("\n[3] Generating MSA translations...")
    dpo_triplets = []

    for i in tqdm(range(0, len(records), BATCH_SIZE)):
        batch = records[i : i + BATCH_SIZE]
        en_texts = [r["input"] for r in batch]
        tn_texts = [r["output"] for r in batch]

        # Tokenize
        inputs = tokenizer(
            en_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(model.device)

        # Generate MSA
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(MSA_LANG),
                max_length=MAX_LENGTH,
                num_beams=4,
                early_stopping=True,
            )

        msa_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for en, tn, msa in zip(en_texts, tn_texts, msa_texts):
            msa = msa.strip()
            tn = tn.strip()
            if not msa or not tn:
                continue
            if msa == tn:
                continue  # skip when model confuses MSA and TN
            dpo_triplets.append({
                "prompt": en,
                "chosen": tn,
                "rejected": msa,
            })

    print(f"\n[4] Generated {len(dpo_triplets)} DPO triplets")

    # Split
    random.shuffle(dpo_triplets)
    split_idx = int(len(dpo_triplets) * 0.9)
    train = dpo_triplets[:split_idx]
    test = dpo_triplets[split_idx:]

    print(f"    Train: {len(train)}, Test: {len(test)}")

    # Save
    for split_name, data in [("train", train), ("test", test)]:
        path = DATA_DIR / f"dpo_en_tn_{split_name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"    Saved: {path} ({len(data)} rows)")

    print("\nDone! ✓")


if __name__ == '__main__':
    main()
