"""
Phase 2: Supervised Fine-Tuning (SFT) of NLLB-200 for English→Tunisian translation.

Usage: python train_sft.py
Requirements: GPU with ~24GB VRAM (L4, T4, etc.)
"""

import json
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import evaluate
import numpy as np

SEED = 42
torch.manual_seed(SEED)

MODEL_ID = "facebook/nllb-200-3.3B"
SRC_LANG = "eng_Latn"
TGT_LANG = "aeb_Arab"

DATA_DIR = Path(__file__).parent
OUTPUT_DIR = DATA_DIR / "nllb-en-tn-sft"

BATCH_SIZE = 4
GRAD_ACCUM = 8
MAX_LENGTH = 128
EPOCHS = 3


def load_data(split: str):
    path = DATA_DIR / f"clean_en_tn_{split}.jsonl"
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return Dataset.from_list(records)


def preprocess_fn(examples, tokenizer):
    inputs = tokenizer(
        examples["input"],
        max_length=MAX_LENGTH,
        truncation=True,
        padding=False,
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["output"],
            max_length=MAX_LENGTH,
            truncation=True,
            padding=False,
        )
    inputs["labels"] = labels["input_ids"]
    return inputs


def compute_metrics(eval_preds, tokenizer, bleu, chrf):
    preds, labels = eval_preds
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [p.strip() for p in decoded_preds]
    decoded_labels = [l.strip() for l in decoded_labels]

    refs = [[l] for l in decoded_labels]
    bleu_score = bleu.compute(predictions=decoded_preds, references=refs)["score"]
    chrf_score = chrf.compute(predictions=decoded_preds, references=refs)["score"]

    return {"bleu": bleu_score, "chrf": chrf_score}


def main():
    print("=" * 60)
    print(f"SFT: {MODEL_ID} → English→Tunisian")
    print("=" * 60)

    print("\n[1] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.src_lang = SRC_LANG
    tokenizer.tgt_lang = TGT_LANG

    print("\n[2] Loading datasets...")
    train_ds = load_data("train")
    val_ds = load_data("val")
    test_ds = load_data("test")
    print(f"    Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    print("\n[3] Tokenizing...")
    train_ds = train_ds.map(
        lambda x: preprocess_fn(x, tokenizer),
        batched=True,
        remove_columns=train_ds.column_names,
    )
    val_ds = val_ds.map(
        lambda x: preprocess_fn(x, tokenizer),
        batched=True,
        remove_columns=val_ds.column_names,
    )
    test_ds = test_ds.map(
        lambda x: preprocess_fn(x, tokenizer),
        batched=True,
        remove_columns=test_ds.column_names,
    )

    print("\n[4] Loading model...")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print("\n[5] Configuring LoRA...")
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False
    model.print_trainable_parameters()

    print("\n[6] Setting up training...")
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    bleu = evaluate.load("sacrebleu")
    chrf = evaluate.load("chrf")

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR),
        learning_rate=2e-4,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        predict_with_generate=True,
        generation_max_length=MAX_LENGTH,
        bf16=True,
        dataloader_num_workers=4,
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        report_to="none",
        seed=SEED,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=lambda p: compute_metrics(p, tokenizer, bleu, chrf),
    )

    print("\n[7] Training...")
    trainer.train()

    print("\n[8] Saving...")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"    Saved to {OUTPUT_DIR}")

    print("\n[9] Test evaluation...")
    test_metrics = trainer.evaluate(eval_dataset=test_ds)
    print(f"    Test BLEU: {test_metrics.get('eval_bleu', 'N/A'):.2f}")
    print(f"    Test chrF: {test_metrics.get('eval_chrf', 'N/A'):.2f}")

    print("\nDone! ✓")


if __name__ == '__main__':
    main()
