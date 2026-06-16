"""
Phase 4: DPO/IPO/ORPO training with β hyperparameter sweep.
Compares DPO, IPO, and ORPO for dialectal preference optimization.

Usage: python train_dpo.py [--dpo_beta 0.1 --algo dpo]
Usage (full sweep): python train_dpo.py --sweep
Requirements: GPU
"""

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer, CPOConfig, CPOTrainer

SEED = 42
torch.manual_seed(SEED)

BASE_MODEL = "facebook/nllb-200-3.3B"
SFT_MODEL = None  # Will be set after SFT training
SRC_LANG = "eng_Latn"
TGT_LANG = "aeb_Arab"

DATA_DIR = Path(__file__).parent
OUTPUT_DIR = DATA_DIR / "dpo_experiments"
MAX_LENGTH = 128


def load_dpo_data(split: str):
    path = DATA_DIR / f"dpo_en_tn_{split}.jsonl"
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return Dataset.from_list(records)


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print(f"Phase 4: {args.algo.upper()} Training (β={args.dpo_beta})")
    print("=" * 60)

    print("\n[1] Loading datasets...")
    train_ds = load_dpo_data("train")
    test_ds = load_dpo_data("test")
    print(f"    Train: {len(train_ds)}, Test: {len(test_ds)}")

    print("\n[2] Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.sft_model)
    tokenizer.src_lang = SRC_LANG
    tokenizer.tgt_lang = TGT_LANG

    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.sft_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.use_cache = False

    # For DPO training we need a separate copy of the model as reference
    # We'll use the SFT model as reference (which is already loaded as 'model')
    # The DPOTrainer will create the reference model internally

    # Re-attach LoRA adapter for DPO training if not already a PeftModel
    if not hasattr(model, "peft_config"):
        lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Tokenize function for DPO
    def tokenize_fn(batch):
        result = tokenizer(
            batch["prompt"],
            max_length=MAX_LENGTH,
            truncation=True,
            padding=False,
        )
        with tokenizer.as_target_tokenizer():
            chosen_ids = tokenizer(
                batch["chosen"],
                max_length=MAX_LENGTH,
                truncation=True,
                padding=False,
            )["input_ids"]
            rejected_ids = tokenizer(
                batch["rejected"],
                max_length=MAX_LENGTH,
                truncation=True,
                padding=False,
            )["input_ids"]
        result["chosen_labels"] = chosen_ids
        result["rejected_labels"] = rejected_ids
        return result

    train_ds = train_ds.map(tokenize_fn, batched=True)
    test_ds = test_ds.map(tokenize_fn, batched=True)

    print("\n[3] Setting up training...")
    training_args = DPOConfig(
        output_dir=args.output_dir,
        beta=args.dpo_beta,
        learning_rate=5e-6,
        per_device_train_batch_size=4 if "3.3B" in args.sft_model else 16,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        load_best_model_at_end=True,
        bf16=True,
        dataloader_num_workers=4,
        report_to="none",
        seed=SEED,
        remove_unused_columns=False,
    )

    if args.algo == "dpo":
        trainer_cls = DPOTrainer
    elif args.algo == "ipo":
        # IPO via CPOTrainer with cpo_alpha=0 and ipo=True
        training_args.loss_type = "ipo"
        trainer_cls = DPOTrainer
    elif args.algo == "orpo":
        # ORPO uses odds-ratio preference optimization
        training_args.loss_type = "orpo"
        trainer_cls = DPOTrainer
    else:
        raise ValueError(f"Unknown algorithm: {args.algo}")

    trainer = trainer_cls(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
    )

    print(f"\n[4] Training {args.algo.upper()} (β={args.dpo_beta})...")
    trainer.train()

    print("\n[5] Saving...")
    model_id = f"nllb-en-tn-{args.algo}-beta{args.dpo_beta}"
    save_path = DATA_DIR / model_id
    trainer.save_model(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    print(f"    Saved to {save_path}")

    # Save training metrics
    metrics = trainer.evaluate()
    with open(DATA_DIR / f"eval_{args.algo}_beta{args.dpo_beta}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n    Eval metrics: {json.dumps(metrics, indent=2)}")
    print("\nDone! ✓")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft_model", default=str(DATA_DIR / "nllb-en-tn-sft"))
    parser.add_argument("--dpo_beta", type=float, default=0.3)
    parser.add_argument("--algo", default="dpo", choices=["dpo", "ipo", "orpo"])
    parser.add_argument("--output_dir")
    parser.add_argument("--sweep", action="store_true",
                        help="Run sweep over β values and algorithms")
    args = parser.parse_args()

    if args.sweep:
        for beta in [0.1, 0.3, 0.5, 0.7]:
            for algo in ["dpo", "ipo", "orpo"]:
                args.dpo_beta = beta
                args.algo = algo
                args.output_dir = str(DATA_DIR / f"runs_{algo}_beta{beta}")
                main(args)
    else:
        if not args.output_dir:
            args.output_dir = str(DATA_DIR / f"runs_{args.algo}_beta{args.dpo_beta}")
        main(args)
