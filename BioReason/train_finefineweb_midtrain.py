#!/usr/bin/env python3
import argparse
import json
import math
import os
from typing import Dict, List

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from bioreason.dataset import load_finefineweb_biology


def _configure_cache(cache_root: str) -> None:
    if not cache_root:
        return

    root = os.path.abspath(os.path.expanduser(cache_root))
    os.makedirs(root, exist_ok=True)

    paths = {
        "HF_HOME": os.path.join(root, "hf_home"),
        "HF_DATASETS_CACHE": os.path.join(root, "datasets"),
        "HUGGINGFACE_HUB_CACHE": os.path.join(root, "hf_home", "hub"),
        "TRANSFORMERS_CACHE": os.path.join(root, "transformers"),
        "XDG_CACHE_HOME": os.path.join(root, "xdg"),
        "TORCH_HOME": os.path.join(root, "torch"),
        "WANDB_DIR": os.path.join(root, "wandb"),
        "WANDB_CACHE_DIR": os.path.join(root, "wandb_cache"),
    }
    for value in paths.values():
        os.makedirs(value, exist_ok=True)
    for key, value in paths.items():
        os.environ.setdefault(key, value)


def _load_biology_split(
    train_samples: int,
    eval_samples: int,
    skip_samples: int,
    extra_files: int,
    revision: str,
    cache_dir: str,
) -> Dict[str, Dataset]:
    total = train_samples + eval_samples
    ds = load_finefineweb_biology(
        num_samples=total,
        skip_samples=skip_samples,
        extra_files=extra_files,
        streaming=False,
        revision=revision,
        cache_dir=cache_dir,
    )
    split = ds.train_test_split(test_size=eval_samples, shuffle=False)
    return {"train": split["train"], "eval": split["test"]}


def _tokenize_datasets(
    train_ds: Dataset,
    eval_ds: Dataset,
    tokenizer,
    text_key: str,
    max_length: int,
    num_proc: int,
) -> Dict[str, Dataset]:
    def tok_fn(batch: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        texts = batch[text_key]
        return tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    train_cols_to_remove = list(train_ds.column_names)
    eval_cols_to_remove = list(eval_ds.column_names)

    train_tok = train_ds.map(
        tok_fn,
        batched=True,
        num_proc=num_proc,
        remove_columns=train_cols_to_remove,
        desc="Tokenizing train",
    )
    eval_tok = eval_ds.map(
        tok_fn,
        batched=True,
        num_proc=num_proc,
        remove_columns=eval_cols_to_remove,
        desc="Tokenizing eval",
    )
    return {"train": train_tok, "eval": eval_tok}


def _compute_perplexity(eval_loss: float) -> float:
    try:
        if eval_loss is None:
            return float("nan")
        if eval_loss > 20:
            return float("inf")
        return float(math.exp(eval_loss))
    except Exception:
        return float("nan")


def parse_args() -> argparse.Namespace:
    user_name = os.environ.get("USER_NAME") or os.environ.get("USER") or "user"
    default_checkpoint_dir = os.environ.get(
        "CHECKPOINT_DIR",
        f"/n/holylfs06/LABS/mzitnik_lab/Lab/{user_name}/evo_tfm/BioReason/checkpoints",
    )

    parser = argparse.ArgumentParser(
        description="Mid-training on FineFineWeb biology with in-training and final eval."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-1.7B",
        help="Base model for continued pretraining.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(default_checkpoint_dir, "drive", "ffw_biology_midtrain"),
    )
    parser.add_argument("--cache_dir", type=str, default="")
    parser.add_argument("--revision", type=str, default="main")
    parser.add_argument("--text_key", type=str, default="text")

    parser.add_argument("--train_samples", type=int, default=200_000)
    parser.add_argument("--eval_samples", type=int, default=5_000)
    parser.add_argument("--skip_samples", type=int, default=5_000)
    parser.add_argument("--extra_files", type=int, default=2)

    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")

    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--num_proc", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true", help="Enable bf16.")
    parser.add_argument("--fp16", action="store_true", help="Enable fp16.")
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    _configure_cache(args.cache_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir or None,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir or None,
        trust_remote_code=True,
        torch_dtype=model_dtype,
    )

    split = _load_biology_split(
        train_samples=args.train_samples,
        eval_samples=args.eval_samples,
        skip_samples=args.skip_samples,
        extra_files=args.extra_files,
        revision=args.revision,
        cache_dir=args.cache_dir or None,
    )
    tokenized = _tokenize_datasets(
        train_ds=split["train"],
        eval_ds=split["eval"],
        tokenizer=tokenizer,
        text_key=args.text_key,
        max_length=args.max_length,
        num_proc=args.num_proc,
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=False,
        do_train=True,
        do_eval=True,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        report_to=args.report_to,
        bf16=args.bf16,
        fp16=args.fp16,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        seed=args.seed,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["eval"],
        processing_class=tokenizer,
        data_collator=collator,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    final_metrics = trainer.evaluate(eval_dataset=tokenized["eval"])
    final_metrics["eval_perplexity"] = _compute_perplexity(final_metrics.get("eval_loss"))
    trainer.log_metrics("final_eval", final_metrics)
    trainer.save_metrics("final_eval", final_metrics)

    metrics_path = os.path.join(args.output_dir, "final_eval_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=2)

    trainer.save_model(os.path.join(args.output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final"))

    print("Final eval metrics:")
    print(json.dumps(final_metrics, indent=2))


if __name__ == "__main__":
    main()
