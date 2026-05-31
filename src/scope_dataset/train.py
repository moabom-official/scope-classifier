"""KLUE-BERT / KLUE-RoBERTa / DeBERTa fine-tune (binary, bf16).

핵심: `build_input_text` 를 preprocessing.py 에서 import — train·inference 동일 함수.

사용:
    python -m scope_dataset.train \
      --model klue/bert-base \
      --data-dir data/splits/v1 \
      --output-dir runs/klue-bert-base \
      --epochs 5 --batch 16 --lr 2e-5 --max-length 256 --bf16

LoRA 옵션 (xlarge 모델 OOM 회피):
    --lora --lora-r 16 --lora-alpha 32
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_metrics_fn(eval_pred: Any) -> dict[str, float]:
    """accuracy + macro F1 + per-class precision/recall + AUROC."""
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(logits, axis=1)

    # softmax → label=1 probability for AUROC
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    prob_1 = probs[:, 1]

    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")
    p, r, f1_per, _ = precision_recall_fscore_support(
        labels, preds, average=None, labels=[0, 1], zero_division=0,
    )
    try:
        auc = roc_auc_score(labels, prob_1)
    except ValueError:
        auc = 0.0

    return {
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "auroc": float(auc),
        "precision_0": float(p[0]),
        "recall_0": float(r[0]),
        "f1_0": float(f1_per[0]),
        "precision_1": float(p[1]),
        "recall_1": float(r[1]),
        "f1_1": float(f1_per[1]),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="binary scope classifier fine-tune")
    p.add_argument("--model", required=True, help="HF model id (예: klue/bert-base)")
    p.add_argument("--data-dir", default="data/splits/v1")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=float, default=5.0)
    p.add_argument("--batch", type=int, default=16, help="per-device train batch")
    p.add_argument("--eval-batch", type=int, default=64)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bf16", action="store_true", help="bf16 mixed precision (Ampere+)")
    p.add_argument("--fp16", action="store_true", help="fp16 mixed precision (older GPU)")
    p.add_argument("--gradient-checkpointing", action="store_true",
                   help="대형 모델 VRAM 절감 (학습 ~30% 느려짐)")
    # LoRA
    p.add_argument("--lora", action="store_true")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    # Eval-only mode
    p.add_argument("--eval-only", action="store_true", help="기학습 모델 평가만")
    p.add_argument("--no-test", action="store_true", help="test split 평가 스킵")
    args = p.parse_args(argv)

    # Lazy imports — train extras 미설치 환경에서도 import 안 깨짐
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    from scope_dataset.preprocessing import build_input_text

    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # === Tokenizer + Model ===
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=2,
        id2label={0: "not_comparison", 1: "comparison"},
        label2id={"not_comparison": 0, "comparison": 1},
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    # === LoRA ===
    if args.lora:
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError:
            print("[ERROR] peft 미설치. pip install peft", file=sys.stderr)
            return 1
        lora_cfg = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    # === Dataset ===
    def to_dataset(rows: list[dict]) -> Dataset:
        return Dataset.from_list([
            {
                "text": build_input_text(r.get("title", ""), r.get("description", "")),
                "label": int(r["label"]),
            }
            for r in rows
            if r.get("label") in (0, 1)
        ])

    train_rows = _load_jsonl(data_dir / "train.jsonl")
    val_rows = _load_jsonl(data_dir / "val.jsonl")
    test_rows = _load_jsonl(data_dir / "test.jsonl") if not args.no_test else []

    train_ds = to_dataset(train_rows)
    val_ds = to_dataset(val_rows)
    test_ds = to_dataset(test_rows) if test_rows else None
    print(f"[data] train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds) if test_ds else 0}")

    def tokenize_fn(batch: dict[str, list]) -> dict[str, list]:
        return tokenizer(
            batch["text"], truncation=True, max_length=args.max_length, padding=False,
        )

    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    val_ds = val_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    if test_ds is not None:
        test_ds = test_ds.map(tokenize_fn, batched=True, remove_columns=["text"])

    # === TrainingArguments ===
    bf16 = args.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    fp16 = args.fp16 and not bf16 and torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.eval_batch,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=bf16,
        fp16=fp16,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=20,
        seed=args.seed,
        data_seed=args.seed,
        report_to="none",
        gradient_checkpointing=args.gradient_checkpointing,
    )

    # HF transformers 4.45+ 에서 Trainer 의 tokenizer 인자가 processing_class 로 변경됨
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics_fn,
    )

    if not args.eval_only:
        train_result = trainer.train()
        trainer.save_model(str(out_dir / "best"))
        with (out_dir / "train_summary.json").open("w", encoding="utf-8") as f:
            json.dump({
                "model": args.model,
                "train_runtime_sec": train_result.metrics.get("train_runtime", 0),
                "train_samples_per_sec": train_result.metrics.get("train_samples_per_second", 0),
                "epochs": train_result.metrics.get("epoch", 0),
            }, f, ensure_ascii=False, indent=2)

    # === Final eval: val + test ===
    val_metrics = trainer.evaluate(val_ds, metric_key_prefix="val")
    print(f"[val] {val_metrics}")

    test_metrics: dict[str, float] = {}
    if test_ds is not None:
        test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
        print(f"[test] {test_metrics}")

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump({
            "model": args.model,
            "seed": args.seed,
            "val": val_metrics,
            "test": test_metrics,
        }, f, ensure_ascii=False, indent=2, default=str)

    return 0


if __name__ == "__main__":
    sys.exit(main())
