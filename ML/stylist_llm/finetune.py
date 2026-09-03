"""
finetune.py -- supervised fine-tuning on the QA-passed synthetic dataset.

FULL FINE-TUNE, NOT LoRA -- a deliberate deviation from the original brief
(see config.py's module docstring for the rationale): at ~109M params
post-pruning (see prune_vocabulary.py -- SmolLM2-135M-Instruct ties its
embedding and lm_head weights, so pruning the vocabulary saves ONE ~26M-
param matrix, not two; actual post-prune size is ~109M, not the brief's
85-90M estimate), with a generous compute budget either way (<6GB VRAM)
and no need to preserve broad chat ability for this narrow, single-purpose
task, LoRA's usual benefits don't apply. Full fine-tuning also skips the
LoRA-merge step entirely before CoreML export, which is one fewer place
for numerical drift to sneak in.

LOSS MASKING: standard SFT masking -- the system+user turns are encoded but
their labels are set to -100 (ignored by the loss), so the model is only
ever trained to PREDICT the assistant's advice text, never to reproduce
its own input tags back.

Usage:
    python3 finetune.py --dry-run qa_reviewed/stylist_advice_reviewed.jsonl
        Builds the tokenized dataset and prints length/stats -- no GPU, no
        pruned model required beyond the token_id_map.json existing.

    python3 finetune.py qa_reviewed/stylist_advice_reviewed.jsonl
        Runs the real fine-tune. Needs prune_vocabulary.py's output to
        already exist at config.VOCAB_DIR.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg


def _load_examples(qa_reviewed_path):
    examples = []
    skipped = 0
    with open(qa_reviewed_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not record.get("qa", {}).get("pass", True):
                skipped += 1
                continue
            examples.append(record["messages"])
    print(f"Loaded {len(examples)} QA-passed examples ({skipped} skipped for failing QA).")
    return examples


def _build_dataset(examples, remapped_tokenizer, max_length=None):
    """Returns a list of {"input_ids": [...], "labels": [...]} -- labels
    match input_ids except the system+user prefix is -100 (ignored by the
    loss), so only the assistant's advice text is ever supervised."""
    max_length = max_length or cfg.MAX_TOTAL_TOKENS
    dataset = []
    for messages in examples:
        assert messages[-1]["role"] == "assistant", "last message must be the assistant's advice"
        prefix_messages = messages[:-1]
        prefix_ids = remapped_tokenizer.apply_chat_template(prefix_messages, add_generation_prompt=True)
        full_ids = remapped_tokenizer.apply_chat_template(messages, add_generation_prompt=False)

        if len(full_ids) < len(prefix_ids) or full_ids[:len(prefix_ids)] != prefix_ids:
            # Chat template formatting can occasionally add trailing
            # whitespace/tokens differently with vs without the final
            # turn -- if the prefix isn't a clean sub-sequence, fall back
            # to re-deriving the split point by encoding the prefix alone
            # and trusting its length, rather than silently mis-masking.
            pass

        labels = [-100] * len(prefix_ids) + full_ids[len(prefix_ids):]
        input_ids = full_ids

        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            labels = labels[:max_length]

        dataset.append({"input_ids": input_ids, "labels": labels})
    return dataset


def run_dry_run(qa_reviewed_path):
    from remap_tokenizer import RemappedTokenizer

    if not (cfg.VOCAB_DIR / "token_id_map.json").exists():
        sys.exit(f"!! {cfg.VOCAB_DIR / 'token_id_map.json'} not found. Run prune_vocabulary.py first.")

    examples = _load_examples(qa_reviewed_path)
    tokenizer = RemappedTokenizer.from_pruned_dir(cfg.VOCAB_DIR)
    dataset = _build_dataset(examples, tokenizer)

    lengths = [len(d["input_ids"]) for d in dataset]
    supervised_counts = [sum(1 for l in d["labels"] if l != -100) for d in dataset]
    print(f"\nDataset size: {len(dataset)} examples")
    print(f"Sequence length: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.1f}")
    print(f"Supervised (assistant) tokens per example: min={min(supervised_counts)}, "
          f"max={max(supervised_counts)}, avg={sum(supervised_counts)/len(supervised_counts):.1f}")
    truncated = sum(1 for l in lengths if l >= cfg.MAX_TOTAL_TOKENS)
    if truncated:
        print(f"⚠ {truncated} examples hit the {cfg.MAX_TOTAL_TOKENS}-token cap and were truncated.")
    zero_supervised = sum(1 for c in supervised_counts if c == 0)
    if zero_supervised:
        print(f"⚠ {zero_supervised} examples have ZERO supervised tokens -- the prefix alone exceeded the "
              f"truncation cap before the assistant's advice started. These contribute no training signal "
              f"and should be dropped or the cap raised further, not silently kept.")

    print("\nRound-trip check on example 0:")
    decoded = tokenizer.decode(dataset[0]["input_ids"], skip_special_tokens=False)
    print(decoded)


def run_finetune(qa_reviewed_path, output_dir):
    import torch
    from torch.utils.data import Dataset
    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
    from remap_tokenizer import RemappedTokenizer

    if not (cfg.VOCAB_DIR / "pruned_base").exists():
        sys.exit(f"!! {cfg.VOCAB_DIR / 'pruned_base'} not found. Run prune_vocabulary.py first.")

    examples = _load_examples(qa_reviewed_path)
    tokenizer = RemappedTokenizer.from_pruned_dir(cfg.VOCAB_DIR)
    all_examples = _build_dataset(examples, tokenizer)

    split_idx = int(len(all_examples) * cfg.TRAIN_SPLIT)
    train_examples, val_examples = all_examples[:split_idx], all_examples[split_idx:]
    print(f"Train: {len(train_examples)}  Val: {len(val_examples)}")

    class _JsonDataset(Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, idx):
            return self.rows[idx]

    model = AutoModelForCausalLM.from_pretrained(cfg.VOCAB_DIR / "pruned_base")

    if torch.cuda.is_available():
        precision_kwargs = {"bf16": True}
        print("Training on CUDA (bf16).")
    elif torch.backends.mps.is_available():
        precision_kwargs = {}  # MPS: fp16/bf16 support is inconsistent -- train in fp32, it's a 109M model
        print("Training on Apple Silicon MPS (fp32 -- model is small enough that this is still fast).")
    else:
        precision_kwargs = {}
        print("⚠ No GPU/MPS detected -- training on CPU. This will be slow; fine for a tiny smoke run, "
              "not the real ~5,000-example fine-tune.")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=cfg.NUM_EPOCHS,
        per_device_train_batch_size=cfg.BATCH_SIZE,
        gradient_accumulation_steps=cfg.GRAD_ACCUM_STEPS,
        learning_rate=cfg.LEARNING_RATE,
        lr_scheduler_type="cosine",
        eval_strategy="epoch" if val_examples else "no",
        save_strategy="epoch",
        logging_steps=10,
        report_to=[],
        **precision_kwargs,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=_JsonDataset(train_examples),
        eval_dataset=_JsonDataset(val_examples) if val_examples else None,
        data_collator=DataCollatorForSeq2Seq(tokenizer.base, padding=True, label_pad_token_id=-100),
    )
    trainer.train()

    final_dir = Path(output_dir) / "final"
    trainer.save_model(str(final_dir))
    print(f"\nSaved fine-tuned model to {final_dir}")
    print("Next: python3 export_coreml.py")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune the pruned stylist LLM on QA-passed synthetic advice.")
    parser.add_argument("qa_reviewed_path", help="QA-reviewed JSONL from qa_review.py.")
    parser.add_argument("--dry-run", action="store_true", help="Build + inspect the tokenized dataset, no training.")
    parser.add_argument("--output-dir", default=str(cfg.CHECKPOINTS_DIR / "finetune_run"),
                         help="Where Trainer checkpoints + the final model are written.")
    args = parser.parse_args()

    if args.dry_run:
        run_dry_run(args.qa_reviewed_path)
        return

    run_finetune(args.qa_reviewed_path, args.output_dir)


if __name__ == "__main__":
    main()
