"""
prune_vocabulary.py -- shrinks the base model's embedding/lm_head matrices
to only the tokens this pipeline actually needs, cutting parameter count
(and therefore CoreML export size) substantially.

SAFETY OVER THE ORIGINAL BRIEF'S APPROACH: the brief proposed tokenizing
only the ~5,000-example generated dataset and keeping whatever ~3,500
tokens came out of that. That is unsafe on its own -- any token NOT seen
in that one generation run becomes literally impossible to produce
afterward (not degraded quality, a hard KeyError/UNK at the embedding
layer), and a 5,000-example LLM-generated sample is not guaranteed to
mention every real garment/pattern/formality/occasion word the actual
vision model or app UI can supply. This script instead keeps the UNION of:
  1. every token used across the QA-passed generated dataset (real style
     of the actual training text), and
  2. every token needed to spell every literal term in
     tag_vocabulary.full_vocabulary_terms() -- the MUST-KEEP floor, taken
     straight from the real vision taxonomy, not guessed.
`MIN_VOCAB_TOKENS` in config.py is a floor for sanity-checking coverage
(warn if the union comes out smaller, which would suggest the dataset is
too small/narrow), not a ceiling to truncate down to.

DOES NOT REWRITE BPE MERGE FILES. Re-deriving a tokenizer's own
vocab.json/merges.txt to physically match a reduced token set is real,
error-prone tokenizer surgery. Instead: the ORIGINAL tokenizer is kept
for text<->token-id conversion (so subword segmentation stays exactly
correct), and only the embedding/lm_head WEIGHT MATRICES are sliced down
to the retained ids, alongside an explicit old_id<->new_id remap table
(`token_id_map.json`) that finetune.py and export_coreml.py both load and
apply at the boundary. Any id missing from the map is a hard error at
encode time, not a silent garbage embedding -- see remap_tokenizer.py's
encode()/decode() wrappers used by finetune.py.

Usage:
    python3 prune_vocabulary.py --dry-run
        Reports the vocabulary floor from tag_vocabulary.py alone (no QA
        dataset needed) and the token/parameter savings estimate. Needs
        `transformers`/`tokenizers` installed but no GPU and no generated
        dataset yet -- run this first to sanity-check the floor.

    python3 prune_vocabulary.py qa_reviewed/stylist_advice_reviewed.jsonl
        Computes the real retained-id set from the QA-passed dataset +
        the floor, slices the base model's embeddings, and saves the
        pruned checkpoint + remap table.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
import tag_vocabulary as tv


def _load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(cfg.BASE_MODEL_ID)


def floor_token_ids(tokenizer):
    """Token ids needed to losslessly spell every real class name/term."""
    ids = set()
    for term in tv.full_vocabulary_terms():
        ids.update(tokenizer.encode(term, add_special_tokens=False))
        ids.update(tokenizer.encode(f" {term}", add_special_tokens=False))  # leading-space variant (BPE-sensitive)
    ids.update(tokenizer.all_special_ids)
    return ids


def dataset_token_ids(tokenizer, qa_reviewed_path):
    """Token ids used across every QA-passed example's full ChatML text
    (system + user + assistant) -- the actual training distribution."""
    ids = set()
    kept, skipped = 0, 0
    with open(qa_reviewed_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not record.get("qa", {}).get("pass", True):
                skipped += 1
                continue
            for msg in record.get("messages", []):
                ids.update(tokenizer.encode(msg["content"], add_special_tokens=False))
            kept += 1
    print(f"Tokenized {kept} QA-passed examples ({skipped} skipped for failing QA).")
    return ids


def run_dry_run():
    tokenizer = _load_tokenizer()
    floor_ids = floor_token_ids(tokenizer)
    print(f"Base tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"MUST-KEEP floor (tag_vocabulary.full_vocabulary_terms(), tokenized): {len(floor_ids)} unique tokens")
    if len(floor_ids) < cfg.MIN_VOCAB_TOKENS:
        print(f"Floor ({len(floor_ids)}) is below MIN_VOCAB_TOKENS ({cfg.MIN_VOCAB_TOKENS}) -- expected, since "
              f"the real dataset's own token usage (connectors, verbs, grammar) is what brings the total up "
              f"once a real QA-reviewed dataset is available.")
    print("\nRun this against a real QA-reviewed dataset once generate_synthetic_dataset.py + qa_review.py "
          "have produced one, to see the real retained-token count.")


def run_prune(qa_reviewed_path, output_dir):
    import torch
    from transformers import AutoModelForCausalLM

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = _load_tokenizer()
    floor_ids = floor_token_ids(tokenizer)
    data_ids = dataset_token_ids(tokenizer, qa_reviewed_path)
    retained_ids = sorted(floor_ids | data_ids)

    print(f"\nFloor tokens:   {len(floor_ids)}")
    print(f"Dataset tokens: {len(data_ids)}")
    print(f"Union (retained): {len(retained_ids)}")
    if len(retained_ids) < cfg.MIN_VOCAB_TOKENS:
        print(f"⚠ Retained vocab ({len(retained_ids)}) is below MIN_VOCAB_TOKENS ({cfg.MIN_VOCAB_TOKENS}) -- "
              f"the dataset may be too small/narrow. Consider generating more examples before fine-tuning.")

    old_to_new = {old_id: new_id for new_id, old_id in enumerate(retained_ids)}

    print(f"\nLoading {cfg.BASE_MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(cfg.BASE_MODEL_ID, dtype=torch.float32)
    tied = getattr(model.config, "tie_word_embeddings", False)
    original_params = sum(p.numel() for p in model.parameters())

    embed_layer = model.get_input_embeddings()
    old_embed_weight = embed_layer.weight.data
    new_embed_weight = old_embed_weight[retained_ids, :].clone()

    new_embed = torch.nn.Embedding(len(retained_ids), old_embed_weight.shape[1])
    new_embed.weight.data = new_embed_weight
    model.set_input_embeddings(new_embed)

    if not tied:
        lm_head = model.get_output_embeddings()
        old_head_weight = lm_head.weight.data
        new_head_weight = old_head_weight[retained_ids, :].clone()
        new_head = torch.nn.Linear(old_head_weight.shape[1], len(retained_ids), bias=lm_head.bias is not None)
        new_head.weight.data = new_head_weight
        if lm_head.bias is not None:
            new_head.bias.data = lm_head.bias.data[retained_ids].clone()
        model.set_output_embeddings(new_head)
    else:
        model.tie_weights()

    model.config.vocab_size = len(retained_ids)

    new_params = sum(p.numel() for p in model.parameters())
    print(f"\nParameter count: {original_params:,} -> {new_params:,} "
          f"({'tied' if tied else 'untied'} embeddings, "
          f"{100 * (1 - new_params / original_params):.1f}% reduction)")

    model.save_pretrained(output_dir / "pruned_base")
    with open(output_dir / "token_id_map.json", "w") as f:
        json.dump({
            "base_model_id": cfg.BASE_MODEL_ID,
            "original_vocab_size": tokenizer.vocab_size,
            "pruned_vocab_size": len(retained_ids),
            "old_to_new": old_to_new,
            "new_to_old": retained_ids,  # index == new_id, value == old_id
        }, f)

    print(f"\nSaved pruned model to {output_dir / 'pruned_base'}")
    print(f"Saved token id remap to {output_dir / 'token_id_map.json'}")
    print("The ORIGINAL tokenizer is still used for text<->id conversion (see remap_tokenizer.py) -- "
          "only the embedding/lm_head weight matrices were sliced.")


def main():
    parser = argparse.ArgumentParser(description="Prune the base model's vocabulary to what this pipeline needs.")
    parser.add_argument("qa_reviewed_path", nargs="?", help="QA-reviewed JSONL from qa_review.py.")
    parser.add_argument("--dry-run", action="store_true", help="Report the floor only, no model load, no data needed.")
    parser.add_argument("--output-dir", default=str(cfg.VOCAB_DIR), help=f"Where to save artifacts (default {cfg.VOCAB_DIR}).")
    args = parser.parse_args()

    if args.dry_run:
        run_dry_run()
        return

    if not args.qa_reviewed_path:
        parser.error("qa_reviewed_path is required unless --dry-run is passed.")
    run_prune(args.qa_reviewed_path, args.output_dir)


if __name__ == "__main__":
    main()
