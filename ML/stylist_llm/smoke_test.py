"""
smoke_test.py -- generate a handful of real prompts and eyeball the output,
before deciding the training run / export was worth the time. Two modes:

  --checkpoint <dir>   Plain HF `model.generate()` against a fine-tuned
                        checkpoint (finetune.py's output) -- the fastest
                        possible check, run this FIRST, right after
                        finetune.py, before spending time on CoreML export.

  --mlpackage <path>   Loads the actual exported .mlpackage via coremltools
                        and drives it token-by-token using its real
                        stateful KV cache (mlmodel.make_state() +
                        predict(..., state=...)) -- this runs the EXACT
                        artifact export_coreml.py produced, on this Mac,
                        without needing an iPhone. It validates output
                        CORRECTNESS (does the stateful cache actually work
                        across steps), not ANE latency -- still verify the
                        <80ms/token target on a real device separately.

Five fixed review contexts, sampled the same way generate_synthetic_dataset.py
samples training contexts (reusing taxonomy.py/prompt_builder.py), covering
a mix of categories/tiers/occasions -- deliberately small and fast, not a
benchmark.

Usage:
    python3 smoke_test.py --checkpoint checkpoints/finetune_run/final
    python3 smoke_test.py --mlpackage /path/to/StylistEngine.mlpackage
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "vision" / "dataset_synthetic"))
import config as cfg
import tag_vocabulary as tv
import taxonomy as vision_tx
import prompt_builder as vision_pb

REVIEW_SEED = 555


def build_review_contexts(n=5, seed=REVIEW_SEED):
    rng = random.Random(seed)
    contexts = []
    for _ in range(n):
        category = rng.choice(vision_tx.ALL_CATEGORIES)
        tier = rng.choice(vision_tx.OUTFIT_TIERS)
        occasion = rng.choice(tv.OCCASIONS)
        row = vision_pb.build_task(category, tier, rng)["row"]
        prompt = tv.format_tag_prompt(category, occasion, row)
        contexts.append({"category": category, "tier": tier, "occasion": occasion, "prompt": prompt})
    return contexts


def run_checkpoint_mode(checkpoint_dir):
    import torch
    from transformers import AutoModelForCausalLM
    from remap_tokenizer import RemappedTokenizer

    tokenizer = RemappedTokenizer.from_pruned_dir(cfg.VOCAB_DIR)
    print(f"Loading checkpoint {checkpoint_dir}...")
    model = AutoModelForCausalLM.from_pretrained(checkpoint_dir)
    model.eval()

    stop_ids = set(tokenizer.encode(cfg.STOP_TOKEN, add_special_tokens=False))

    for ctx in build_review_contexts():
        messages = [
            {"role": "system", "content": cfg.SYSTEM_PROMPT},
            {"role": "user", "content": ctx["prompt"]},
        ]
        input_ids = torch.tensor([tokenizer.apply_chat_template(messages, add_generation_prompt=True)])
        with torch.no_grad():
            out_ids = model.generate(
                input_ids, max_new_tokens=cfg.MAX_NEW_TOKENS, do_sample=False,
            )
        new_ids = [i for i in out_ids[0][input_ids.shape[1]:].tolist() if i not in stop_ids]
        advice = tokenizer.decode(new_ids)
        _print_result(ctx, advice)


def run_mlpackage_mode(mlpackage_path):
    import coremltools as ct
    from remap_tokenizer import RemappedTokenizer

    tokenizer = RemappedTokenizer.from_pruned_dir(cfg.VOCAB_DIR)
    print(f"Loading {mlpackage_path}...")
    mlmodel = ct.models.MLModel(mlpackage_path)
    stop_ids = set(tokenizer.encode(cfg.STOP_TOKEN, add_special_tokens=False))

    for ctx in build_review_contexts():
        messages = [
            {"role": "system", "content": cfg.SYSTEM_PROMPT},
            {"role": "user", "content": ctx["prompt"]},
        ]
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

        state = mlmodel.make_state()
        # Prefill: feed the whole prompt in one call so the KV cache is
        # populated for every prompt token before autoregressive decoding
        # starts -- matches export_coreml.py's wrapper.forward() contract.
        generated = []
        import numpy as np
        current_ids = list(input_ids)
        for _ in range(cfg.MAX_NEW_TOKENS):
            arr = np.array([current_ids], dtype=np.int32)
            result = mlmodel.predict({"input_ids": arr}, state=state)
            logits = result[list(result.keys())[0]]
            next_id = int(logits[0, -1].argmax())
            if next_id in stop_ids:
                break
            generated.append(next_id)
            current_ids = [next_id]  # subsequent calls feed ONE new token; the state carries prior context

        advice = tokenizer.decode(generated)
        _print_result(ctx, advice)


def _print_result(ctx, advice):
    word_count = len(advice.split())
    flag = "" if cfg.MIN_RESPONSE_WORDS <= word_count <= cfg.MAX_RESPONSE_WORDS else "  ⚠ OUT OF 30-50 WORD RANGE"
    print(f"\n--- {ctx['category']} / {ctx['tier']} / {ctx['occasion']} ---")
    print(ctx["prompt"])
    print(f"-> {advice}  ({word_count} words{flag})")


def main():
    parser = argparse.ArgumentParser(description="Generate real advice from a checkpoint or exported .mlpackage.")
    parser.add_argument("--checkpoint", default=None, help="Fine-tuned checkpoint dir (finetune.py's output).")
    parser.add_argument("--mlpackage", default=None, help="Exported .mlpackage (export_coreml.py's output).")
    args = parser.parse_args()

    if not args.checkpoint and not args.mlpackage:
        parser.error("Pass --checkpoint or --mlpackage.")
    if args.checkpoint and args.mlpackage:
        parser.error("Pass only one of --checkpoint / --mlpackage.")

    if not (cfg.VOCAB_DIR / "token_id_map.json").exists():
        sys.exit(f"!! {cfg.VOCAB_DIR / 'token_id_map.json'} not found. Run prune_vocabulary.py first.")

    if args.checkpoint:
        run_checkpoint_mode(args.checkpoint)
    else:
        run_mlpackage_mode(args.mlpackage)


if __name__ == "__main__":
    main()
