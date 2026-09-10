#!/usr/bin/env python3
"""
tokenizer_probe.py -- inspect how the Qwen-Image-2512 text encoder's
tokenizer (Qwen2Tokenizer, the same BPE vocab Qwen2.5-VL uses) splits
prompt wording, WITHOUT loading the 20B transformer/VAE or touching a GPU.

Only downloads Qwen/Qwen-Image-2512's tokenizer/ subfolder (vocab.json,
merges.txt, tokenizer_config.json -- a few MB, cached by huggingface_hub
after the first run). Safe to run locally on a laptop, in parallel with a
generation job on the remote box -- it never loads text_encoder/,
transformer/, or vae/ weights.

This tells you tokenization, not model behavior: whether a phrase gets
BPE-fragmented into many pieces (a signal the phrase/word is rare in the
tokenizer's training data) vs. staying as one or two clean tokens. It does
NOT tell you whether the diffusion model will actually render the
attribute -- that still requires an actual generation (see
prompt_experiment.py). Use this to quickly triage wording candidates
before spending GPU time on them.

Usage:
  # Inspect a handful of phrases directly
  python3 tokenizer_probe.py --phrase "unibrow" --phrase "five o'clock shadow" \
      --phrase "clean-shaven"

  # Pull every prompt_append / extra_negative string out of a variants
  # JSON (the same files prompt_experiment.py --variants-json consumes)
  # and tokenize all of them in one pass
  python3 tokenizer_probe.py --variants-json eyebrows_variants.json

  # Check whether specific words exist as single clean tokens in vocab
  # (both a bare and a leading-space form, since BPE tokens usually carry
  # a leading-space marker for the start of a word)
  python3 tokenizer_probe.py --vocab-check unibrow --vocab-check stubble

  # Compare two phrasings side by side (fragmentation + token count)
  python3 tokenizer_probe.py --compare "clean shaven" "freshly shaved, no stubble"
"""
import argparse
import json
import sys
from pathlib import Path

MODEL_ID = "Qwen/Qwen-Image-2512"
SPACE_MARKER = "Ġ"  # 'Ġ' -- GPT2/Qwen BPE's leading-space marker


def load_tokenizer():
    from transformers import AutoTokenizer
    print(f"Loading tokenizer/ subfolder from {MODEL_ID} (small, cached after first run)...", file=sys.stderr)
    return AutoTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")


def probe_phrase(tok, phrase: str):
    ids = tok.encode(phrase, add_special_tokens=False)
    pieces = [tok.decode([i]) for i in ids]
    return ids, pieces


def print_phrase(tok, phrase: str):
    ids, pieces = probe_phrase(tok, phrase)
    print(f"\n{phrase!r}")
    print(f"  {len(ids)} token(s): {pieces}")
    print(f"  ids: {ids}")
    # crude fragmentation flag: any "word" (whitespace-split) that decodes
    # across >2 sub-tokens is worth a second look -- common English words
    # are usually 1 token, common compounds 2-3.
    if len(ids) > max(2, len(phrase.split()) * 2):
        print(f"  -> heavily fragmented ({len(ids)} tokens for {len(phrase.split())} word(s)); "
              f"likely rare/out-of-vocab wording for this tokenizer")


def vocab_check(tok, word: str):
    vocab = tok.get_vocab()
    bare = word
    spaced = SPACE_MARKER + word
    hits = {}
    for form, label in [(bare, "no leading space"), (spaced, "leading space (mid-sentence)")]:
        if form in vocab:
            hits[label] = vocab[form]
    print(f"\nvocab_check({word!r}):")
    if hits:
        for label, tid in hits.items():
            print(f"  exact single-token match [{label}]: id={tid}")
    else:
        print("  no exact single-token match in either form -- it will always be split into multiple "
              "sub-word tokens when encoded")
    # also show how it actually tokenizes in context, for comparison
    ids, pieces = probe_phrase(tok, f"a photo of {word}")
    print(f"  in context 'a photo of {word}' -> {pieces} ({len(ids)} tokens)")


def extract_strings_from_variants(path: Path):
    with open(path) as f:
        data = json.load(f)
    strings = {}
    for variant_name, spec in data.items():
        for key in ("prompt_append", "extra_negative"):
            val = spec.get(key)
            if val:
                strings[f"{variant_name}.{key}"] = val
    return strings


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phrase", action="append", default=[], help="Phrase to tokenize. Repeatable.")
    parser.add_argument("--variants-json", default=None,
                         help="Path to a {variant: {prompt_append, extra_negative}} JSON file "
                              "(same format as prompt_experiment.py --variants-json) -- tokenizes every string in it.")
    parser.add_argument("--vocab-check", action="append", default=[], dest="vocab_words",
                         help="Word to check for an exact single-token vocab match. Repeatable.")
    parser.add_argument("--compare", nargs=2, metavar=("PHRASE_A", "PHRASE_B"), default=None,
                         help="Tokenize two phrasings side by side.")
    args = parser.parse_args()

    if not (args.phrase or args.variants_json or args.vocab_words or args.compare):
        parser.error("give at least one of --phrase / --variants-json / --vocab-check / --compare")

    tok = load_tokenizer()
    print(f"Tokenizer: {type(tok).__name__}, vocab_size={tok.vocab_size}\n" + "=" * 70)

    for word in args.vocab_words:
        vocab_check(tok, word)

    for phrase in args.phrase:
        print_phrase(tok, phrase)

    if args.variants_json:
        strings = extract_strings_from_variants(Path(args.variants_json))
        print(f"\n--- {args.variants_json}: {len(strings)} string(s) ---")
        for label, text in strings.items():
            print(f"\n[{label}]")
            print_phrase(tok, text)

    if args.compare:
        a, b = args.compare
        print("\n--- compare ---")
        ids_a, pieces_a = probe_phrase(tok, a)
        ids_b, pieces_b = probe_phrase(tok, b)
        print(f"A {a!r}: {len(ids_a)} tokens -> {pieces_a}")
        print(f"B {b!r}: {len(ids_b)} tokens -> {pieces_b}")


if __name__ == "__main__":
    main()
