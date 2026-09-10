#!/usr/bin/env python3
"""
prompt_experiment.py -- Isolated A/B prompt testing for the synthetic dataset
generator. Runs ONLY on the dedicated vast_test box, never on the main fleet
-- writes to a distinct /data/prompt_test_output/ tree (not
/data/qwen_dataset_output/), so it can never collide with, get pulled into,
or get mistaken for the production dataset or its fleet_monitor.py sync.

Finds N real tasks matching a given row attribute (e.g. facial_hair_style ==
"clean_shaven") via the SAME deterministic task builder the production run
uses, then generates each one twice: once with the unmodified production
prompt/negative_prompt ("baseline"), once with an experimental variant
(stronger positive phrasing and/or an added negative-prompt clause,
"variant") -- a same-seed paired comparison, so any difference in output is
attributable to the prompt change, not random seed luck.

Writes PNGs + a manifest.json (filename -> prompt/negative_prompt/row used)
per condition into its own subfolder, for verify_dataset_vertex.py --manifest
to audit afterward.

Usage (on the test box, inside tmux):
  python3 prompt_experiment.py --attribute facial_hair_style --value clean_shaven \
      --category Men_Grooming --count 10 \
      --prompt-append "The face is completely bare with zero facial hair of any kind." \
      --extra-negative "stubble, beard, mustache, five o'clock shadow, facial hair, goatee"
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import taxonomy as tx
import prompt_builder as pb

OUT_ROOT = Path(os.environ.get("LOOKMAX_TEST_DATA_DIR", str(SCRIPT_DIR / "prompt_test_output")))


def find_matching_tasks(category: str, tier_cycle: list[str], count: int, seed: int,
                         attribute: str | None = None, value: str | None = None,
                         prompt_regex=None, max_search: int = 200000):
    """Deterministically searches increasing seeds for `count` tasks in `category`
    (cycling through `tier_cycle`) that match. Two filter modes (row attribute
    is preferred when the label is a real structured column, e.g. mid_type ==
    "hoodie" or facial_hair_style == "clean_shaven"; prompt_regex is needed for
    anything that's only free text, e.g. face-shape phrases like "double chin"
    which are rendering-time identity diversity, never a label column -- see
    taxonomy.py's IDENTITY MATRIX comment)."""
    found = []
    seed_counter = seed
    tier_idx = 0
    while len(found) < count and (seed_counter - seed) < max_search:
        tier = tier_cycle[tier_idx % len(tier_cycle)]
        rng = random.Random(seed_counter)
        task = pb.build_task(category, tier, rng)
        matched = True
        if attribute is not None:
            matched = matched and task["row"].get(attribute) == value
        if prompt_regex is not None:
            matched = matched and bool(prompt_regex.search(task["prompt"]))
        if matched:
            found.append({"seed": seed_counter, "tier": tier, **task})
        seed_counter += 1
        tier_idx += 1
    return found


def main():
    parser = argparse.ArgumentParser(description="Isolated A/B prompt experiment (test box only).")
    parser.add_argument("--attribute", default=None, help="Row key to filter on, e.g. facial_hair_style or mid_type")
    parser.add_argument("--value", default=None, help="Row value to match, e.g. clean_shaven or hoodie")
    parser.add_argument("--prompt-contains", default=None,
                         help="Case-insensitive regex matched against the built prompt text instead of/in "
                              "addition to --attribute/--value -- needed for identity-matrix text that isn't "
                              "a row column, e.g. 'double chin'.")
    parser.add_argument("--category", required=True, choices=tx.ALL_CATEGORIES)
    parser.add_argument("--tiers", default="polished,average,flaw_mild,flaw_severe",
                         help="Comma-separated tiers to cycle through while searching for matches.")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=90000, help="Search start seed (kept far from the "
                         "production TASK_SEED range so this never coincides with a real dataset index).")
    parser.add_argument("--prompt-append", default="", help="Sentence appended to the positive prompt for the 'variant' condition.")
    parser.add_argument("--extra-negative", default="", help="Comma-separated phrases appended to the negative prompt for the 'variant' condition.")
    parser.add_argument("--variant-name", default="variant")
    parser.add_argument("--skip-baseline", action="store_true", help="Only generate the variant condition, not baseline.")
    parser.add_argument("--variants-json", default=None,
                         help="Path to a JSON file of {variant_name: {\"prompt_append\": str, \"extra_negative\": str}} "
                              "to test several wording strategies against the SAME found task set in one pipeline "
                              "load (much faster than separate --prompt-append runs when comparing many variants). "
                              "Overrides --prompt-append/--extra-negative/--variant-name when given.")
    args = parser.parse_args()

    if not args.attribute and not args.prompt_contains:
        sys.exit("Need --attribute/--value or --prompt-contains to filter on.")

    tier_cycle = [t.strip() for t in args.tiers.split(",")]
    prompt_regex = None
    if args.prompt_contains:
        import re
        prompt_regex = re.compile(args.prompt_contains, re.IGNORECASE)

    filter_desc = " AND ".join(filter(None, [
        f"{args.attribute}=={args.value}" if args.attribute else None,
        f"prompt~={args.prompt_contains!r}" if args.prompt_contains else None,
    ]))
    print("=" * 80)
    print(f"Prompt experiment: {args.category} where {filter_desc}")
    print("=" * 80)
    print(f"Searching for {args.count} matching tasks (seeds starting at {args.seed})...")
    tasks = find_matching_tasks(args.category, tier_cycle, args.count, args.seed,
                                 attribute=args.attribute, value=args.value, prompt_regex=prompt_regex)
    print(f"Found {len(tasks)} matches (seeds: {[t['seed'] for t in tasks]})\n")

    conditions = []
    if not args.skip_baseline:
        conditions.append(("baseline", None, None))
    if args.variants_json:
        with open(args.variants_json) as f:
            variants = json.load(f)
        for name, spec in variants.items():
            append = spec.get("prompt_append", "")
            extra_neg = spec.get("extra_negative", "")
            neg = tx.NEGATIVE_PROMPT + (", " + extra_neg if extra_neg else "")
            conditions.append((name, append or None, neg if extra_neg else None))
    else:
        variant_neg = tx.NEGATIVE_PROMPT + (", " + args.extra_negative if args.extra_negative else "")
        conditions.append((args.variant_name, args.prompt_append, variant_neg if args.extra_negative else None))

    import qwen_pipeline as qp
    pipe, can_batch = qp.load_pipeline()

    for cond_name, prompt_append, negative_prompt in conditions:
        out_dir = OUT_ROOT / cond_name
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest = {}

        print(f"\n--- Condition: {cond_name} ---")
        for t in tasks:
            prompt = t["prompt"]
            if prompt_append:
                prompt = f"{prompt}\n{prompt_append}"
            gen_task = {"prompt": prompt, "resolution": t["resolution"]}
            if negative_prompt:
                gen_task["negative_prompt"] = negative_prompt

            filename = f"{cond_name}_{t['seed']}_{args.category}_{t['tier']}.png"
            images = qp.generate(pipe, [gen_task], [t["seed"]], num_inference_steps=tx.NUM_INFERENCE_STEPS_FULL)
            images[0].save(out_dir / filename)
            manifest[filename] = {
                "prompt": prompt,
                "negative_prompt": negative_prompt or tx.NEGATIVE_PROMPT,
                "category": args.category,
                "tier": t["tier"],
                "seed": t["seed"],
                "row": t["row"],
            }
            print(f"  {filename}")

        with open(out_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"  -> {len(manifest)} images + manifest.json written to {out_dir}")

    qp.unload(pipe)
    print("\nDone. Verify with:")
    for cond_name, _, _ in conditions:
        print(f"  python3 verify_dataset_vertex.py --manifest {OUT_ROOT / cond_name / 'manifest.json'} "
              f"--images-dir {OUT_ROOT / cond_name} --label {cond_name}")


if __name__ == "__main__":
    main()
