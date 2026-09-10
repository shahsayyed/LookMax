#!/usr/bin/env python3
"""
gemini_image_test.py -- One-off cross-model comparison: does the same
clean-shaven prompt-adherence gap we found in Qwen-Image-2512 also show up in
Gemini's image model (gemini-2.5-flash-image), and does prompt-strengthening
fix it there too? Standalone, no GPU needed (Vertex API call), never touches
the production dataset or fleet.

Writes PNGs + manifest.json to gemini_image_test_output/, verifiable with
verify_dataset_vertex.py --manifest the same way as prompt_experiment.py output.
"""
import json
import os
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
CREDENTIALS_FILE = REPO_ROOT / "lookmax-generation-513e3f9ab69e.json"
OUT_DIR = SCRIPT_DIR / "gemini_wording_sweep_output"
MODEL = "gemini-2.5-flash-image"

BASE_PROMPTS = {
    "90005": (
        "A head-and-shoulders portrait photograph of a 55-year-old Hispanic man, heavy set, sharp angular face, "
        "prominent nose, facing the camera directly, looking at the camera.\n"
        "Hair: long hair, lying naturally with no product, neither messy nor styled.\n"
        "Facial hair: {facial_hair}\n"
        "Skin: ordinary skin texture with a few small visible pores.\n"
        "Eyebrows: natural and unshaped, neither groomed nor unkempt.\n"
        "Setting: standing against a plain neutral-colored wall, slightly harsh direct on-camera flash lighting, "
        "centered in frame, slightly low camera angle.\n"
        "Photorealistic candid photograph, natural skin texture, sharp focus, 85mm lens, head and shoulders in frame."
    ),
    "90019": (
        "A head-and-shoulders portrait photograph of a 19-year-old Black man, slim, sharp angular face, "
        "prominent nose, facing the camera directly, looking at the camera.\n"
        "Hair: medium-length hair, visibly greasy and unwashed, strands clumped together with an oily sheen, "
        "flattened against the scalp.\n"
        "Facial hair: {facial_hair}\n"
        "Skin: dry, visibly flaking skin with rough patches on the forehead and around the nose.\n"
        "Eyebrows: thick and completely unshaped, growing together above the nose.\n"
        "Setting: standing in front of a bathroom mirror, a faint reflection visible, soft natural window light "
        "from the side, centered in frame, slightly low camera angle.\n"
        "Photorealistic candid photograph, natural skin texture, sharp focus, 85mm lens, head and shoulders in frame."
    ),
}

SAMPLES_PER_VARIANT = 3

# Same 6 named variants tested on Qwen (clean_shaven_variants.json), adapted to a single
# combined prompt string since Gemini's image model has no separate negative-prompt channel --
# "negative_only" here means the base phrase is left as-is but an explicit avoid-list is added,
# same content as Qwen's negative_only, just folded into the one prompt.
FACIAL_HAIR_VARIANTS = {
    "baseline": "completely clean-shaven with smooth, even skin.",
    "positive_only": (
        "completely clean-shaven with smooth, even skin. The face is completely bare with zero facial hair of "
        "any kind, freshly shaved, smooth jawline and smooth upper lip."
    ),
    "negative_only": (
        "completely clean-shaven with smooth, even skin. Avoid: stubble, beard, mustache, five o'clock shadow, "
        "facial hair, goatee, sideburns, unshaven look."
    ),
    "combined": (
        "completely clean-shaven with smooth, even skin. The face is completely bare with zero facial hair of "
        "any kind, freshly shaved, smooth jawline and smooth upper lip. Avoid: stubble, beard, mustache, "
        "five o'clock shadow, facial hair, goatee, sideburns, unshaven look."
    ),
    "short_direct": "freshly shaved, not a trace of stubble anywhere on the face.",
    "technical_precise": (
        "The jaw, chin, cheeks, and upper lip are entirely free of any facial hair growth, indistinguishable "
        "from a state immediately after shaving, no stubble shadow, no five o'clock shadow, no peach fuzz."
    ),
    "directive_caps": (
        "IMPORTANT: the man's face MUST be perfectly smooth with ABSOLUTELY NO facial hair. No stubble. No "
        "five o'clock shadow. No beard. No mustache. Completely hairless jaw, chin, and cheeks -- as if shaved "
        "moments ago with a fresh razor."
    ),
}


def get_client():
    with open(CREDENTIALS_FILE) as f:
        project_id = json.load(f)["project_id"]
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(CREDENTIALS_FILE.resolve())
    from google import genai
    from google.genai import types
    return genai.Client(vertexai=True, project=project_id, location="global",
                         http_options=types.HttpOptions(timeout=45000))


def generate_one(client, types, prompt, max_retries=6):
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"]),
            )
            for part in resp.candidates[0].content.parts:
                if part.inline_data is not None:
                    return part.inline_data.data
            return None
        except Exception as e:
            err = str(e)
            if ("429" in err or "RESOURCE_EXHAUSTED" in err or "timed out" in err) and attempt < max_retries - 1:
                wait = (attempt + 1) * 20
                print(f"[retry {attempt+1}/{max_retries}, waiting {wait}s: {err[:80]}]", end=" ", flush=True)
                time.sleep(wait)
                continue
            raise
    return None


def main():
    from google.genai import types

    client = get_client()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Preserve any images/manifest already saved from a prior partial run.
    manifest = {}
    if (OUT_DIR / "manifest.json").exists():
        manifest = json.loads((OUT_DIR / "manifest.json").read_text())

    for base_id, template in BASE_PROMPTS.items():
        for variant_name, facial_hair_phrase in FACIAL_HAIR_VARIANTS.items():
            for sample_i in range(1, SAMPLES_PER_VARIANT + 1):
                filename = f"{base_id}_{variant_name}_{sample_i}.png"
                if filename in manifest and (OUT_DIR / filename).exists():
                    print(f"Skipping {filename} (already done)")
                    continue
                prompt = template.format(facial_hair=facial_hair_phrase)
                print(f"Generating {filename} ...", end=" ", flush=True)
                try:
                    data = generate_one(client, types, prompt)
                    if data:
                        (OUT_DIR / filename).write_bytes(data)
                        manifest[filename] = {"prompt": prompt, "category": "Men_Grooming",
                                               "tier": "n/a", "base_id": base_id, "variant": variant_name}
                        with open(OUT_DIR / "manifest.json", "w") as f:
                            json.dump(manifest, f, indent=2)
                        print("OK")
                    else:
                        print("NO IMAGE IN RESPONSE")
                except Exception as e:
                    print(f"FAILED (giving up): {e}")
                time.sleep(15)  # space out requests to stay under the image-gen quota

    with open(OUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n{len(manifest)} images + manifest.json written to {OUT_DIR}")
    print(f"\nVerify with:\n  python3 verify_dataset_vertex.py --manifest {OUT_DIR / 'manifest.json'} "
          f"--images-dir {OUT_DIR} --label gemini_image_test")


if __name__ == "__main__":
    main()
