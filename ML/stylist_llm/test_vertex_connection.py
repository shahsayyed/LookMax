"""
test_vertex_connection.py -- Self-check script to verify Google Cloud Vertex AI / Agent Platform
connectivity, service account authentication, and Gemini 3.8 Flash response formatting
before initiating full dataset generation.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

# Ensure module imports from ML/stylist_llm and ML/vision
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "vision" / "dataset_synthetic"))

import config as cfg
import tag_vocabulary as tv
import taxonomy as vision_tx
import prompt_builder as vision_pb
import qa_review as qa

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.genai")

_META_CHATTER_OPENERS = (
    "sure", "here's", "here is", "as a stylist", "certainly", "of course",
    "i'd suggest", "i would suggest", "great question", "absolutely",
)


def validate_stylist_response(text, score=None):
    """Validates response against LookMax quality rules and effort-vs-genetics guardrails."""
    stripped = text.strip()
    if not stripped:
        return False, "empty response"

    lower = stripped.lower()
    if any(lower.startswith(opener) for opener in _META_CHATTER_OPENERS):
        return False, f"meta-chatter opener: '{stripped[:30]}...'"

    banned_match = qa._BANNED_RE.search(stripped)
    if banned_match:
        return False, f"forbidden genetics/body-shape phrase: '{banned_match.group(0)}'"

    unobs_match = qa._UNOBSERVABLE_RE.search(stripped)
    if unobs_match:
        return False, f"unobservable item mentioned: '{unobs_match.group(0)}'"

    word_count = len(stripped.split())
    if word_count < cfg.MIN_RESPONSE_WORDS:
        return False, f"too short ({word_count} words, need >= {cfg.MIN_RESPONSE_WORDS})"
    if word_count > cfg.MAX_RESPONSE_WORDS:
        return False, f"too long ({word_count} words, need <= {cfg.MAX_RESPONSE_WORDS})"

    lines = [l.strip() for l in stripped.splitlines() if l.strip()]
    bullet_lines = [l for l in lines if l.startswith("- ") or l.startswith("• ") or l.startswith("* ")]
    if len(bullet_lines) != len(lines):
        return False, f"non-checklist format ({len(lines) - len(bullet_lines)} lines lack bullet prefix)"

    if len(lines) < 2 or len(lines) > 5:
        return False, f"bullet count out of range ({len(lines)} lines, need 2-5)"

    for idx, l in enumerate(lines):
        line_words = len(l.split())
        if line_words > 16:
            return False, f"line {idx+1} too long ({line_words} words, need <= 15)"

    if score is not None:
        try:
            s = float(score)
            if s >= 9.0 and len(lines) > 3:
                return False, f"high score ({s:.1f}) should have 2-3 polish tips, got {len(lines)}"
            elif s < 9.0 and len(lines) < 3:
                return False, f"lower score ({s:.1f}) should have 3-5 fixes, got {len(lines)}"
        except (ValueError, TypeError):
            pass

    return True, "Passed all LookMax guardrails"


def main():
    parser = argparse.ArgumentParser(description="Test Vertex AI / Agent Platform connection and credentials.")
    parser.add_argument(
        "--credentials",
        default="lookmax-generation-513e3f9ab69e.json",
        help="Path to Google Cloud service account JSON key (default lookmax-generation-513e3f9ab69e.json)."
    )
    parser.add_argument(
        "--model",
        default="gemini-3.8-flash",
        help="Model ID to test (default gemini-3.8-flash)."
    )
    parser.add_argument(
        "--location",
        default="global",
        help="Google Cloud Vertex AI location (default global)."
    )
    args = parser.parse_args()

    creds_path = Path(args.credentials)
    if not creds_path.is_absolute():
        creds_path = cfg.REPO_ROOT / creds_path

    print("=" * 80)
    print("VERTEX AI / AGENT PLATFORM CONNECTION TEST")
    print("=" * 80)

    # 1. Verify Credentials file
    if not creds_path.exists():
        print(f"❌ Credentials file not found: {creds_path}")
        sys.exit(1)

    try:
        with open(creds_path, "r") as f:
            creds_data = json.load(f)
    except Exception as e:
        print(f"❌ Failed to read credentials JSON: {e}")
        sys.exit(1)

    project_id = creds_data.get("project_id")
    client_email = creds_data.get("client_email")

    print(f"✓ Found Service Account Key: {creds_path.name}")
    print(f"  • Project ID:   {project_id}")
    print(f"  • Client Email: {client_email}")
    print(f"  • Location:     {args.location}")
    print(f"  • Model:        {args.model}")

    # Set environment variable for Google Cloud SDK
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(creds_path.resolve())

    # 2. Initialize genai Client with Vertex AI
    print("\n[1/3] Initializing google-genai Client (Vertex AI mode)...")
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(
            vertexai=True,
            project=project_id,
            location=args.location,
        )
        print("✓ Client initialized successfully.")
    except Exception as e:
        print(f"❌ Failed to initialize genai.Client: {e}")
        sys.exit(1)

    # 3. Build a realistic test context
    print("\n[2/3] Sampling a test context with Apple Vision signals...")
    import random
    rng = random.Random(42)
    sample_cat = "Men_Outfit"
    sample_tier = "flaw_mild"
    row = vision_pb.build_task(sample_cat, sample_tier, rng)["row"]
    row["posture"] = "slight_slouch"
    row["lighting"] = "well_lit"
    row["color_harmony"] = "clashing_tones"
    sample_prompt = tv.format_tag_prompt(sample_cat, "Casual Hangout", row)
    score = row.get("score")

    print(f"Sample Context: {sample_cat} | {sample_tier} | Score: {score}")
    print("Input prompt:")
    for line in sample_prompt.splitlines():
        print(f"  {line}")

    # 4. Dispatch call to Vertex AI
    print(f"\n[3/3] Sending generate_content request to '{args.model}'...")
    t0 = time.time()
    try:
        response = client.models.generate_content(
            model=args.model,
            contents=sample_prompt,
            config=types.GenerateContentConfig(
                system_instruction=cfg.SYSTEM_PROMPT,
                temperature=cfg.GEMINI_TEMPERATURE,
                max_output_tokens=cfg.GEMINI_GENERATION_MAX_OUTPUT_TOKENS,
            )
        )
        elapsed = time.time() - t0
    except Exception as e:
        print(f"❌ Generation failed ({type(e).__name__}): {e}")
        print("\nTip: If location unsupported, try passing --location global or another region.")
        sys.exit(1)

    text = response.text or ""
    print(f"✓ Response received in {elapsed:.2f}s:")
    print("-" * 50)
    print(text)
    print("-" * 50)

    # 5. Quality Validation
    ok, reason = validate_stylist_response(text, score=score)
    word_count = len(text.split())
    bullet_count = len([l for l in text.splitlines() if l.strip().startswith(("- ", "• ", "* "))])

    print("\nQuality & Guardrail Check:")
    print(f"  • Bullet count: {bullet_count}")
    print(f"  • Word count:   {word_count} words")
    print(f"  • Validation:   {'✓ ' if ok else '✗ '}{reason}")

    if not ok:
        print("\n⚠️ Note: The format or length did not pass the strict check on the first try.")
        print("   The full generator automatically retries up to 5 times per context.")
    else:
        print("\n🎉 Connection, authentication, and output formatting verified successfully!")


if __name__ == "__main__":
    main()
