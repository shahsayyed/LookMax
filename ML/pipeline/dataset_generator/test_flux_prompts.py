import os
# Force HuggingFace to use high-performance transfer internally
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

import torch
from diffusers import FluxPipeline
from pathlib import Path

# ==========================================
# TEST SCRIPT CONFIGURATION
# ==========================================
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("Please set HF_TOKEN environment variable with your HuggingFace token.")

OUTPUT_DIR = Path("test_images")
OUTPUT_DIR.mkdir(exist_ok=True)

# Test Prompts using our exact alignment logic
TEST_PROMPTS = {
    "test_men_grooming.png": "A front-facing head-and-shoulders portrait, perfectly centered face, straight-on eye-level camera angle, looking directly at the camera, bright even studio lighting of a 28-year-old Caucasian man, athletic, perfectly styled textured voluminous hair with pomade, sharp lined-up short beard, flawless clear glowing skin, groomed eyebrows. Photorealistic, ultra detailed, 85mm lens.",
    
    "test_women_grooming.png": "A front-facing head-and-shoulders portrait, perfectly centered face, straight-on eye-level camera angle, looking directly at the camera, bright even studio lighting of a 35-year-old East Asian woman, slim, frizzy highly messy unbrushed hair, uneven poorly blended foundation, smudged eyeliner, visible acne breakouts, dry lips. Photorealistic, ultra detailed, 85mm lens.",
    
    "test_men_outfit.png": "A front-facing full-body portrait, perfectly centered, standing straight, straight-on eye-level camera angle, looking directly at the camera, head to toe visible, bright even studio lighting of a 45-year-old Black man, average build, wearing a severely wrinkled and overly baggy stained grey t-shirt, clashing neon colors, sloppy untucked styling. Photorealistic, ultra detailed, 85mm lens.",
    
    "test_women_outfit.png": "A front-facing full-body portrait, perfectly centered, standing straight, straight-on eye-level camera angle, looking directly at the camera, head to toe visible, bright even studio lighting of a 22-year-old Hispanic woman, curvy, wearing a perfectly tailored elegant outfit, highly flattering silhouette, pristine crisp fabrics, perfectly color coordinated, well balanced proportions. Photorealistic, ultra detailed, 85mm lens."
}

if __name__ == "__main__":
    print("Loading FLUX.1 [dev] for testing... This takes a moment.")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16
    )
    pipe.enable_model_cpu_offload()

    print("Generating 4 test images...")
    
    for filename, prompt in TEST_PROMPTS.items():
        print(f"Generating: {filename}...")
        image = pipe(
            prompt,
            height=1024,
            width=1024,
            guidance_scale=3.5,
            num_inference_steps=28,
            max_sequence_length=512
        ).images[0]
        
        filepath = OUTPUT_DIR / filename
        image.save(filepath)
        print(f"Saved to {filepath}")

    print("✅ Testing complete! Check the 'test_images' folder.")
