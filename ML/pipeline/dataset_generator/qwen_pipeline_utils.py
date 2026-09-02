"""
Shared GPU-aware pipeline loader for the Qwen-Image-2512 scripts.

Qwen-Image-2512's full bf16 pipeline (20B transformer + Qwen2.5-VL text
encoder + VAE) is ~58GB on disk/VRAM. That does NOT fit resident on a 32GB
or 48GB card (confirmed by hitting CUDA OOM on an RTX 6000 Ada 48GB earlier)
-- those need enable_model_cpu_offload(), which keeps only the actively-
running component on GPU and swaps the rest to CPU RAM, and only ever
processes one image at a time (batching would need multiple components
resident simultaneously, defeating the point of offloading).

A 96GB+ card (e.g. RTX PRO 6000 Blackwell) comfortably fits the whole
pipeline resident with ~38GB headroom left over -- enough to both skip
offload entirely (faster: no GPU<->CPU swapping) AND run several images
through one batched forward pass at once (true parallel generation, not
just multiple processes fighting over one GPU -- two full model copies
alone would need ~116GB, more than any single card here has).

This module auto-detects which situation you're in from the ACTUAL GPU
on the current machine, rather than hardcoding an assumption baked in for
whichever card was used when a script was last edited -- so the same
scripts keep working correctly if you move to a different machine again.
"""
import torch
from diffusers import QwenImagePipeline

FULL_RESIDENT_MIN_VRAM_GB = 80  # threshold with real headroom above the ~58GB pipeline size


def load_qwen_pipeline():
    """Returns (pipe, can_batch). can_batch is True only when the full pipeline
    is resident on GPU (offload mode forces batch size to 1)."""
    pipe = QwenImagePipeline.from_pretrained("Qwen/Qwen-Image-2512", torch_dtype=torch.bfloat16)

    props = torch.cuda.get_device_properties(0)
    total_vram_gb = props.total_memory / (1024 ** 3)
    print(f"Detected GPU: {props.name}, {total_vram_gb:.1f}GB total VRAM")

    if total_vram_gb >= FULL_RESIDENT_MIN_VRAM_GB:
        print(f"VRAM >= {FULL_RESIDENT_MIN_VRAM_GB}GB -- loading the full pipeline resident on GPU "
              f"(.to('cuda'), no offload). Parallel batched generation is available.")
        pipe.to("cuda")
        can_batch = True
    else:
        print(f"VRAM < {FULL_RESIDENT_MIN_VRAM_GB}GB -- using enable_model_cpu_offload(). "
              f"Parallel batching is disabled in this mode; generation is one image at a time.")
        pipe.enable_model_cpu_offload()
        can_batch = False

    return pipe, can_batch


def chunked(seq, size):
    """Split seq into consecutive chunks of at most `size` items each -- the last chunk may be smaller."""
    for i in range(0, len(seq), size):
        yield seq[i:i + size]
