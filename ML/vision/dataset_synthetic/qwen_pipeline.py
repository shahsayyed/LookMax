"""
qwen_pipeline.py -- single shared model-loading + generation entry point for
Qwen-Image-2512 (`diffusers` QwenImagePipeline). smoke_test.py,
validation_sweep.py, and full_run.py all call load_pipeline()/generate()
from here so every script that actually touches the GPU shares byte-
identical inference settings (steps, cfg scale, negative prompt source).

`torch` and `diffusers` are imported LAZILY, inside functions, not at
module scope -- smoke_test.py's --dry-run and validation_sweep.py's
--coverage-only modes must run on a machine with no CUDA stack (a laptop,
a CI box) without failing on `import torch` before they ever reach code
that needs a GPU.

GPU-AWARE LOADING (ported from
ML/archive/dataset_generator_v7/qwen_pipeline_utils.py): the full bf16
Qwen-Image-2512 pipeline (20B transformer + Qwen2.5-VL text encoder + VAE)
is ~58GB. A card with less than FULL_RESIDENT_MIN_VRAM_GB cannot hold it
all resident -- it needs enable_model_cpu_offload(), which forces batch
size 1 (only the actively-running component is ever on GPU). A card at or
above the threshold can hold the whole pipeline resident (.to("cuda")),
which is both faster (no GPU<->CPU swap) and technically able to batch --
though see full_run.py's DEFAULT_GEN_BATCH_SIZE comment: on the one card
actually measured (RTX PRO 6000 Blackwell, 96GB), batching gave NO
throughput benefit because the 20B-param transformer at 1024x1024 already
saturates that GPU's compute at batch=1. Don't assume that measurement
holds on a different card -- it's a documented finding, not a law.

NO LORA / NO LIGHTNING DISTILLATION: this module always loads the full
Qwen-Image-2512 checkpoint at NUM_INFERENCE_STEPS_FULL (or _TEST) steps.
Few-step distilled models (e.g. Qwen-Image-Lightning) lose prompt
adherence worst on NEGATIVE/undesirable attributes first -- distillation
pulls generation toward the model's aesthetic mode, and "greasy unwashed
hair" or "deeply wrinkled, stained" is exactly the kind of instruction
that gets silently softened away. That failure mode would quietly destroy
the flaw tier this whole dataset depends on, so it is never wired in here.
"""
import taxonomy as tx

FULL_RESIDENT_MIN_VRAM_GB = 80  # threshold with real headroom above the ~58GB pipeline size

MODEL_ID = "Qwen/Qwen-Image-2512"


def load_pipeline(device=None):
    """Returns (pipe, can_batch). can_batch is True only when the full
    pipeline is resident on GPU (offload mode forces batch size 1).
    `device` can be None, 'cuda', or a specific device e.g. 'cuda:0', 'cuda:1'."""
    import torch
    from diffusers import QwenImagePipeline

    print(f"Loading {MODEL_ID}... this can take a while on first run (~58GB download).")
    pipe = QwenImagePipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)

    if not torch.cuda.is_available():
        print("No CUDA GPU detected -- falling back to CPU. This will be extremely slow "
              "and is only sensible for a tiny sanity check, not a real generation run.")
        return pipe, False

    target_device = device or "cuda"
    device_idx = 0
    if ":" in str(target_device):
        try:
            device_idx = int(str(target_device).split(":")[1])
        except (ValueError, IndexError):
            device_idx = 0

    props = torch.cuda.get_device_properties(device_idx)
    total_vram_gb = props.total_memory / (1024 ** 3)
    print(f"Detected GPU ({target_device}): {props.name}, {total_vram_gb:.1f}GB total VRAM")

    if total_vram_gb >= FULL_RESIDENT_MIN_VRAM_GB:
        print(f"VRAM >= {FULL_RESIDENT_MIN_VRAM_GB}GB -- loading the full pipeline resident on "
              f"GPU (.to('{target_device}'), no offload). Parallel batched generation is available (but see "
              f"full_run.py's DEFAULT_GEN_BATCH_SIZE comment before assuming it helps).")
        pipe.to(target_device)
        can_batch = True
    else:
        print(f"VRAM < {FULL_RESIDENT_MIN_VRAM_GB}GB -- using enable_model_cpu_offload(). "
              f"Parallel batching is disabled in this mode; generation is one image at a time.")
        pipe.enable_model_cpu_offload(device=target_device)
        can_batch = False

    return pipe, can_batch


def chunked(seq, size):
    """Split seq into consecutive chunks of at most `size` items each."""
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def generate(pipe, tasks, seeds, num_inference_steps=None):
    """tasks: list of dicts each with 'prompt' and 'resolution' (w, h) --
    all tasks in one call MUST share the same resolution (the model can't
    mix sizes in one batched forward pass). seeds: list of ints, same
    length as tasks, one per-image deterministic seed (see full_run.py's
    task-list seeding for why these must be stable across runs for resume
    to work). Returns a list of PIL Images, same order as tasks.

    num_inference_steps defaults to taxonomy.NUM_INFERENCE_STEPS_FULL --
    pass taxonomy.NUM_INFERENCE_STEPS_TEST explicitly for quick smoke
    tests. Always full (non-distilled) inference either way -- see module
    docstring on why a Lightning/LoRA few-step path is never used here."""
    import torch

    if not tasks:
        return []
    resolutions = {t["resolution"] for t in tasks}
    if len(resolutions) != 1:
        raise ValueError(f"generate() requires all tasks in one call to share a resolution, got {resolutions}")
    width, height = next(iter(resolutions))

    steps = num_inference_steps or tx.NUM_INFERENCE_STEPS_FULL
    generators = [torch.Generator("cpu").manual_seed(s) for s in seeds]

    result = pipe(
        prompt=[t["prompt"] for t in tasks],
        negative_prompt=tx.NEGATIVE_PROMPT,
        height=height,
        width=width,
        num_inference_steps=steps,
        true_cfg_scale=tx.TRUE_CFG_SCALE,
        generator=generators,
    )
    return result.images


def unload(pipe):
    """Best-effort GPU memory release -- not required for correctness (the
    driving process reclaims everything on exit regardless), but makes
    freed VRAM visible in nvidia-smi immediately, which matters when
    scripts are run back-to-back in the same shell/tmux session."""
    import torch
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
