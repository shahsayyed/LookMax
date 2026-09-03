"""
export_coreml.py -- converts the fine-tuned, pruned model to a CoreML
.mlpackage with INT4 weight quantization and a STATEFUL KV cache, targeting
iOS 18+ (confirmed product decision -- see config.py).

WHY STATEFUL / KV-CACHE, NOT THE BRIEF'S BARE ct.convert(): the original
brief's export snippet converts a plain forward pass with no cache
handling at all. For autoregressive generation, that means every new token
would require re-running the ENTIRE growing prefix from scratch through
the transformer -- for a ~40-60 token response that's O(n^2) work, and the
brief's <80ms target almost certainly will not hold without a KV cache.
This is the single highest-risk part of the whole spec (see the pipeline's
PLAN.md's "Known risks" section) -- more than the training recipe.

APPROACH: wrap the HF model with `transformers.cache_utils.StaticCache`
(fixed-size pre-allocated cache buffers -- the shape export/AOT compilers
including coremltools are designed to trace), expose the cache's key/value
tensors as PyTorch buffers, and hand them to `ct.convert(..., states=[...])`
so CoreML compiles them as persistent ANE-resident state rather than
re-materializing them on every call. This mirrors Apple's documented
stateful-model pattern (WWDC24 "Bring your machine learning and AI models
to Apple silicon"), applied to this specific pruned SmolLM2 checkpoint.

HONESTY ABOUT VERIFICATION: the StaticCache mechanics this wrapper relies
on (early_initialization()'s exact args, per-layer .keys/.values buffer
shapes, and a real two-call prefill-then-decode round trip: 3-token
prefill, then 1 more token continuing from cache_position=3) were verified
directly against the real, downloaded SmolLM2-135M-Instruct weights and
the actual installed transformers 5.16 in this environment -- both calls
ran and produced correctly-shaped logits with the cache buffers genuinely
populated (not a guess at the API). What has NOT been run end-to-end here
is the `torch.jit.trace` + `ct.convert(..., states=...)` step itself (no
fine-tuned checkpoint exists yet -- see PLAN.md), which is the part of
stateful HF-to-CoreML export most likely to need a few rounds of
trace-error iteration even for experienced practitioners, independent of
whether the underlying cache logic is correct. Budget time for that, and
verify actual per-token latency on a real device against the 80ms target
once conversion succeeds.

QUANTIZATION API CORRECTED FROM THE BRIEF: the brief calls
`coremltools.optimize.torch.quantization.linear_quantize_weights` (a
PRE-conversion, torch-side API) on an already-converted `.mlpackage`,
which is the wrong stage for that function. The correct POST-conversion
API, confirmed present in the installed coremltools 9.0, is
`coremltools.optimize.coreml.linear_quantize_weights`, used below.

MODEL SIZE, CORRECTED FROM THE BRIEF: SmolLM2-135M-Instruct ties its
embedding and lm_head weights (confirmed via AutoConfig.tie_word_embeddings
== True), so vocabulary pruning saves ONE ~26M-param matrix, not two.
Post-prune the model is ~109M params, not the brief's 85-90M estimate.
At INT4 (~0.5 bytes/param) that's ~54MB of weight storage alone, before
CoreML packaging overhead -- likely landing around 55-60MB total, not the
brief's <45MB target. This is a real, quantified gap worth a product
decision (accept ~55-60MB, try a smaller base model, or ship this model
as a post-install download rather than embedded in the initial .ipa)
rather than something to silently paper over.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg


def _build_stateful_wrapper(hf_model, max_context_tokens):
    import torch
    from transformers.cache_utils import StaticCache

    class StatefulStylistModel(torch.nn.Module):
        """Wraps the fine-tuned causal LM with a fixed-size StaticCache so
        the K/V buffers can be exposed to coremltools as persistent state
        (ct.StateType) instead of being re-created on every forward call."""

        def __init__(self, model, max_len):
            super().__init__()
            self.model = model
            # StaticCache's __init__ only takes config/max_cache_len in the
            # transformers version this was verified against (5.16) -- the
            # per-layer key/value tensors are allocated lazily and must be
            # forced into existence via early_initialization() before they
            # can be registered as buffers (confirmed by direct
            # instantiation against SmolLM2's config: layers start with
            # keys/values == None until this call).
            self.cache = StaticCache(config=model.config, max_cache_len=max_len)
            num_kv_heads = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
            head_dim = getattr(model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads)
            self.cache.early_initialization(
                batch_size=1, num_heads=num_kv_heads, head_dim=head_dim,
                dtype=torch.float32, device=torch.device("cpu"),
            )
            # Expose the cache's per-layer tensors as named buffers so
            # ct.convert's `states=` argument can find them by name.
            for i, layer in enumerate(self.cache.layers):
                self.register_buffer(f"k_cache_{i}", layer.keys, persistent=False)
                self.register_buffer(f"v_cache_{i}", layer.values, persistent=False)
            self.register_buffer("cache_position", torch.zeros(1, dtype=torch.long), persistent=False)

        def forward(self, input_ids):
            seq_len = input_ids.shape[-1]
            position_ids = self.cache_position + torch.arange(seq_len, dtype=torch.long)
            out = self.model(
                input_ids=input_ids,
                past_key_values=self.cache,
                use_cache=True,
                cache_position=position_ids,
            )
            self.cache_position += seq_len
            return out.logits

        def state_buffer_names(self):
            names = [f"k_cache_{i}" for i in range(len(self.cache.layers))]
            names += [f"v_cache_{i}" for i in range(len(self.cache.layers))]
            names.append("cache_position")
            return names

    return StatefulStylistModel(hf_model, max_context_tokens)


def run_export(finetuned_dir, output_path):
    import torch
    import coremltools as ct
    from transformers import AutoModelForCausalLM

    print(f"Loading fine-tuned model from {finetuned_dir}...")
    model = AutoModelForCausalLM.from_pretrained(finetuned_dir)
    model.eval()

    # The StaticCache needs room for the FULL sequence (input prompt +
    # every generated token), not just the input bound -- see config.py's
    # MAX_TOTAL_TOKENS comment for why this was measured and corrected up
    # from the brief's single 128-token figure.
    wrapper = _build_stateful_wrapper(model, cfg.MAX_TOTAL_TOKENS)
    wrapper.eval()

    print("Tracing (TorchScript)...")
    example_input = torch.zeros((1, 1), dtype=torch.long)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example_input, strict=False)

    print(f"Converting to CoreML (minimum_deployment_target={cfg.IOS_MIN_DEPLOYMENT})...")
    states = [
        ct.StateType(
            wrapped_type=ct.TensorType(shape=getattr(wrapper, name).shape),
            name=name,
        )
        for name in wrapper.state_buffer_names()
    ]
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input_ids", shape=(1, ct.RangeDim(1, cfg.MAX_INPUT_TOKENS)), dtype=int)],
        states=states,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=getattr(ct.target, cfg.IOS_MIN_DEPLOYMENT),
        convert_to="mlprogram",
    )

    print(f"Quantizing weights to {cfg.QUANT_DTYPE.upper()} ({cfg.QUANT_MODE})...")
    quant_config = ct.optimize.coreml.OptimizationConfig(
        global_config=ct.optimize.coreml.OpLinearQuantizerConfig(
            mode=cfg.QUANT_MODE, dtype=cfg.QUANT_DTYPE,
        )
    )
    quantized = ct.optimize.coreml.linear_quantize_weights(mlmodel, config=quant_config)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    quantized.save(str(output_path))

    size_mb = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file()) / (1024 ** 2)
    print(f"\nSaved {output_path} ({size_mb:.1f} MB)")
    if size_mb > 45:
        print(f"⚠ {size_mb:.1f}MB exceeds the brief's <45MB target -- see this file's module docstring "
              f"for why that target was optimistic given SmolLM2-135M-Instruct's tied embeddings. This is "
              f"expected, not a bug in this export.")
    print("\nBefore shipping: verify actual per-token latency on a real iOS 18 device against the "
          f"{cfg.TARGET_LATENCY_MS}ms target, and confirm the StaticCache-based stateful export round-trips "
          "correctly through Xcode's model preview / a real CoreML runtime call, not just conversion success.")


def run_dry_run():
    print("export_coreml.py has no --dry-run mode of its own -- there is nothing meaningful to preview "
          "without a real fine-tuned checkpoint (unlike the earlier pipeline stages, there's no taxonomy/"
          "prompt text to print here, only a model conversion). Use finetune.py --dry-run and "
          "prune_vocabulary.py --dry-run to sanity-check the stages that feed this one.")


def main():
    parser = argparse.ArgumentParser(description="Export the fine-tuned stylist model to a stateful, INT4 CoreML package.")
    parser.add_argument("finetuned_dir", nargs="?", default=str(cfg.CHECKPOINTS_DIR / "finetune_run" / "final"))
    parser.add_argument("--output", default=str(cfg.MODELS_DIR / "StylistEngine.mlpackage"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        run_dry_run()
        return

    if not Path(args.finetuned_dir).exists():
        sys.exit(f"!! {args.finetuned_dir} not found. Run finetune.py first.")

    run_export(args.finetuned_dir, args.output)


if __name__ == "__main__":
    main()
