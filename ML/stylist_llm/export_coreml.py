"""
export_coreml.py -- converts the fine-tuned, pruned model to a CoreML
.mlpackage targeting iOS 18+ (confirmed product decision -- see config.py).
Quantization level is config.py's QUANT_DTYPE (currently "none", i.e. FP16
with no separate weight-quantization step) or the --quant CLI flag -- see
"MODEL SIZE" below for why FP16 is the current default despite being the
largest option.

STATELESS, NOT A STATEFUL KV-CACHE: an earlier version of this file wrapped
the model in a `transformers.cache_utils.StaticCache` and exported it as a
stateful CoreML model (`ct.StateType`) so each new token's forward pass
could reuse cached keys/values instead of recomputing the whole growing
prefix. That approach was abandoned after exhausting two independent PyTorch
export paths, both hitting confirmed, narrow upstream bugs rather than
something fixable here:
  - `torch.jit.trace`: its tracer is value-based (it records what happened
    during ONE concrete execution), and repeatedly either dropped or
    conflated per-layer cache-position buffers that happened to share a
    value during that one trace, no matter how the state was structured.
  - `torch.export`: traces correctly (buffer identity is tracked
    structurally, not by value, which fixed the jit.trace issue), but
    CoreML's automatic state derivation for torch.export models depends on
    `graph_signature.buffers_to_mutate`, which is only populated by
    `ExportedProgram.run_decompositions()` -- and that call itself crashes
    for this specific model ("expected compiled_fn to be GraphModule, got
    <class 'function'>", deep in PyTorch's AOT-autograd joint-graph export
    path) on both the installed torch 2.13.0 and torch 2.7.0 (coremltools'
    own last-tested version, ruling out a version-skew explanation) --
    confirmed to be triggered specifically by this model's mutable
    StaticCache buffers via a minimal reproduction, not by the model's size
    or complexity.

Given that, this file recomputes the whole growing sequence (prompt +
tokens generated so far) on every forward call instead of maintaining any
persistent state -- no buffers to mutate, no StaticCache, no `ct.StateType`,
and therefore none of the above. This is O(n^2) in total compute across a
full generation rather than O(n), but for this model's short sequences (a
system+tags prompt bounded at MAX_INPUT_TOKENS, plus at most
MAX_RESPONSE_WORDS-ish generated tokens, all well under MAX_TOTAL_TOKENS)
on a ~106M-param model, that is expected to still be fast -- verify actual
per-token latency on a real device against TARGET_LATENCY_MS regardless,
same as any export would require.

QUANTIZATION API CORRECTED FROM THE BRIEF: the brief calls
`coremltools.optimize.torch.quantization.linear_quantize_weights` (a
PRE-conversion, torch-side API) on an already-converted `.mlpackage`,
which is the wrong stage for that function. The correct POST-conversion
API, confirmed present in the installed coremltools 9.0, is
`coremltools.optimize.coreml.linear_quantize_weights`, used below.

MODEL SIZE, CORRECTED FROM THE BRIEF: SmolLM2-135M-Instruct ties its
embedding and lm_head weights (confirmed via AutoConfig.tie_word_embeddings
== True), so vocabulary pruning saves ONE ~26M-param matrix, not two.
Post-prune the model is ~108M params, not the brief's 85-90M estimate. Real
measured sizes at each quantization level (checkpoint is FP32 PyTorch,
415MB, for reference): FP16 (--quant none) 207MB, INT8 104MB, INT4 52MB --
none within the brief's <45MB target, a gap already known before
quantization was even considered.

FP16 IS THE CURRENT DEFAULT, NOT THE SMALLEST OPTION: confirmed directly via
a 20-prompt side-by-side comparison (checkpoint vs. each export, identical
greedy decoding) that INT4 causes real quality regressions on this
model -- responses that trail off before naturally stopping, and in the
worst case a token-repetition loop ("...define your buzz buzz buzz buzz
buzz...", 53 repeats) -- while FP16 matches the FP32 checkpoint exactly on
13/20 prompts and is fully coherent on all 20. This is specifically an
INT4-quantization effect, not a flaw in the stateless export architecture
above: the same architecture at FP16 is clean. Revisit INT8/INT4 as a
separate size-optimization pass once shipped, each re-verified with the
same comparison before trusting it -- don't assume quality carries over
from this baseline. A no-repeat n-gram guard in the generation loop (no
re-export needed) would likely fix the repetition-loop failure mode
specifically and is worth trying before writing off INT4 entirely.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg


def _build_stateless_wrapper(hf_model):
    import torch

    class StatelessStylistModel(torch.nn.Module):
        """Plain forward pass, no cache: the caller feeds the WHOLE sequence
        (prompt + every token generated so far) on every call and reads the
        last position's logits for the next token. See this file's module
        docstring for why there's no persistent KV-cache state here."""

        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids):
            return self.model(input_ids=input_ids, use_cache=False).logits

    return StatelessStylistModel(hf_model)


def _patch_coremltools_numpy2_cast_bug():
    """coremltools 9.0's torch->MIL `_cast` op handler (used for the `int`/`bool`
    torch ops) does `dtype(x.val)` on a length-1 numpy array, relying on NumPy's
    pre-2.0 behavior of silently converting a single-element array to a Python
    scalar. NumPy >=2.0 removed that implicit conversion for anything but a true
    0-dimensional array, so this raises `TypeError: only 0-dimensional arrays can
    be converted to Python scalars` -- confirmed directly (`int(np.array([3]))`
    fails, `int(np.array([3]).item())` works) and confirmed there is no newer
    coremltools release yet (9.0 is latest as of this pipeline's build). Patched
    here (not by downgrading the repo's shared numpy, which every other ML
    pipeline also depends on) by squeezing to a true scalar before the dtype
    conversion -- otherwise identical to the original."""
    import numpy as np
    from coremltools.converters.mil.frontend.torch import ops as _torch_ops

    def _cast(context, node, dtype, dtype_name):
        inputs = _torch_ops._get_inputs(context, node, expected=1)
        x = inputs[0]
        if not (len(x.shape) == 0 or np.all([d == 1 for d in x.shape])):
            raise ValueError("input to cast must be either a scalar or a length 1 tensor")
        if x.can_be_folded_to_const():
            val = x.val
            if hasattr(val, "ndim") and val.ndim > 0:
                val = val.item()
            res = x if isinstance(val, dtype) else _torch_ops.mb.const(val=dtype(val), name=node.name)
        elif len(x.shape) > 0:
            x = _torch_ops.mb.squeeze(x=x, name=node.name + "_item")
            res = _torch_ops.mb.cast(x=x, dtype=dtype_name, name=node.name)
        else:
            res = _torch_ops.mb.cast(x=x, dtype=dtype_name, name=node.name)
        context.add(res, node.name)

    _torch_ops._cast = _cast


def _patch_coremltools_mixed_shape_view_bug():
    """coremltools 9.0's `view`/`reshape` converter (used for the very common
    `tensor.view(bsz, seq_len, -1, head_dim)`-style reshape in HF's attention code)
    only has a working path for a shape list where EVERY dimension is a rank-0 Var
    (see its `isinstance(shape, list) and all(isinstance(dim, Var) and len(dim.shape)
    == 0 for dim in shape)` check). HF's reshape here mixes dynamically-traced dims
    (bsz, seq_len -- real Vars) with plain Python int literals (-1 resolved elsewhere,
    head_dim), so that all-Var check fails, the list falls through un-normalized, and
    the subsequent `mb.cast(x=shape, dtype="int32")` on a raw mixed list fails with
    "Cannot add const [Var, Var, Var, Var]" (confirmed directly against this export --
    `_add_const` can't fold a list containing genuine non-constant Var entries).
    Patched by normalizing EVERY entry (Var or plain int) to a rank-0 int32 Var before
    concatenating, which is otherwise identical to the original all-Var branch's logic.
    Applied via the torch-op registry directly (not by reassigning the module-level
    `view` name), since dispatch in convert_single_node looks up
    `_TORCH_OPS_REGISTRY.get_func(...)`, not the ops module's own attributes."""
    import numpy as np
    from coremltools.converters.mil.frontend.torch import ops as _torch_ops
    from coremltools.converters.mil.frontend.torch.torch_op_registry import _TORCH_OPS_REGISTRY
    from coremltools.converters.mil.mil import types as _types

    mb, Var, ListVar = _torch_ops.mb, _torch_ops.Var, _torch_ops.ListVar

    def view(context, node):
        inputs = _torch_ops._get_inputs(context, node, expected=2)
        x = inputs[0]
        shape = inputs[1]

        if isinstance(shape, Var) and np.prod(shape.shape) == 0:
            assert np.prod(x.shape) <= 1, "Reshape to empty shape works only for scalar and single-element tensor"
            context.add(mb.identity(x=x, name=node.name))
            return

        if isinstance(shape, ListVar):
            length = mb.list_length(ls=shape)
            indices = mb.range_1d(start=0, end=length, step=1)
            shape = mb.list_gather(ls=shape, indices=indices)

        if isinstance(shape, list):
            normalized = []
            for size in shape:
                if isinstance(size, Var):
                    # Some dims (e.g. derived from a dynamic input's .size(-1)) trace as
                    # rank-1 length-1 tensors, not true rank-0 scalars -- squeeze first so
                    # every entry has matching rank before concatenating (mirrors
                    # coremltools' own `_try_whole_slice` defensive squeeze elsewhere).
                    if len(size.shape) > 0:
                        size = mb.squeeze(x=size)
                    normalized.append(size if size.dtype == _types.int32 else mb.cast(x=size, dtype="int32"))
                else:
                    normalized.append(mb.const(val=np.array(int(size), dtype=np.int32)))
            shape = mb.concat(values=normalized, axis=0)

        shape = mb.cast(x=shape, dtype="int32")
        if _types.is_complex(x.dtype):
            real, imag = (
                mb.reshape(x=part, shape=shape, name=node.name)
                for part in (mb.complex_real(data=x), mb.complex_imag(data=x))
            )
            view = mb.complex(real_data=real, imag_data=imag, name=node.name)
        else:
            view = mb.reshape(x=x, shape=shape, name=node.name)
        context.add(view)

    for name in ("view", "view_copy", "_unsafe_view", "reshape"):
        _TORCH_OPS_REGISTRY.set_func_by_name(view, name)


def _patch_coremltools_to_op_overloads():
    """coremltools 9.0's torch->MIL frontend loader (internal_graph.py's from_exir_node)
    only recognizes the `to.dtype` and `_to_copy` ATen overloads of the `to()` cast op
    for the torch.export/EXIR frontend -- confirmed directly, one overload at a time
    ("NotImplementedError: Unsupported fx node to, kind to.dtype_layout", then the same
    for "to.device") against real nodes this export produces. `torch.ops.aten.to` has 7
    overloads total (device, dtype, other, dtype_layout, prim_Device, prim_dtype,
    prim_other; confirmed via `torch.ops.aten.to.overloads()`), all keyword-argument
    variants of the same cast; the existing `to()` handler already resolves `dtype` via
    a keyword-argument lookup (`_parse_keyword_args`) that doesn't depend on which
    overload name triggered it, so registering every overload name against the SAME
    handler up front avoids finding the rest one crash at a time."""
    from coremltools.converters.mil.frontend.torch.torch_op_registry import _TORCH_OPS_REGISTRY

    to_func = _TORCH_OPS_REGISTRY.get_func("to")
    for overload in ("device", "dtype", "other", "dtype_layout", "prim_Device", "prim_dtype", "prim_other"):
        _TORCH_OPS_REGISTRY.set_func_by_name(to_func, f"to.{overload}")


def _patch_coremltools_alias_op():
    """coremltools 9.0 registers a no-op passthrough handler for `alias_copy` (and
    clone/detach/contiguous/etc., all under its `noop` function) but not the plain
    `alias` op -- confirmed both directly ("NotImplementedError: Unsupported fx node
    alias, kind alias" against a real node this export produces) and by inspecting the
    registration list itself (`alias_copy` present, `alias` absent) -- an evident
    oversight, since `alias` is exactly the same no-op-view case `alias_copy` already
    covers. Registering it against the existing `noop` handler."""
    from coremltools.converters.mil.frontend.torch.torch_op_registry import _TORCH_OPS_REGISTRY

    _TORCH_OPS_REGISTRY.set_func_by_name(_TORCH_OPS_REGISTRY.get_func("alias_copy"), "alias")


def _patch_coremltools_diff_op():
    """coremltools 9.0 has no converter at all for `aten::diff` (confirmed: absent from
    _TORCH_OPS_REGISTRY.name_to_func_mapping, and "NotImplementedError: Unsupported fx
    node diff, kind diff" against a real node this export produces -- transformers'
    dynamic-sequence-length attention/mask code calls it once this export's `dynamic_shapes`
    makes the sequence dimension genuinely variable, rather than the fixed length the
    earlier attempts at this export used). `diff(x, n=1, dim=-1, prepend=None, append=None)`
    is `x[1:] - x[:-1]` along `dim`, computed after first concatenating `prepend`/`append`
    onto `x` along that same dim if given. Only n=1 is implemented (confirmed sufficient:
    a real node from this export needed prepend support, at which point this was extended
    to cover it -- see the prepend/append concat below); a non-1 n raises clearly rather
    than silently computing the wrong thing."""
    from coremltools.converters.mil.frontend.torch import ops as _torch_ops
    from coremltools.converters.mil.frontend.torch.torch_op_registry import _TORCH_OPS_REGISTRY

    mb = _torch_ops.mb

    def diff(context, node):
        inputs = _torch_ops._get_inputs(context, node)
        x = inputs[0]
        n = inputs[1].val if len(inputs) > 1 and inputs[1] is not None else 1
        dim = inputs[2].val if len(inputs) > 2 and inputs[2] is not None else -1
        prepend = inputs[3] if len(inputs) > 3 else None
        append = inputs[4] if len(inputs) > 4 else None
        if n != 1:
            raise NotImplementedError(f"diff: only n=1 is implemented (got n={n})")
        rank = x.rank
        axis = dim if dim >= 0 else rank + dim

        parts = [p for p in (prepend, x, append) if p is not None]
        if len(parts) > 1:
            x = mb.concat(values=parts, axis=axis)

        begin1 = [0] * rank
        begin1[axis] = 1
        begin_mask1 = [True] * rank
        begin_mask1[axis] = False
        x1 = mb.slice_by_index(x=x, begin=begin1, end=[0] * rank, begin_mask=begin_mask1, end_mask=[True] * rank)

        end2 = [0] * rank
        end2[axis] = -1
        end_mask2 = [True] * rank
        end_mask2[axis] = False
        x2 = mb.slice_by_index(x=x, begin=[0] * rank, end=end2, begin_mask=[True] * rank, end_mask=end_mask2)

        res = mb.sub(x=x1, y=x2, name=node.name)
        context.add(res)

    _TORCH_OPS_REGISTRY.set_func_by_name(diff, "diff")


def _patch_coremltools_new_ones_op():
    """coremltools 9.0 has a converter for `aten::new_full` (`tensor.new_full(size,
    fill_value)`) but not `aten::new_ones` (`tensor.new_ones(size)`, i.e. the same thing
    with fill_value implicitly 1) -- confirmed both directly ("NotImplementedError:
    Unsupported fx node new_ones, kind new_ones" against a real node this export
    produces) and by inspecting the registration list (new_full and new_zeros present,
    new_ones absent). Implemented as new_full's exact logic with val=1 fixed, reusing
    its own `_make_fill_op` helper."""
    from coremltools.converters.mil.frontend.torch import ops as _torch_ops
    from coremltools.converters.mil.frontend.torch.torch_op_registry import _TORCH_OPS_REGISTRY

    def new_ones(context, node):
        inputs = _torch_ops._get_inputs(context, node)
        size = inputs[1]
        result = _torch_ops._make_fill_op(size, 1, node.name)
        context.add(result)

    _TORCH_OPS_REGISTRY.set_func_by_name(new_ones, "new_ones")


def _patch_coremltools_bitwise_and_mixed_dtype():
    """coremltools 9.0's `bitwise_and` converter only delegates to `logical_and` when
    BOTH inputs are already bool -- confirmed directly ("The `bitwise_and` op only
    supports boolean input, but get [int, bool]" against a real node from this export's
    mask-combination logic) and by inspecting its source (`if all(types.is_bool(...) for
    ...): logical_and(...) else: raise`). `logical_and`'s own implementation already
    casts both operands to bool unconditionally (`mb.cast(x=x, dtype="bool")`), so it
    already handles a mixed int/bool pair correctly -- `bitwise_and`'s stricter check is
    an unnecessary refusal, not a genuine limitation of the op it delegates to. Extended
    to also delegate whenever AT LEAST ONE input is bool (the only shape a boolean
    attention/causal mask combined with an int tensor can take, and the treat-nonzero-
    as-true semantics logical_and applies are exactly what mask code intends here) --
    an all-int pair (true bitwise arithmetic, not boolean masking) still raises, exactly
    as before, since that was never supported and isn't what this model needs."""
    from coremltools.converters.mil.frontend.torch import ops as _torch_ops
    from coremltools.converters.mil.frontend.torch.torch_op_registry import _TORCH_OPS_REGISTRY
    from coremltools.converters.mil.mil import types as _types

    def bitwise_and(context, node):
        inputs = _torch_ops._get_inputs(context, node)
        if any(_types.is_bool(i.dtype) for i in inputs):
            _torch_ops.logical_and(context, node)
        else:
            raise NotImplementedError(
                f"The `bitwise_and` op only supports boolean input, but got {[i.dtype for i in inputs]}."
            )

    for name in ("bitwise_and", "and"):
        _TORCH_OPS_REGISTRY.set_func_by_name(bitwise_and, name)


def _patch_coremltools_inplace_op_dispatch_for_exir():
    """coremltools 9.0's torch->MIL frontend loader (internal_graph.py's from_exir_node)
    checks `kind not in _TORCH_OPS_REGISTRY` using the registry's `__contains__`, which
    does a bare dict lookup with NO in-place/functional normalization -- unlike
    `TorchOpsRegistry.get_func()` (used everywhere op dispatch actually HAPPENS), which
    correctly strips a trailing `_` first (`unify_inplace_and_functional`: "sub_" ->
    "sub"). So an in-place op reaching the exir loader by its raw ATen name fails the
    registry CONTAINMENT check even though a handler for its functional form is already
    registered -- confirmed directly against an earlier (stateful) version of this
    export ("NotImplementedError: Unsupported fx node add_, kind add_"). Kept as a
    defensive fix even though the current stateless model has no buffer mutations of
    its own to produce such nodes -- HF's model code may still emit an in-place op
    coremltools only registered functionally, and this costs nothing if unused."""
    from coremltools.converters.mil.frontend.torch.torch_op_registry import TorchOpsRegistry
    from coremltools.converters.mil.frontend.torch.utils import sanitize_op_kind, unify_inplace_and_functional

    def __contains__(self, key):
        return unify_inplace_and_functional(sanitize_op_kind(key)) in self.name_to_func_mapping

    TorchOpsRegistry.__contains__ = __contains__


def run_export(finetuned_dir, output_path, quant=None):
    import torch
    import coremltools as ct
    from transformers import AutoModelForCausalLM

    quant = quant or cfg.QUANT_DTYPE

    _patch_coremltools_numpy2_cast_bug()
    _patch_coremltools_mixed_shape_view_bug()
    _patch_coremltools_to_op_overloads()
    _patch_coremltools_inplace_op_dispatch_for_exir()
    _patch_coremltools_alias_op()
    _patch_coremltools_diff_op()
    _patch_coremltools_new_ones_op()
    _patch_coremltools_bitwise_and_mixed_dtype()

    print(f"Loading fine-tuned model from {finetuned_dir}...")
    model = AutoModelForCausalLM.from_pretrained(finetuned_dir)
    model.eval()

    wrapper = _build_stateless_wrapper(model)
    wrapper.eval()

    print("Exporting (torch.export)...")
    # A representative, non-trivial length for the initial trace -- torch.export
    # traces from concrete example values but the `dynamic_shapes` declaration below
    # is what actually makes the exported program accept any length in [1, MAX_TOTAL_TOKENS],
    # not this specific example.
    example_input = torch.zeros((1, 8), dtype=torch.long)
    seq_len_dim = torch.export.Dim("seq_len", min=1, max=cfg.MAX_TOTAL_TOKENS)
    with torch.no_grad():
        exported = torch.export.export(
            wrapper, (example_input,), dynamic_shapes={"input_ids": {1: seq_len_dim}},
        )
        # ct.convert only accepts ATEN or EDGE dialect, not the TRAINING dialect
        # torch.export.export() produces by default -- unlike the abandoned stateful
        # version of this file, run_decompositions() works fine here (with no mutable
        # buffers, none of the AOT-autograd joint-graph issues that affected the
        # stateful export apply).
        exported = exported.run_decompositions({})

    print(f"Converting to CoreML (minimum_deployment_target={cfg.IOS_MIN_DEPLOYMENT})...")
    mlmodel = ct.convert(
        exported,
        inputs=[ct.TensorType(name="input_ids", shape=(1, ct.RangeDim(1, cfg.MAX_TOTAL_TOKENS)), dtype=int)],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=getattr(ct.target, cfg.IOS_MIN_DEPLOYMENT),
        convert_to="mlprogram",
    )

    if quant == "none":
        print("Skipping weight quantization -- shipping FP16 (compute_precision above), no INT4/INT8 step.")
        quantized = mlmodel
    else:
        print(f"Quantizing weights to {quant.upper()} ({cfg.QUANT_MODE})...")
        quant_config = ct.optimize.coreml.OptimizationConfig(
            global_config=ct.optimize.coreml.OpLinearQuantizerConfig(
                mode=cfg.QUANT_MODE, dtype=quant,
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
          f"{cfg.TARGET_LATENCY_MS}ms target -- this export recomputes the whole sequence on every "
          "call (no KV-cache, see this file's module docstring), so latency will grow with response "
          "length in a way a cached export wouldn't. Confirm it's still acceptable at the longest "
          "realistic generation length before shipping.")


def run_dry_run():
    print("export_coreml.py has no --dry-run mode of its own -- there is nothing meaningful to preview "
          "without a real fine-tuned checkpoint (unlike the earlier pipeline stages, there's no taxonomy/"
          "prompt text to print here, only a model conversion). Use finetune.py --dry-run and "
          "prune_vocabulary.py --dry-run to sanity-check the stages that feed this one.")


def main():
    parser = argparse.ArgumentParser(description="Export the fine-tuned stylist model to an INT4 CoreML package.")
    parser.add_argument("finetuned_dir", nargs="?", default=str(cfg.CHECKPOINTS_DIR / "finetune_run" / "final"))
    parser.add_argument("--output", default=str(cfg.MODELS_DIR / "StylistEngine.mlpackage"))
    parser.add_argument("--quant", choices=["int4", "int8", "none"], default=None,
                         help=f"Weight quantization level (default {cfg.QUANT_DTYPE}, from config.py). "
                              "'none' ships FP16 weights with no separate quantization step.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        run_dry_run()
        return

    if not Path(args.finetuned_dir).exists():
        sys.exit(f"!! {args.finetuned_dir} not found. Run finetune.py first.")

    run_export(args.finetuned_dir, args.output, quant=args.quant)


if __name__ == "__main__":
    main()
