"""Core ML export for wake word models.

The wake-word inference pipeline has three stages:

  1. mel spectrogram         (frozen, bundled as ``melspectrogram.onnx``)
  2. speech embedding CNN    (frozen, bundled as ``embedding_model.onnx``)
  3. wake-word classifier    (trained PyTorch model, varies per wake word)

This module converts each of those to Core ML ``.mlpackage`` files so the
inference path on macOS / iOS can run on the Apple Neural Engine (ANE) through
Core ML instead of shipping an ONNX runtime.

The classifier path (``export_classifier_coreml``) is the common case: it is
called from ``run_export_coreml`` for every trained model.  The two frozen
frontend models are converted once via ``convert_frontend_models`` and the
resulting ``.mlpackage`` files are committed alongside the Swift package.

Conversion strategy for the frozen ONNX models: ONNX -> PyTorch (onnx2torch) ->
torch.jit.trace -> coremltools.  ``coremltools.converters.onnx`` was removed in
coremltools 6+ so we take the PyTorch route, which also matches the
well-supported path used for the classifier itself.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from ..config import WakeWordConfig
from ..models.pipeline import WakeWordClassifier

if TYPE_CHECKING:
    import coremltools as ct

logger = logging.getLogger(__name__)


# Deployment target is intentionally iOS 16 / macOS 13: that is the first
# version with solid ML Program (mlprogram) support, fp16 weights and ANE
# scheduling for the op set used by the embedding CNN.
_DEFAULT_DEPLOYMENT_TARGET_NAME = "iOS16"


def _ct() -> ct:
    try:
        import coremltools as ct
    except ImportError as e:
        raise ImportError(
            "coremltools is required for Core ML export. "
            "Install with: pip install 'livekit-wakeword[coreml]'"
        ) from e
    return ct


def _precision(ct_mod, fp32: bool):  # type: ignore[no-untyped-def]
    if fp32:
        return ct_mod.precision.FLOAT32
    return ct_mod.precision.FLOAT16


def _deployment_target(ct_mod):  # type: ignore[no-untyped-def]
    return getattr(ct_mod.target, _DEFAULT_DEPLOYMENT_TARGET_NAME)


def _remove_existing(path: Path) -> None:
    """``.mlpackage`` is a directory; ``shutil.rmtree`` it if present."""
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def export_classifier_coreml(
    config: WakeWordConfig,
    model_path: Path,
    output_path: Path,
    fp32: bool = False,
) -> Path:
    """Export the trained classifier head to a Core ML ``.mlpackage``.

    The classifier operates on pre-extracted embeddings:

    - input  ``embeddings`` : ``(1, 16, 96)`` float32
    - output ``score``      : ``(1, 1)`` float32 in ``[0, 1]``

    Args:
        config: WakeWordConfig (used only for building the classifier).
        model_path: Path to the trained ``.pt`` state dict.
        output_path: Destination ``.mlpackage`` path.
        fp32: If True, keep weights in float32. Defaults to float16, which is
            what runs on the ANE.

    Returns:
        ``output_path``.
    """
    ct = _ct()

    model = WakeWordClassifier(config)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    # Keep the batch axis fixed at 1: Core ML can still recompile for other
    # batch sizes but on-device wakeword detection only ever uses batch=1.
    example = (torch.randn(1, 16, 96),)
    # `torch.export` + `run_decompositions({})` collapses the ATEN-level
    # MultiheadAttention call into primitive ops that coremltools can ingest.
    # We prefer this over `torch.jit.trace` because tracing emits an
    # `aten::Int(Tensor)` node in MultiheadAttention's shape handling that
    # the CoreML converter can't lower.
    exported = torch.export.export(model, example).run_decompositions({})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _remove_existing(output_path)

    mlmodel = ct.convert(
        exported,
        inputs=[
            ct.TensorType(name="embeddings", shape=(1, 16, 96), dtype=float),
        ],
        outputs=[ct.TensorType(name="score", dtype=float)],
        convert_to="mlprogram",
        compute_precision=_precision(ct, fp32),
        minimum_deployment_target=_deployment_target(ct),
    )
    mlmodel.short_description = (
        f"LiveKit wake word classifier ({config.model.model_type.value}, "
        f"{config.model.model_size.value}) — inputs (1,16,96) embeddings, "
        "output in [0, 1]."
    )
    mlmodel.save(str(output_path))

    logger.info("Exported classifier Core ML package to %s", output_path)
    return output_path


def _patch_onnx2torch_for_coreml() -> None:
    """Patch ``onnx2torch`` operators that coremltools cannot lower.

    onnx2torch's default implementations for ``Min``/``Max`` call
    ``Tensor.broadcast_to`` which coremltools' Torch frontend does not support
    (as of coremltools 8.x). Since we only need binary min/max, override
    ``apply_reduction`` to use ``torch.minimum`` / ``torch.maximum``, which
    broadcast natively and lower cleanly to Core ML.
    """
    try:
        from onnx2torch.node_converters.min_max import OnnxMinMax  # type: ignore
    except ImportError:
        return

    def _apply_reduction(self, *tensors):  # type: ignore[no-untyped-def]
        op = torch.maximum if self._operator is torch.amax else torch.minimum
        result = tensors[0]
        for t in tensors[1:]:
            result = op(result, t)
        return result

    OnnxMinMax.apply_reduction = _apply_reduction  # type: ignore[assignment]


def _load_onnx_as_torch(onnx_path: Path) -> torch.nn.Module:
    """Load an ONNX model and return an equivalent PyTorch ``nn.Module``.

    Uses ``onnx2torch`` for structural conversion. The resulting module is
    set to eval mode but otherwise unmodified.
    """
    try:
        from onnx2torch import convert  # type: ignore
    except ImportError as e:
        raise ImportError(
            "onnx2torch is required to convert the frozen ONNX frontends. "
            "Install with: pip install 'livekit-wakeword[coreml]'"
        ) from e

    import onnx  # onnx comes along with onnx2torch

    _patch_onnx2torch_for_coreml()

    model = onnx.load(str(onnx_path))
    torch_model = convert(model)
    torch_model.eval()
    return torch_model


class _MelSpectrogramModule(torch.nn.Module):
    """Byte-equivalent reimplementation of the frozen ``melspectrogram.onnx`` model.

    The original ONNX graph is a torchlibrosa export that uses dynamic ``Clip``
    bounds (``max = ReduceMax(x) - 80``) — ``onnx2torch`` rejects this, so we
    reconstruct the same computation in pure PyTorch ops. The learned weights
    (STFT real/imag conv kernels and the mel filterbank) are loaded from the
    original ONNX initializer tensors so numerics stay identical to the
    ONNX-runtime path.

    Graph (from inspection of ``melspectrogram.onnx``):

        audio -> Conv(real) -> square ┐
                                      ├-> add -> MatMul(melW) -> Clip(min=0)
              -> Conv(imag) -> square ┘
          -> log -> * 10 / ln(10)  (i.e. 10 * log10(x))
          -> sub 0.0
          -> max = ReduceMax(x)
          -> Clip(max - 80, inf)
          = output
    """

    def __init__(
        self,
        conv_real_weight: torch.Tensor,  # (257, 1, 512)
        conv_imag_weight: torch.Tensor,  # (257, 1, 512)
        mel_filterbank: torch.Tensor,  # (257, 32)
    ):
        super().__init__()
        self.register_buffer("conv_real_weight", conv_real_weight)
        self.register_buffer("conv_imag_weight", conv_imag_weight)
        # mel filterbank: (257, 32). We'll matmul with (..., 257) @ (257, 32).
        self.register_buffer("mel_filterbank", mel_filterbank)

    @classmethod
    def from_onnx(cls, onnx_path: Path) -> _MelSpectrogramModule:
        import onnx
        from onnx import numpy_helper

        m = onnx.load(str(onnx_path))
        weights: dict[str, torch.Tensor] = {}
        for init in m.graph.initializer:
            arr = numpy_helper.to_array(init).copy()
            weights[init.name] = torch.from_numpy(arr)

        real_w = weights["0.stft.conv_real.weight"].float()
        imag_w = weights["0.stft.conv_imag.weight"].float()
        mel_w = weights["1.melW"].float()
        return cls(real_w, imag_w, mel_w)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        # audio: (batch, samples) -> (batch, 1, samples) for Conv1d
        x = audio.unsqueeze(1)
        # stride=160, hop=160 (matches torchlibrosa default STFT export)
        real = torch.nn.functional.conv1d(x, self.conv_real_weight, stride=160)
        imag = torch.nn.functional.conv1d(x, self.conv_imag_weight, stride=160)
        # power spectrogram
        power = real * real + imag * imag  # (batch, 257, frames)
        # mel projection: (batch, frames, 257) @ (257, 32) -> (batch, frames, 32)
        power_bft = power.transpose(1, 2)
        mel = power_bft @ self.mel_filterbank  # (batch, frames, 32)
        # Clip(min=0, max=+inf)
        mel = torch.clamp(mel, min=0.0)
        # power_to_db: 10 * log10(x + eps), with the same "ref = max - 80 dB"
        # floor as torchlibrosa.
        # Use log(x) / log(10) to match the ONNX graph (which divides by
        # ln(10) = 2.3025851).
        log10 = torch.log(mel.clamp(min=1e-10)) / 2.302585092994046
        log_db = log10 * 10.0
        log_db = log_db - 0.0  # (the "- Sub_30" constant from ONNX)
        # Floor at (ReduceMax - 80)
        max_per = log_db.amax(dim=(1, 2), keepdim=True)
        floor = max_per - 80.0
        out = torch.maximum(log_db, floor)
        # Output shape from ONNX was (batch, 1, frames, 32); match that.
        return out.unsqueeze(1)


def convert_mel_frontend(
    mel_onnx_path: Path,
    output_path: Path,
    *,
    fp32: bool = False,
    min_samples: int = 16000,
    max_samples: int = 48000,
    default_samples: int = 32000,
) -> Path:
    """Convert the mel-spectrogram ONNX frontend to a Core ML ``.mlpackage``.

    The mel model's input is ``(batch=1, samples)``; the number of samples
    depends on how long an audio chunk the caller passes in. We declare a
    ``RangeDim`` so a single ``.mlpackage`` handles chunks from ``min_samples``
    to ``max_samples`` (covering 1–3 s at 16 kHz).

    The output of the raw ONNX model is ``(1, 1, time, 32)``. We append the
    openWakeWord ``x/10 + 2`` normalization so the resulting Core ML model is
    drop-in compatible with the Python / Rust pipelines.
    """
    ct = _ct()

    base = _MelSpectrogramModule.from_onnx(mel_onnx_path)

    class MelWithNorm(torch.nn.Module):
        def __init__(self, inner: torch.nn.Module):
            super().__init__()
            self.inner = inner

        def forward(self, audio: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            raw = self.inner(audio)
            # Inner output is (1, 1, time, 32); flatten to (time, 32) so the
            # Swift side only has one format to consume.
            if raw.dim() == 4:
                raw = raw.squeeze(1)
            if raw.dim() == 3:
                raw = raw.squeeze(0)
            return raw / 10.0 + 2.0

    wrapped = MelWithNorm(base)
    wrapped.eval()

    dummy_audio = torch.zeros(1, default_samples, dtype=torch.float32)
    samples_sym = torch.export.Dim(
        "samples",
        min=min_samples,
        max=max_samples,
    )
    exported = torch.export.export(
        wrapped,
        (dummy_audio,),
        dynamic_shapes=({1: samples_sym},),
    ).run_decompositions({})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _remove_existing(output_path)

    samples_dim = ct.RangeDim(
        lower_bound=min_samples,
        upper_bound=max_samples,
        default=default_samples,
    )
    mlmodel = ct.convert(
        exported,
        inputs=[
            ct.TensorType(
                name="audio",
                shape=(1, samples_dim),
                dtype=float,
            ),
        ],
        outputs=[ct.TensorType(name="mel", dtype=float)],
        convert_to="mlprogram",
        compute_precision=_precision(ct, fp32),
        minimum_deployment_target=_deployment_target(ct),
    )
    mlmodel.short_description = (
        "LiveKit wake word mel spectrogram frontend — input (1, samples) 16 kHz "
        "float32 in [-1, 1]; output (time, 32) normalized mel features "
        "(x/10 + 2 applied)."
    )
    mlmodel.save(str(output_path))
    logger.info("Exported mel frontend Core ML package to %s", output_path)
    return output_path


def convert_embedding_model(
    embedding_onnx_path: Path,
    output_path: Path,
    *,
    fp32: bool = False,
    batch_size: int = 16,
) -> Path:
    """Convert the speech embedding ONNX CNN to a Core ML ``.mlpackage``.

    Input: ``(batch, 76, 32, 1)`` channels-last mel windows, where ``batch``
    is fixed at ``batch_size`` (default 16 — matches the number of embeddings
    the classifier needs).

    Output: ``(batch, 96)`` embedding vectors (we squeeze the spatial 1x1
    dimensions that the underlying CNN leaves in place).

    A fixed batch is used because the ONNX model's Reshape nodes rely on a
    known leading dim at export time. In practice the wake-word pipeline
    always evaluates a fixed 16-window tail, so a single Core ML package
    covers every call site.
    """
    ct = _ct()

    base = _load_onnx_as_torch(embedding_onnx_path)

    class EmbeddingWithSqueeze(torch.nn.Module):
        def __init__(self, inner: torch.nn.Module):
            super().__init__()
            self.inner = inner

        def forward(self, mel_windows: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            raw = self.inner(mel_windows)
            # Raw ONNX output is (batch, 1, 1, 96); flatten the spatial dims.
            if raw.dim() == 4:
                raw = raw.squeeze(2).squeeze(1)
            return raw

    wrapped = EmbeddingWithSqueeze(base)
    wrapped.eval()

    dummy = torch.zeros(batch_size, 76, 32, 1, dtype=torch.float32)
    # ``torch.export`` can't walk the onnx2torch Reshape wrapper cleanly
    # (it uses data-dependent predicates). Use ``torch.jit.trace`` instead,
    # which yields a static graph for this purely convolutional model.
    traced = torch.jit.trace(wrapped, dummy, check_trace=False, strict=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _remove_existing(output_path)

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(
                name="mel_windows",
                shape=(batch_size, 76, 32, 1),
                dtype=float,
            ),
        ],
        outputs=[ct.TensorType(name="embeddings", dtype=float)],
        convert_to="mlprogram",
        compute_precision=_precision(ct, fp32),
        minimum_deployment_target=_deployment_target(ct),
    )
    mlmodel.short_description = (
        "LiveKit wake word speech embedding CNN — input (batch, 76, 32, 1) "
        "normalized mel windows; output (batch, 96) embedding vectors."
    )
    mlmodel.save(str(output_path))
    logger.info("Exported embedding Core ML package to %s", output_path)
    return output_path


def convert_frontend_models(
    mel_onnx: Path,
    embedding_onnx: Path,
    output_dir: Path,
    *,
    fp32: bool = False,
) -> tuple[Path, Path]:
    """Convert both frozen frontend ONNX models to ``.mlpackage`` files.

    This is the one-shot helper used to regenerate the Swift-package bundled
    resources. Output paths are ``{output_dir}/melspectrogram.mlpackage`` and
    ``{output_dir}/embedding_model.mlpackage``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    mel_out = convert_mel_frontend(
        mel_onnx,
        output_dir / "melspectrogram.mlpackage",
        fp32=fp32,
    )
    emb_out = convert_embedding_model(
        embedding_onnx,
        output_dir / "embedding_model.mlpackage",
        fp32=fp32,
    )
    return mel_out, emb_out


def _rewrite_classifier_onnx_for_onnx2torch(model):  # type: ignore[no-untyped-def]
    """Normalize a classifier ONNX graph so ``onnx2torch`` can ingest it.

    Two opset-18-era idioms trip ``onnx2torch`` up:

    * ``ReduceMean`` moved its ``axes`` from an attribute to a second input in
      opset 18. ``onnx2torch`` only registers v1/11/13 converters, which read
      ``axes`` from the attribute. We inline the second-input constant back
      into an ``axes`` attribute.
    * ``Reshape``'s ``allowzero=1`` is unsupported; we drop it (our models do
      not rely on the zero-sized-dim semantics).

    Finally, we downgrade the opset import to 17 so ``onnx2torch`` picks the
    v17 ``LayerNormalization`` converter and the normalized v13 ``ReduceMean``.
    """
    import onnx

    init_map = {init.name: init for init in model.graph.initializer}
    const_map = {}
    for node in model.graph.node:
        if node.op_type == "Constant" and len(node.output) == 1:
            for attr in node.attribute:
                if attr.name == "value":
                    const_map[node.output[0]] = onnx.numpy_helper.to_array(attr.t)

    new_nodes = []
    for node in model.graph.node:
        if node.op_type == "ReduceMean" and len(node.input) >= 2:
            axes_name = node.input[1]
            axes = None
            if axes_name in init_map:
                axes = onnx.numpy_helper.to_array(init_map[axes_name]).tolist()
            elif axes_name in const_map:
                axes = const_map[axes_name].tolist()
            if axes is None:
                new_nodes.append(node)
                continue
            attrs = {
                a.name: onnx.helper.get_attribute_value(a)
                for a in node.attribute
                if a.name != "noop_with_empty_axes"
            }
            attrs["axes"] = axes
            new_nodes.append(
                onnx.helper.make_node(
                    "ReduceMean",
                    [node.input[0]],
                    list(node.output),
                    name=node.name,
                    **attrs,
                )
            )
        elif node.op_type == "Reshape":
            attrs = {
                a.name: onnx.helper.get_attribute_value(a)
                for a in node.attribute
                if a.name != "allowzero"
            }
            new_nodes.append(
                onnx.helper.make_node(
                    "Reshape",
                    list(node.input),
                    list(node.output),
                    name=node.name,
                    **attrs,
                )
            )
        else:
            new_nodes.append(node)

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    for imp in model.opset_import:
        if imp.domain == "":
            imp.version = 17
    return model


def convert_classifier_from_onnx(
    onnx_path: Path,
    output_path: Path,
    *,
    fp32: bool = False,
) -> Path:
    """Convert an already-exported classifier ``.onnx`` to a Core ML ``.mlpackage``.

    Unlike :func:`export_classifier_coreml`, this path does not need access to
    the original PyTorch source or config: it goes ONNX → onnx2torch →
    ``torch.jit.trace`` → coremltools. Useful for converting legacy
    ``.onnx`` artifacts that ship with demo apps or older releases.

    Input/output tensor shapes match ``export_classifier_coreml``:

    - input  ``embeddings`` : ``(1, 16, 96)`` float32
    - output ``score``      : ``(1, 1)`` float32 in ``[0, 1]``
    """
    ct = _ct()

    try:
        import onnx  # noqa: F401
        from onnx2torch import convert as onnx2torch_convert  # type: ignore
    except ImportError as e:
        raise ImportError(
            "onnx + onnx2torch are required for ONNX → Core ML conversion. "
            "Install with: pip install 'livekit-wakeword[coreml]'"
        ) from e

    import tempfile

    import onnx

    _patch_onnx2torch_for_coreml()

    model = onnx.load(str(onnx_path))
    model = _rewrite_classifier_onnx_for_onnx2torch(model)

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=True) as tmp:
        onnx.save(model, tmp.name)
        torch_mod = onnx2torch_convert(tmp.name)
    torch_mod.eval()

    dummy = torch.randn(1, 16, 96, dtype=torch.float32)
    traced = torch.jit.trace(torch_mod, dummy, check_trace=False, strict=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _remove_existing(output_path)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="embeddings", shape=(1, 16, 96), dtype=float)],
        outputs=[ct.TensorType(name="score", dtype=float)],
        convert_to="mlprogram",
        compute_precision=_precision(ct, fp32),
        minimum_deployment_target=_deployment_target(ct),
    )
    mlmodel.short_description = (
        "LiveKit wake word classifier (converted from ONNX) — "
        "inputs (1,16,96) embeddings, output in [0, 1]."
    )
    mlmodel.save(str(output_path))
    logger.info("Converted classifier ONNX %s -> Core ML %s", onnx_path, output_path)
    return output_path


def run_export_coreml(config: WakeWordConfig, fp32: bool = False) -> Path:
    """Export the trained classifier for ``config`` to Core ML.

    Mirrors :func:`livekit.wakeword.export.onnx.run_export` so the two export
    formats have interchangeable entry points.
    """
    model_dir = config.model_output_dir
    model_path = model_dir / f"{config.model_name}.pt"

    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found: {model_path}")

    mlpackage_path = model_dir / f"{config.model_name}.mlpackage"
    export_classifier_coreml(config, model_path, mlpackage_path, fp32=fp32)
    return mlpackage_path
