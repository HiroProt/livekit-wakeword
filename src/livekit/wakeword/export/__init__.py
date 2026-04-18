"""Export trained wake-word models to deployable formats."""

from .onnx import export_classifier, quantize_onnx, run_export


def __getattr__(name: str) -> object:
    # Core ML is an optional extra; don't force coremltools as a transitive
    # dependency of `from livekit.wakeword.export import ...` imports that
    # only want the ONNX path.
    if name in {
        "export_classifier_coreml",
        "convert_frontend_models",
        "convert_mel_frontend",
        "convert_embedding_model",
        "convert_classifier_from_onnx",
        "run_export_coreml",
    }:
        from . import coreml

        return getattr(coreml, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "export_classifier",
    "quantize_onnx",
    "run_export",
    "export_classifier_coreml",
    "convert_frontend_models",
    "convert_mel_frontend",
    "convert_embedding_model",
    "convert_classifier_from_onnx",
    "run_export_coreml",
]
