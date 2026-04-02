"""Evaluate a wake word model and produce a DET curve plot."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort

from ..config import WakeWordConfig

logger = logging.getLogger(__name__)


def _load_validation_features(
    config: WakeWordConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Load positive and negative validation features from pre-extracted .npy files.

    Returns:
        (positive_features, negative_features, positive_vad, negative_vad)
        Features are shaped (N, 16, 96).  VAD arrays are (N,) floats or
        None when VAD scores are not available.
    """
    model_dir = config.model_output_dir
    pos_path = model_dir / "positive_features_test.npy"
    neg_path = model_dir / "negative_features_test.npy"
    pos_vad_path = model_dir / "positive_vad_test.npy"
    neg_vad_path = model_dir / "negative_vad_test.npy"

    pos = np.load(str(pos_path)) if pos_path.exists() else np.zeros((0, 16, 96))
    neg = np.load(str(neg_path)) if neg_path.exists() else np.zeros((0, 16, 96))
    pos_vad = np.load(str(pos_vad_path)) if pos_vad_path.exists() else None
    neg_vad = np.load(str(neg_vad_path)) if neg_vad_path.exists() else None

    # Also include general negative validation features if available
    val_path = config.data_path / "features" / "validation_set_features.npy"
    val_vad_path = config.data_path / "features" / "validation_set_vad.npy"
    if val_path.exists():
        val_neg = np.load(str(val_path))
        if val_neg.ndim == 2:
            n_full = (val_neg.shape[0] // 16) * 16
            remainder = val_neg.shape[0] - n_full
            if remainder > 0:
                logger.warning(
                    "Dropping %d/%d validation samples (not divisible by 16)",
                    remainder,
                    val_neg.shape[0],
                )
            val_neg = val_neg[:n_full].reshape(-1, 16, 96)
        neg = np.concatenate([neg, val_neg], axis=0) if neg.shape[0] > 0 else val_neg

        # Extend negative VAD scores if available
        if val_vad_path.exists():
            val_vad = np.load(str(val_vad_path))
            if val_neg.ndim == 2:
                val_vad = val_vad[:n_full // 16] if len(val_vad) > n_full // 16 else val_vad
            if neg_vad is not None:
                neg_vad = np.concatenate([neg_vad, val_vad])
            else:
                neg_vad = val_vad
        elif neg_vad is not None:
            # No VAD for validation set — fill with 1.0 (assume speech)
            neg_vad = np.concatenate(
                [neg_vad, np.ones(val_neg.shape[0], dtype=np.float32)]
            )

    if pos.shape[0] == 0:
        raise ValueError(
            f"No positive validation features found at {pos_path}. "
            "Run the generate/augment pipeline first."
        )
    if neg.shape[0] == 0:
        raise ValueError(
            f"No negative validation features found. "
            f"Checked {neg_path} and {val_path}. "
            "Run setup and the generate/augment pipeline first."
        )

    logger.info(f"Loaded {pos.shape[0]} positive, {neg.shape[0]} negative validation samples")
    if pos_vad is not None:
        logger.info("VAD scores available — will apply VAD mask during eval")
    return pos, neg, pos_vad, neg_vad


def _predict_onnx(
    session: ort.InferenceSession,
    features: np.ndarray,
    batch_size: int = 1,
) -> np.ndarray:
    """Run ONNX model on feature batches, return scores array."""
    input_name = session.get_inputs()[0].name
    all_scores: list[np.ndarray] = []
    for i in range(0, len(features), batch_size):
        batch = features[i : i + batch_size].astype(np.float32)
        outputs = session.run(None, {input_name: batch})
        all_scores.append(outputs[0].squeeze(-1))
    return np.concatenate(all_scores, axis=0)


def _compute_det_curve(
    pos_scores: np.ndarray,
    neg_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute DET curve points (FPR, FNR) across thresholds.

    Returns:
        (thresholds, fpr, fnr) arrays sorted by ascending threshold.
    """
    thresholds = np.linspace(0.0, 1.0, 1001)
    fpr = np.array([np.mean(neg_scores >= t) for t in thresholds])
    fnr = np.array([np.mean(pos_scores < t) for t in thresholds])
    return thresholds, fpr, fnr


def _compute_aut(fpr: np.ndarray, fnr: np.ndarray) -> float:
    """Compute Area Under the DET curve (AUT) using the trapezoidal rule.

    Lower is better (0 = perfect).
    We integrate FNR as a function of FPR (sorted by ascending FPR).
    """
    # Sort by FPR for proper integration
    sort_idx = np.argsort(fpr)
    fpr_sorted = fpr[sort_idx]
    fnr_sorted = fnr[sort_idx]
    return float(np.trapezoid(fnr_sorted, fpr_sorted))


def _plot_det_curve(
    fpr: np.ndarray,
    fnr: np.ndarray,
    aut: float,
    model_name: str,
    output_path: Path,
    metrics: dict[str, float],
) -> None:
    """Render DET curve to PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    fig, ax = plt.subplots(figsize=(8, 7))

    # Plot DET curve (FPR vs FNR)
    sort_idx = np.argsort(fpr)
    ax.plot(fpr[sort_idx] * 100, fnr[sort_idx] * 100, linewidth=2, color="#2563eb")

    # Shade AUT area
    ax.fill_between(
        fpr[sort_idx] * 100,
        fnr[sort_idx] * 100,
        alpha=0.15,
        color="#2563eb",
    )

    ax.set_xlabel("False Positive Rate (%)", fontsize=13)
    ax.set_ylabel("False Negative Rate (%)", fontsize=13)
    ax.set_title(f"DET Curve \u2014 {model_name}", fontsize=15, fontweight="bold")

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
    ax.grid(True, alpha=0.3)

    # Diagonal reference (random classifier)
    ax.plot([0, 100], [100, 0], "--", color="gray", alpha=0.5, label="Random")

    # Annotation box with metrics
    text_lines = [
        f"AUT: {aut:.4f}",
        f"FPPH: {metrics['fpph']:.2f}",
        f"Recall: {metrics['recall']:.1%}",
        f"Threshold: {metrics['threshold']:.2f}",
        f"Optimal Thresh: {metrics['optimal_threshold']:.2f}",
    ]
    ax.text(
        0.97,
        0.97,
        "\n".join(text_lines),
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#ccc", alpha=0.9),
        fontfamily="monospace",
    )

    ax.legend(loc="lower left", fontsize=10)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    logger.info(f"DET curve saved to {output_path}")


def run_eval(
    config: WakeWordConfig,
    model_path: str | Path,
    vad_threshold: float | None = None,
) -> dict[str, float]:
    """Run full evaluation: compute scores, DET curve, AUT, and save plot + metrics JSON.

    Args:
        config: Wake word configuration (used to locate validation data).
        model_path: Path to the ONNX classifier model to evaluate.
        vad_threshold: When set, apply the VAD gate during evaluation.
            Clips whose VAD speech probability is below this threshold
            receive a score of 0.0.  Pass ``None`` to skip VAD gating.

    Returns:
        Dict with keys: aut, fpph, recall, accuracy, threshold
    """
    # Load ONNX model
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    logger.info(f"Loaded model from {model_path}")

    # Load validation data
    pos_features, neg_features, pos_vad, neg_vad = _load_validation_features(config)

    # Run predictions
    logger.info("Running predictions on validation set...")
    pos_scores = _predict_onnx(session, pos_features)
    neg_scores = _predict_onnx(session, neg_features)

    # Apply VAD mask: zero out scores where VAD says no speech
    if vad_threshold is not None:
        if pos_vad is not None:
            rejected_pos = pos_vad < vad_threshold
            pos_scores[rejected_pos] = 0.0
            logger.info(
                "VAD rejected %d/%d positive clips (%.1f%%)",
                rejected_pos.sum(), len(rejected_pos),
                100.0 * rejected_pos.sum() / max(len(rejected_pos), 1),
            )
        if neg_vad is not None:
            rejected_neg = neg_vad < vad_threshold
            neg_scores[rejected_neg] = 0.0
            logger.info(
                "VAD rejected %d/%d negative clips (%.1f%%)",
                rejected_neg.sum(), len(rejected_neg),
                100.0 * rejected_neg.sum() / max(len(rejected_neg), 1),
            )

    # Compute DET curve
    thresholds, fpr, fnr = _compute_det_curve(pos_scores, neg_scores)

    # Compute AUT
    aut = _compute_aut(fpr, fnr)

    # Compute summary metrics at fixed threshold 0.5 for consistent comparison
    clip_duration = config.augmentation.clip_duration
    validation_hours = neg_features.shape[0] * clip_duration / 3600.0

    from ..training.metrics import evaluate_model, find_best_threshold

    fixed = evaluate_model(
        pos_scores,
        neg_scores,
        threshold=0.5,
        validation_hours=validation_hours,
    )

    optimal = find_best_threshold(
        pos_scores,
        neg_scores,
        validation_hours=validation_hours,
        target_fpph=config.target_fp_per_hour,
    )

    # Build results
    results = {
        "aut": aut,
        "fpph": fixed["fpph"],
        "recall": fixed["recall"],
        "accuracy": fixed["accuracy"],
        "threshold": fixed["threshold"],
        "optimal_threshold": optimal["threshold"],
        "optimal_recall": optimal["recall"],
        "optimal_fpph": optimal["fpph"],
        "n_positive": int(pos_features.shape[0]),
        "n_negative": int(neg_features.shape[0]),
        "validation_hours": round(validation_hours, 2),
    }

    # Save plot
    output_dir = config.model_output_dir
    plot_path = output_dir / f"{config.model_name}_det.png"
    plot_metrics = {**fixed, "optimal_threshold": optimal["threshold"]}
    _plot_det_curve(fpr, fnr, aut, config.model_name, plot_path, plot_metrics)

    # Save metrics JSON
    metrics_path = output_dir / f"{config.model_name}_eval.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(results, indent=2) + "\n")
    logger.info(f"Eval metrics saved to {metrics_path}")

    return results
