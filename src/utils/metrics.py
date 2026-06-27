import math

import torch


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def classification_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    """Compute binary classification metrics from logits and labels."""
    probs = torch.sigmoid(logits.detach().cpu()).float()
    labels = labels.detach().cpu().float()
    preds = (probs >= 0.5).float()

    tp = int(((preds == 1) & (labels == 1)).sum().item())
    tn = int(((preds == 0) & (labels == 0)).sum().item())
    fp = int(((preds == 1) & (labels == 0)).sum().item())
    fn = int(((preds == 0) & (labels == 1)).sum().item())

    accuracy = _safe_div(tp + tn, len(labels))
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }
    roc_auc = binary_roc_auc(probs, labels)
    if roc_auc is not None:
        metrics["roc_auc"] = roc_auc
    return metrics


def binary_roc_auc(probs: torch.Tensor, labels: torch.Tensor) -> float | None:
    """Compute ROC-AUC without sklearn; return None when one class is absent."""
    positives = probs[labels == 1]
    negatives = probs[labels == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return None
    comparisons = (positives[:, None] > negatives[None, :]).float()
    ties = (positives[:, None] == negatives[None, :]).float() * 0.5
    return float((comparisons + ties).mean().item())


def regression_metrics(tc_pred: torch.Tensor, tc_true: torch.Tensor, high_tc_threshold: float = 77.0) -> dict[str, float | None]:
    """Compute Tc regression metrics globally and on scientific slices."""
    pred = tc_pred.detach().cpu().float()
    true = tc_true.detach().cpu().float()
    err = pred - true
    abs_err = err.abs()
    metrics: dict[str, float | None] = {
        "mae": float(abs_err.mean().item()),
        "rmse": float(torch.sqrt((err**2).mean()).item()),
    }

    supra_mask = true > 0
    high_mask = true > high_tc_threshold
    metrics["mae_superconductors"] = float(abs_err[supra_mask].mean().item()) if supra_mask.any() else None
    metrics["mae_high_tc"] = float(abs_err[high_mask].mean().item()) if high_mask.any() else None
    return metrics


def format_metrics(metrics: dict) -> str:
    """Render nested metric dictionaries in a compact readable form."""
    lines = []
    for key, value in metrics.items():
        if isinstance(value, float):
            value = 0.0 if math.isnan(value) else value
            lines.append(f"{key}: {value:.4f}")
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines)
