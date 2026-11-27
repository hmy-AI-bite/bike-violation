"""Untrained hypotheses for a frozen-mask development ablation.

Neither lower silhouette nor whole silhouette is a true ground-contact estimate.
No human parking labels or annotated contact points are accepted by this module.
"""
import math

import numpy as np


def lower_envelope(mask, ratio=0.02):
    """Retain a thin band above the lowest occupied pixel in each column."""
    if not 0 < ratio <= 1:
        raise ValueError("ratio must be in (0, 1]")
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("mask must be 2D")
    ys = np.nonzero(mask)[0]
    if not len(ys):
        return np.zeros_like(mask)
    thickness = max(2, math.ceil(ratio * (int(ys.max()) - int(ys.min()) + 1)))
    occupied = mask.any(axis=0)
    bottom = mask.shape[0] - 1 - np.argmax(mask[::-1], axis=0)
    return mask & occupied[None, :] & (np.arange(mask.shape[0])[:, None] > bottom - thickness)


def fraction_verdict(evidence, allow):
    """Apply the legacy .1/.9 thresholds to a different representation."""
    count = int(evidence.sum())
    if not count:
        return "无法判定", None
    inside = float(np.count_nonzero(evidence & allow) / count)
    return ("合规" if inside >= .9 else "压线" if inside > .1 else "出界"), inside


def ownership_diagnostics(masks, boxes):
    """Mask leak outside its prompt and exact maximum inter-mask IoU.

    High overlap is only an ambiguity flag: heavy occlusion may legitimately
    overlap, and low overlap is not evidence that an instance is correct.
    """
    masks = np.asarray(masks, dtype=bool)
    n, height, width = masks.shape
    if len(boxes) != n:
        raise ValueError("mask/box counts differ")
    areas = masks.reshape(n, -1).sum(axis=1)
    packed = np.packbits(masks.reshape(n, -1), axis=1)
    popcount = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)
    max_iou = np.zeros(n)
    neighbor = [None] * n
    for i in range(n):
        for j in range(i + 1, n):
            intersection = int(popcount[np.bitwise_and(packed[i], packed[j])].sum())
            union = int(areas[i] + areas[j]) - intersection
            iou = intersection / union if union else 0.
            for a, b in ((i, j), (j, i)):
                if iou > max_iou[a]:
                    max_iou[a], neighbor[a] = iou, b
    diagnostics = []
    for i, (_, cx, cy, bw, bh) in enumerate(boxes):
        x1 = min(width, max(0, math.floor((cx - bw / 2) * width)))
        x2 = min(width, max(0, math.ceil((cx + bw / 2) * width)))
        y1 = min(height, max(0, math.floor((cy - bh / 2) * height)))
        y2 = min(height, max(0, math.ceil((cy + bh / 2) * height)))
        retained = int(masks[i, y1:y2, x1:x2].sum())
        leak = 1 - retained / int(areas[i]) if areas[i] else 0.
        diagnostics.append(dict(box_leak=leak, max_mask_iou=float(max_iou[i]),
                                neighbor_index=neighbor[i]))
    return diagnostics


def candidate_predictions(mask, allow, baseline, diagnostics):
    """Fixed hypotheses, not a parameter sweep or a learned classifier.

    Whole-mask .02 reuses the existing box-rule tolerance, but has different
    semantics. Ownership .80/.05 are exploratory diagnostics, not paper values
    or thresholds validated on independent data.
    """
    if baseline["verdict"] in ("无法判定", "未标定"):
        return {name: baseline["verdict"] for name in
                ("band", "column_envelope", "silhouette_2pct", "band_with_review")}
    envelope, _ = fraction_verdict(lower_envelope(mask), allow)
    outside = float(np.count_nonzero(mask & ~allow) / max(1, np.count_nonzero(mask)))
    ambiguous = diagnostics["max_mask_iou"] >= .8 or diagnostics["box_leak"] > .05
    return dict(band=baseline["verdict"], column_envelope=envelope,
                silhouette_2pct="出界" if outside > .02 else "合规",
                band_with_review="无法判定" if ambiguous else baseline["verdict"])
