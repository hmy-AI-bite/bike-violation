"""可在 CPU 上独立验证的区域配置、类别映射与掩膜判定规则。"""
import math
import numpy as np

BAD_VERDICTS = frozenset({"压线", "出界", "占道"})


def bicycle_class_id(names):
    items = names.items() if isinstance(names, dict) else enumerate(names)
    matches = [int(k) for k, v in items if str(v).strip().lower() == "bicycle"]
    if len(matches) != 1:
        raise ValueError("模型必须包含唯一的 bicycle 类别，实际 names=%r" % (names,))
    return matches[0]


def validate_polygon(poly):
    pts = np.asarray(poly, dtype=float)
    if (pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 2
            or not np.isfinite(pts).all() or (pts < 0).any() or (pts > 1).any()):
        raise ValueError("区域多边形必须含至少三个有限的归一化坐标点")
    area = abs(np.dot(pts[:, 0], np.roll(pts[:, 1], 1))
               - np.dot(pts[:, 1], np.roll(pts[:, 0], 1)))
    if area <= 1e-12:
        raise ValueError("区域多边形面积为零")


def resolve_zones(scene, stem):
    per_image = (scene.get("images") or {}).get(stem, {})
    # 缺少键才继承；显式 [] 表示清空，不能使用 or 回落默认区域。
    allow = per_image.get("allow", scene.get("allow", []))
    deny = per_image.get("deny", scene.get("deny", []))
    if allow:
        validate_polygon(allow)
    for poly in deny:
        validate_polygon(poly)
    return allow, deny


def contact_band(mask, ratio=0.02):
    if not 0 < ratio <= 1:
        raise ValueError("接地点条带比例必须在 (0, 1] 内")
    mask = np.asarray(mask, dtype=bool)
    ys = np.where(mask)[0]
    if not ys.size:
        return np.zeros_like(mask)
    height = int(ys.max() - ys.min() + 1)
    band = max(2, int(math.ceil(ratio * height)))
    return mask & (np.arange(mask.shape[0])[:, None] >= int(ys.max()) - band + 1)


def classify_mask(mask, allow_mask, deny_mask, *, has_zone, score,
                  min_area=200, min_score=0.0, contact_ratio=0.02, deny_tol=0.05):
    result = dict(verdict="未标定", contact_in=-1.0, deny=-1.0, reason="no_zone")
    if not has_zone:
        return result
    mask = np.asarray(mask, dtype=bool)
    reason = ("invalid_score" if not math.isfinite(float(score)) else
              "low_score" if score < min_score else
              "small_mask" if mask.sum() < max(1, min_area) else "")
    if reason:
        result.update(verdict="无法判定", reason=reason)
        return result
    contact = contact_band(mask, contact_ratio)
    c_frac = float((contact & allow_mask).sum() / contact.sum())
    d_frac = float((mask & deny_mask).sum() / mask.sum())
    verdict = ("占道" if d_frac > deny_tol else "合规" if c_frac >= 0.9 else
               "压线" if c_frac > 0.1 else "出界")
    return dict(verdict=verdict, contact_in=c_frac, deny=d_frac, reason="")


def count_verdicts(rows):
    counts = {v: 0 for v in ("合规", "压线", "出界", "占道", "无法判定", "未标定")}
    for row in rows:
        counts[row["verdict"]] += 1
    return counts
