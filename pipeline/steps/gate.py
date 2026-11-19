"""pipeline/steps/gate.py

Cổng quyết định Release / Hold / Rework.

Quy tắc:
    1. Bất kỳ issue severity="error" nào → Rework (không check reviewed)
    2. Nếu warn_ratio >= threshold và n_images > 0 → Hold
    3. Ngược lại → Release

Đếm ảnh, không đếm issue (1 ảnh có 3 warn = 1 ảnh có vấn đề).
Bỏ qua: warn trên ảnh không có trong stems (dataset-level, ảnh đã xoá)
và warn trên ảnh đã được người duyệt.
"""
from pipeline.checks._patch_issue import patch_issue_row
patch_issue_row()


_DEFAULT_WARN_RATIO = 0.1

_RATIO_KEYS = (
    "warn_ratio_max",
    "max_warn_ratio",
    "warn_ratio_threshold",
    "warn_ratio",
    "threshold",
)


# ─────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────

def _get(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _threshold(cfg):
    candidates = []
    if cfg is not None:
        if isinstance(cfg, dict):
            candidates.append(cfg.get("gate"))
        else:
            candidates.append(getattr(cfg, "gate", None))
        candidates.append(cfg)

    for c in candidates:
        if c is None:
            continue
        for key in _RATIO_KEYS:
            v = _get(c, key)
            if v is not None:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    continue
    return _DEFAULT_WARN_RATIO


# ─────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────

def decide(issues, stems, cfg=None, reviewed=None):
    issues = issues or []
    stems = list(stems or [])
    reviewed = set(reviewed or [])
    stems_set = set(stems)
    n_images = len(stems)
    threshold = _threshold(cfg)

    n_errors = 0
    error_images = set()
    warn_images = set()

    for iss in issues:
        sev = getattr(iss, "severity", None)
        img = getattr(iss, "image", None)

        if sev == "error":
            n_errors += 1
            if img is not None:
                error_images.add(img)
            continue

        if sev == "warn":
            if img is None:
                continue
            if img not in stems_set:
                continue
            if img in reviewed:
                continue
            warn_images.add(img)

    n_warn_images = len(warn_images)
    warn_ratio = (n_warn_images / n_images) if n_images > 0 else 0.0

    reasons = []

    if n_errors > 0:
        decision = "Rework"
        reasons.append(f"{n_errors} error issue(s) — không thể Release")
        if error_images:
            sample = sorted(error_images)[:5]
            reasons.append(
                f"ảnh bị ảnh hưởng (mẫu): {', '.join(sample)}"
                + (" ..." if len(error_images) > 5 else ""))
    elif n_images > 0 and warn_ratio >= threshold:
        decision = "Hold"
        reasons.append(
            f"warn_ratio {warn_ratio:.2%} >= threshold {threshold:.2%} "
            f"({n_warn_images}/{n_images} ảnh)")
    else:
        decision = "Release"
        if n_warn_images == 0:
            reasons.append("không có cảnh báo warn chưa duyệt")
        else:
            reasons.append(
                f"warn_ratio {warn_ratio:.2%} < threshold {threshold:.2%} "
                f"({n_warn_images}/{n_images} ảnh)")

    return {
        "decision": decision,
        "n_errors": n_errors,
        "error_images": sorted(error_images),
        "n_warn_images": n_warn_images,
        "warn_images": sorted(warn_images),
        "n_images": n_images,
        "warn_ratio": warn_ratio,
        "threshold": threshold,
        "reasons": reasons,
    }