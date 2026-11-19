"""pipeline/checks/label_format.py

Kiểm tra CÚ PHÁP file nhãn YOLO. KHÔNG phán xét ngữ nghĩa.

Dùng chung `pipeline.common.Issue` — patch `Issue.row()` qua _patch_issue.

8 mã lỗi:
    BAD_FORMAT        — ≠5 field, không phải số, hoặc nan/inf
    BAD_CLASS_ID      — class ngoài [0, num_classes-1]
    BOX_ZERO          — w <= 0 hoặc h <= 0
    BOX_OUT_OF_IMAGE  — box tràn > tol; extra["overflow"] = max tràn
    BOX_TINY          — cạnh ngắn (pixel) < min_side_px
    BOX_HUGE          — diện tích >= huge_area
    DUP_BOX           — trùng box TRƯỚC cùng class (IoU > dup_iou)
    MISSING_LABEL     — không có file .txt

QUAN TRỌNG — determinism:
    Mọi message KHÔNG được chứa absolute path (tmp_path thay đổi giữa
    các lần chạy → CSV khác nhau → test_deterministic fail).
    Chỉ dùng image_id / tên file thuần.
"""
import math
from pathlib import Path

from pipeline.checks._patch_issue import patch_issue_row
from pipeline.common import Issue

patch_issue_row()


# ─────────────────────────────────────────────────────────────
# Cấu hình
# ─────────────────────────────────────────────────────────────

_DEFAULTS = {
    "num_classes":  4,
    "overflow_tol": 0.001,
    "min_side_px":  12.0,
    "huge_area":    0.9,
    "dup_iou":      0.8,
}

_ALIAS = {
    "num_classes":  ("num_classes", "n_classes", "nc", "num_cls"),
    "overflow_tol": ("overflow_tol", "tol", "bound_tol"),
    "min_side_px":  ("min_side_px", "tiny_px", "min_side"),
    "huge_area":    ("huge_area", "max_area"),
    "dup_iou":      ("dup_iou", "iou_threshold"),
}

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _opt(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _cfg(cfg, key):
    for k in _ALIAS[key]:
        v = _opt(cfg, k, None)
        if v is not None:
            return v
    return _DEFAULTS[key]


# ─────────────────────────────────────────────────────────────
# Quét ảnh trong ds
# ─────────────────────────────────────────────────────────────

def _iter_images(ds):
    """Yield (image_id, width, height) bằng cách quét ds.images_dir."""
    images_dir = getattr(ds, "images_dir", None)
    if images_dir is None:
        return
    images_dir = Path(images_dir)
    if not images_dir.is_dir():
        return

    size_method = getattr(ds, "image_size", None)

    for img_path in sorted(images_dir.iterdir()):
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in _IMG_EXTS:
            continue

        image_id = img_path.stem
        w, h = None, None
        if callable(size_method):
            for arg in (image_id, img_path):
                try:
                    r = size_method(arg)
                except Exception:
                    continue
                if r is None:
                    continue
                try:
                    w, h = float(r[0]), float(r[1])
                    break
                except (TypeError, IndexError, ValueError):
                    continue

        if w is None:
            try:
                from PIL import Image
                with Image.open(img_path) as im:
                    w, h = float(im.size[0]), float(im.size[1])
            except Exception:
                pass

        yield image_id, w, h


# ─────────────────────────────────────────────────────────────
# Parse + IoU
# ─────────────────────────────────────────────────────────────

def _parse_line(raw):
    parts = raw.split()
    if len(parts) != 5:
        return None
    try:
        cls = int(parts[0])
        x, y, w, h = (float(p) for p in parts[1:])
    except ValueError:
        return None
    if not all(math.isfinite(v) for v in (x, y, w, h)):
        return None
    return cls, x, y, w, h


def _iou(a, b):
    ax1, ax2 = a[0] - a[2] / 2, a[0] + a[2] / 2
    ay1, ay2 = a[1] - a[3] / 2, a[1] + a[3] / 2
    bx1, bx2 = b[0] - b[2] / 2, b[0] + b[2] / 2
    by1, by2 = b[1] - b[3] / 2, b[1] + b[3] / 2
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


# ─────────────────────────────────────────────────────────────
# Kiểm một file nhãn
# ─────────────────────────────────────────────────────────────

def _check_file(image_id, label_path, w_img, h_img, cfg):
    issues = []
    nc        = int(_cfg(cfg, "num_classes"))
    tol       = float(_cfg(cfg, "overflow_tol"))
    min_px    = float(_cfg(cfg, "min_side_px"))
    huge_area = float(_cfg(cfg, "huge_area"))
    dup_iou   = float(_cfg(cfg, "dup_iou"))

    # ⚠️ DETERMINISM: KHÔNG đưa absolute path vào message — chỉ tên file thuần.
    if label_path is None or not Path(label_path).is_file():
        fname = Path(label_path).name if label_path else f"{image_id}.txt"
        issues.append(Issue(
            check="label_format", severity="error",
            code="MISSING_LABEL", image=image_id,
            message=f"missing label file: {fname}"))
        return issues

    text = Path(label_path).read_text(encoding="utf-8", errors="replace")
    seen = []

    for line_no, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip():
            continue

        parsed = _parse_line(raw)
        if parsed is None:
            issues.append(Issue(
                check="label_format", severity="error",
                code="BAD_FORMAT", image=image_id, line=line_no,
                message=f"cannot parse: {raw!r}"))
            continue

        cls, x, y, w, h = parsed

        if not (0 <= cls < nc):
            issues.append(Issue(
                check="label_format", severity="error",
                code="BAD_CLASS_ID", image=image_id, line=line_no,
                message=f"class {cls} not in [0,{nc-1}]",
                extra={"class_id": cls}))
            continue

        if w <= 0 or h <= 0:
            issues.append(Issue(
                check="label_format", severity="error",
                code="BOX_ZERO", image=image_id, line=line_no,
                message=f"w={w}, h={h} must be > 0"))
            continue

        x_min, x_max = x - w / 2, x + w / 2
        y_min, y_max = y - h / 2, y + h / 2
        overflow = max(0.0, -x_min, x_max - 1.0, -y_min, y_max - 1.0)
        if overflow > tol:
            issues.append(Issue(
                check="label_format", severity="warn",
                code="BOX_OUT_OF_IMAGE", image=image_id, line=line_no,
                message=f"overflow={overflow:.6f} > tol={tol}",
                extra={"overflow": overflow}))

        if w_img and h_img:
            side_px = min(w * w_img, h * h_img)
            if side_px < min_px:
                issues.append(Issue(
                    check="label_format", severity="warn",
                    code="BOX_TINY", image=image_id, line=line_no,
                    message=f"short side {side_px:.2f}px < {min_px}px",
                    extra={"side_px": side_px}))

        if w * h >= huge_area:
            issues.append(Issue(
                check="label_format", severity="warn",
                code="BOX_HUGE", image=image_id, line=line_no,
                message=f"area={w*h:.4f} >= {huge_area}"))

        cur = (x, y, w, h)
        for prev_cls, px, py, pw, ph in seen:
            if prev_cls == cls and _iou(cur, (px, py, pw, ph)) > dup_iou:
                issues.append(Issue(
                    check="label_format", severity="warn",
                    code="DUP_BOX", image=image_id, line=line_no,
                    message=f"duplicate of class-{cls} box on earlier line"))
                break
        seen.append((cls, x, y, w, h))

    return issues


# ─────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────

def run(ds, cfg=None):
    labels_dir = getattr(ds, "labels_dir", None)
    if labels_dir is None:
        return []
    labels_dir = Path(labels_dir)
    issues = []
    for image_id, w_img, h_img in _iter_images(ds):
        issues.extend(_check_file(
            image_id, labels_dir / f"{image_id}.txt", w_img, h_img, cfg))
    return issues