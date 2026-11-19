"""pipeline/checks/second_opinion.py

Kiểm tra NGỮ NGHĨA bằng cách đối chiếu nhãn với model predictions.
Chỉ CẢNH BÁO (severity="warn") — không sửa, không từ chối dữ liệu.

4 mã lỗi:
    POSSIBLE_WRONG_CLASS  — pred trùng box nhãn khác class (IoU >= iou_match)
    POSSIBLE_MISSING      — pred không khớp nhãn nào, đủ conf + đủ lớn
    POSSIBLE_LOOSE_BOX    — pred cùng class nhưng IoU lỏng [iou_missing, iou_loose)
    POSSIBLE_FALSE_POS    — nhãn không được pred nào phủ (IoU < iou_missing)
"""
import math
from pathlib import Path

from pipeline.checks._patch_issue import patch_issue_row
from pipeline.common import Issue

patch_issue_row()


# ─────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _iter_images(ds):
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
        yield image_id, w, h


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


def _get_so_cfg(cfg):
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg.get("second_opinion", {}) or {}
    return getattr(cfg, "second_opinion", {}) or {}


def _get_num_classes(cfg):
    if isinstance(cfg, dict):
        classes = cfg.get("classes")
        if classes:
            return len(classes)
        return int(cfg.get("num_classes", cfg.get("nc", 4)))
    classes = getattr(cfg, "classes", None)
    if classes:
        return len(classes)
    return int(getattr(cfg, "num_classes", 4))


# ─────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────

def run(ds, cfg=None, preds=None):
    preds = preds or {}
    so = _get_so_cfg(cfg)

    conf         = float(so.get("conf",         0.5))
    fp_conf      = float(so.get("fp_conf",      0.25))
    min_area_px  = float(so.get("min_area_px",  600.0))
    iou_match    = float(so.get("iou_match",    0.5))
    iou_missing  = float(so.get("iou_missing",  0.3))
    iou_loose    = float(so.get("iou_loose",    0.6))
    num_classes  = _get_num_classes(cfg)

    labels_dir = getattr(ds, "labels_dir", None)
    if labels_dir is None:
        return []
    labels_dir = Path(labels_dir)

    issues = []

    for image_id, w_img, h_img in _iter_images(ds):
        label_path = labels_dir / f"{image_id}.txt"
        if not label_path.is_file():
            continue

        text = label_path.read_text(encoding="utf-8", errors="replace")
        labels = []
        for line_no, raw in enumerate(text.splitlines(), start=1):
            if not raw.strip():
                continue
            parsed = _parse_line(raw)
            if parsed is None:
                continue
            cls, x, y, w, h = parsed
            if not (0 <= cls < num_classes):
                continue
            labels.append((line_no, cls, x, y, w, h))

        image_preds = preds.get(image_id) or []
        if not labels or not image_preds:
            continue

        W = float(w_img) if w_img else 640.0
        H = float(h_img) if h_img else 480.0

        label_covered = [False] * len(labels)

        for pred in image_preds:
            if len(pred) < 6:
                continue
            try:
                pcls, pconf = int(pred[0]), float(pred[1])
                px, py, pw, ph = (float(pred[i]) for i in range(2, 6))
            except (TypeError, ValueError):
                continue
            if not (0 <= pcls < num_classes):
                continue

            best_same = 0.0
            best_diff = 0.0
            best_any  = 0.0

            for li, (_, lcls, lx, ly, lw, lh) in enumerate(labels):
                iou = _iou((px, py, pw, ph), (lx, ly, lw, lh))
                if iou > best_any:
                    best_any = iou
                if lcls == pcls:
                    if iou > best_same:
                        best_same = iou
                else:
                    if iou > best_diff:
                        best_diff = iou
                if pconf >= fp_conf and iou >= iou_missing:
                    label_covered[li] = True

            if best_same >= iou_match:
                continue

            area_px = (pw * W) * (ph * H)

            if best_diff >= iou_match and pconf >= conf:
                issues.append(Issue(
                    check="second_opinion",
                    severity="warn",
                    code="POSSIBLE_WRONG_CLASS",
                    image=image_id,
                    message=(f"pred class {pcls} (conf={pconf:.2f}) "
                             f"trùng box nhãn class khác (IoU={best_diff:.2f})"),
                    line=None,
                ))
            elif pconf >= conf and area_px >= min_area_px and best_any < iou_missing:
                issues.append(Issue(
                    check="second_opinion",
                    severity="warn",
                    code="POSSIBLE_MISSING",
                    image=image_id,
                    message=(f"pred class {pcls} (conf={pconf:.2f}, "
                             f"area={area_px:.0f}px²) không khớp nhãn nào"),
                    line=None,
                ))
            elif best_same >= iou_missing and best_same < iou_loose and pconf >= conf:
                issues.append(Issue(
                    check="second_opinion",
                    severity="warn",
                    code="POSSIBLE_LOOSE_BOX",
                    image=image_id,
                    message=(f"pred class {pcls} khớp nhãn nhưng box lỏng "
                             f"(IoU={best_same:.2f})"),
                    line=None,
                ))

        for li, covered in enumerate(label_covered):
            if not covered:
                line_no = labels[li][0]
                issues.append(Issue(
                    check="second_opinion",
                    severity="warn",
                    code="POSSIBLE_FALSE_POS",
                    image=image_id,
                    message=f"nhãn dòng {line_no} không được pred nào phủ",
                    line=line_no,
                ))

    return issues