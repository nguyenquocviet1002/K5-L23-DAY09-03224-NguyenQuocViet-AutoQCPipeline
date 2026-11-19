"""pipeline/steps/autofix.py

Tự động sửa lỗi CÚ PHÁP (severity="error"). KHÔNG đụng vào warn.

Actions:
    drop_line      — xoá dòng khỏi file nhãn
    clip_box       — clamp box về [0,1]
    rename_classes — sửa names trong data.yaml
    remove_image   — xoá cả ảnh + file nhãn
    needs_label    — báo người, KHÔNG tạo file rỗng

Ưu tiên cùng dòng: drop_line > clip_box.
"""
from collections import defaultdict
from pathlib import Path

from pipeline.checks._patch_issue import patch_issue_row
patch_issue_row()


_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

_DROP_LINE_CODES    = {"BAD_FORMAT", "BAD_CLASS_ID", "BOX_ZERO", "DUP_BOX"}
_CLIP_CODES         = {"BOX_OUT_OF_IMAGE"}
_REMOVE_IMAGE_CODES = {"DUP_IMAGE_EXACT", "LEAK_TEST"}
_RENAME_CODES       = {"BAD_CLASS_NAME"}
_NEEDS_LABEL_CODES  = {"MISSING_LABEL_FILE", "MISSING_LABEL"}

_ACTION_PRIORITY = {"clip_box": 1, "drop_line": 2}


# ─────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────

def _get(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _box_fully_outside(xc, yc, w, h):
    x_min, x_max = xc - w / 2, xc + w / 2
    y_min, y_max = yc - h / 2, yc + h / 2
    if x_max <= 0 or x_min >= 1:
        return True
    if y_max <= 0 or y_min >= 1:
        return True
    return False


def _clip_box_line(line):
    parts = line.split()
    if len(parts) != 5:
        return line
    try:
        cls = int(parts[0])
        xc, yc, w, h = (float(p) for p in parts[1:])
    except ValueError:
        return line

    x_min, x_max = xc - w / 2, xc + w / 2
    y_min, y_max = yc - h / 2, yc + h / 2

    x_min_c, x_max_c = max(0.0, x_min), min(1.0, x_max)
    y_min_c, y_max_c = max(0.0, y_min), min(1.0, y_max)

    new_w = x_max_c - x_min_c
    new_h = y_max_c - y_min_c
    new_xc = (x_min_c + x_max_c) / 2
    new_yc = (y_min_c + y_max_c) / 2

    return f"{cls} {new_xc:.6f} {new_yc:.6f} {new_w:.6f} {new_h:.6f}"


def _rename_classes_in_yaml(ds, cfg):
    classes = _get(cfg, "classes")
    if not classes:
        return False
    yaml_path = getattr(ds, "yaml_path", None)
    if yaml_path is None:
        return False
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        return False

    try:
        import yaml
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        data["names"] = list(classes)
        yaml_path.write_text(
            yaml.safe_dump(data, allow_unicode=True, sort_keys=False,
                           default_flow_style=False),
            encoding="utf-8",
        )
        return True
    except Exception:
        try:
            import re
            text = yaml_path.read_text(encoding="utf-8")
            text = re.sub(r"^names\s*:.*?(?=^\S|\Z)", "",
                          text, flags=re.MULTILINE | re.DOTALL)
            names_yaml = "names:\n" + "\n".join(
                f"  - {c}" for c in classes) + "\n"
            yaml_path.write_text(text.rstrip() + "\n" + names_yaml,
                                 encoding="utf-8")
            return True
        except Exception:
            return False


def _remove_image_files(ds, image_id):
    removed = False
    images_dir = getattr(ds, "images_dir", None)
    if images_dir and Path(images_dir).is_dir():
        for p in Path(images_dir).iterdir():
            if p.is_file() and p.stem == image_id:
                try:
                    p.unlink()
                    removed = True
                except OSError:
                    pass
    labels_dir = getattr(ds, "labels_dir", None)
    if labels_dir:
        lp = Path(labels_dir) / f"{image_id}.txt"
        if lp.exists():
            try:
                lp.unlink()
                removed = True
            except OSError:
                pass
    return removed


def _resolve_label_path(ds, image_id):
    if hasattr(ds, "label_path"):
        try:
            return Path(ds.label_path(image_id))
        except Exception:
            pass
    labels_dir = getattr(ds, "labels_dir", None)
    if labels_dir:
        return Path(labels_dir) / f"{image_id}.txt"
    return None


# ─────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────

def fix(ds, issues, cfg=None):
    fixes = []
    if not issues:
        return fixes

    # ── Bước 1: lọc error, gom theo image ──
    by_image = defaultdict(list)
    for iss in issues:
        if getattr(iss, "severity", None) != "error":
            continue
        img = getattr(iss, "image", None)
        if img is None:
            continue
        by_image[img].append(iss)

    if not by_image:
        return fixes

    # ── Bước 2: xác định image cần xoá, yaml cần rename ──
    images_to_remove = set()
    rename_needed = False
    for img, img_issues in by_image.items():
        for iss in img_issues:
            code = getattr(iss, "code", None)
            if code in _REMOVE_IMAGE_CODES:
                images_to_remove.add(img)
            elif code in _RENAME_CODES:
                rename_needed = True

    # ── Bước 3: rename yaml (file-level) ──
    if rename_needed:
        before = None
        yaml_path = getattr(ds, "yaml_path", None)
        if yaml_path and Path(yaml_path).exists():
            try:
                import yaml
                before = yaml.safe_load(
                    Path(yaml_path).read_text(encoding="utf-8"))
            except Exception:
                pass
        _rename_classes_in_yaml(ds, cfg)
        fixes.append({
            "image": "data.yaml",
            "line": None,
            "action": "rename_classes",
            "before": before,
            "after": _get(cfg, "classes"),
        })

    # ── Bước 4: remove_image ──
    for img in images_to_remove:
        _remove_image_files(ds, img)
        fixes.append({
            "image": img,
            "line": None,
            "action": "remove_image",
            "before": img,
            "after": None,
        })

    # ── Bước 5: line-level cho ảnh còn lại ──
    for img, img_issues in by_image.items():
        if img in images_to_remove:
            continue

        label_path = _resolve_label_path(ds, img)
        if label_path is None:
            continue

        if not label_path.exists():
            for iss in img_issues:
                if getattr(iss, "code", None) in _NEEDS_LABEL_CODES:
                    fixes.append({
                        "image": img,
                        "line": None,
                        "action": "needs_label",
                        "before": None,
                        "after": None,
                    })
                    break
            continue

        text = label_path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()

        line_actions = {}

        for iss in img_issues:
            line_no = getattr(iss, "line", None)
            code = getattr(iss, "code", None)
            if line_no is None or code is None:
                continue
            if line_no < 1 or line_no > len(lines):
                continue

            raw = lines[line_no - 1]
            action = None

            if code in _DROP_LINE_CODES:
                action = "drop_line"
            elif code in _CLIP_CODES:
                parts = raw.split()
                if len(parts) == 5:
                    try:
                        xc, yc, w, h = (float(parts[i]) for i in range(1, 5))
                        action = ("drop_line"
                                  if _box_fully_outside(xc, yc, w, h)
                                  else "clip_box")
                    except ValueError:
                        action = "drop_line"
                else:
                    action = "drop_line"

            if action is None:
                continue

            prev = line_actions.get(line_no)
            if prev is None or _ACTION_PRIORITY[action] > _ACTION_PRIORITY[prev]:
                line_actions[line_no] = action

        if not line_actions:
            continue

        new_lines = []
        for idx, raw in enumerate(lines, start=1):
            act = line_actions.get(idx)
            if act == "drop_line":
                fixes.append({
                    "image": img, "line": idx, "action": "drop_line",
                    "before": raw, "after": None,
                })
            elif act == "clip_box":
                new_line = _clip_box_line(raw)
                if new_line != raw:
                    fixes.append({
                        "image": img, "line": idx, "action": "clip_box",
                        "before": raw, "after": new_line,
                    })
                    new_lines.append(new_line)
                else:
                    new_lines.append(raw)
            else:
                new_lines.append(raw)

        out = "\n".join(new_lines)
        if out:
            out += "\n"
        label_path.write_text(out, encoding="utf-8")

    return fixes