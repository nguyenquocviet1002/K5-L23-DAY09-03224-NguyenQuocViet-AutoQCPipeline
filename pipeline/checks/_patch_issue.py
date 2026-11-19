"""Pipeline compatibility patch — KHÔNG sửa file giảng viên.

Starter code `pipeline/common.py` thiếu method `Issue.row()` mà
`pipeline/steps/report.py::write_issues_csv` cần để ghi CSV.
Ta bổ sung method này tại RUNTIME (không sửa common.py) để tôn trọng
file gốc của giảng viên.

Cách dùng: trong mọi module của mình, gọi `patch_issue_row()` khi import.
Idempotent: nếu `Issue.row` đã tồn tại (starter cập nhật) → no-op.
"""
import json


def patch_issue_row():
    """Thêm method `Issue.row()` vào `pipeline.common.Issue` nếu chưa có."""
    from pipeline.common import Issue

    if hasattr(Issue, "row"):
        return  # starter đã có → không override

    def _row(self):
        """Serialize Issue → dict cho csv.DictWriter (7 cột report.py cần)."""
        return {
            "check":    getattr(self, "check",    ""),
            "severity": getattr(self, "severity", ""),
            "code":     getattr(self, "code",     ""),
            "image":    getattr(self, "image",    ""),
            "line":     "" if getattr(self, "line", None) is None else self.line,
            "message":  getattr(self, "message",  ""),
            "extra":    json.dumps(self.extra, ensure_ascii=False)
                        if getattr(self, "extra", None) else "",
        }

    Issue.row = _row