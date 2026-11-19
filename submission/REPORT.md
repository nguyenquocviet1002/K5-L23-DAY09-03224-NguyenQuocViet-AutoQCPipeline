# Day 09 Lab — Nguyễn Quốc Việt: 03224 - Nguyễn Quốc Việt

## 1. Kết luận trước
Pipeline kết luận **Release**. Quét 224 ảnh raw → loại 31 ảnh ở intake (`BAD_SIZE` 3, `CORRUPT_IMAGE` 2, `DUP_IMAGE_EXACT` 8, `DUP_IMAGE_NEAR` 7, `LEAK_TEST` 11) → **193 ảnh vào QC**, **193 ảnh ra dataset** `submission/export/clean_v1` (`submission/reports/08_summary.json:3-9,18`; `submission/export/clean_v1/manifest.json`). Không còn error, tỉ lệ warn chưa duyệt = **0.0 < 0.1**, exit code **0** theo ánh xạ trong `pipeline/run.py:26,187`. `07_error_list_v3.csv` vẫn còn 2 dòng `BOX_HUGE` (`img_0128`, `img_0196`) nhưng cả 2 ảnh đã nằm trong 52 ảnh được duyệt, nên gate bỏ qua (`pipeline/steps/gate.py:89-96`).

## 2. Lỗi và phân việc
Lấy từ `02_error_list_v1` (gom theo mã lỗi). Cột "ai sửa": **máy** (autofix) hay **người** (hàng đợi); chỉ ghi những mã có mặt trong lô của bạn. Bảng dưới gồm 68 dòng QC; 31 dòng còn lại của `02_error_list_v1` là lỗi intake nêu ở mục 1, bị loại tại `steps/intake.py` trước khi autofix chạy.

| Mã lỗi | Số ảnh/dòng | Ai sửa (máy/người) | Vì sao |
|---|---:|---|---|
| BAD_FORMAT | 3 | **Máy** — `drop_line` | Dòng nhãn sai cú pháp/số `nan`; xóa dòng hỏng (`autofix.py:23,251-252`). |
| BAD_CLASS_ID | 6 | **Máy** — `drop_line` | `id` ngoài `data.yaml names`; xóa dòng (`autofix.py:23`). |
| BOX_ZERO | 4 | **Máy** — `drop_line` | Box có `w` hoặc `h` bằng 0, không hợp lệ (`label_format.py:194-199`). |
| BAD_CLASS_NAME | 2 | **Máy** — `rename_classes` | Tên lớp hoa/thường sai (`Person`, `Bus`); máy chuẩn hóa lại `data.yaml` (`autofix.py:78-111`). |
| MISSING_LABEL | 6 | **Người** — gán nhãn | Ảnh có file nhưng nội dung thiếu nhãn; máy không bịa nhãn (`autofix.py:222-233`). |
| MISSING_LABEL_FILE | 6 | **Người** — gán nhãn | Không có file `.txt` (`img_0031, 0053, 0126, 0135, 0170, 0219`); chỉ phát `needs_label`. |
| BOX_OUT_OF_IMAGE | 9 | **Người** — xác nhận | Dù nằm trong nhóm có thể `clip_box`, mã này phát ở mức **warn** nên không tự sửa (`label_format.py:201-209`). |
| BOX_TINY | 6 | **Người** — xác nhận | Box rất nhỏ theo ngưỡng px; cần người quyết giữ hay bỏ (`label_format.py:211-218`). |
| DUP_BOX | 7 | **Người** — xác nhận | Hai box chồng nhau, phát warn nên không tự gộp (`label_format.py:226-234`). |
| BOX_HUGE | 4 | **Người** — xác nhận | Box chiếm gần toàn ảnh, có thể là nhãn máy lỗi (`label_format.py:220-224`). |
| EMPTY_LABEL | 11 | **Người** — xác nhận | File nhãn rỗng; ảnh có thể thực sự không có đối tượng (`missing_labels.py:14-16`). |
| CLASS_RARE | 4 | **Người** — xác nhận | Lớp xuất hiện quá ít, dễ là gán nhầm (`stats.py:24-26`). |

**Vì sao `05_review_queue.csv` xếp ảnh đầu hàng như vậy**: ảnh được ưu tiên theo **điểm ưu tiên có trọng số giảm dần** (`review.weights`), hòa theo tên ảnh — không phải số lượng cảnh báo (`pipeline/steps/review_queue.py:10-24`). Trọng số `NEEDS_LABEL = 10` đẩy 6 ảnh thiếu nhãn lên đầu (`img_0031`…), tiếp theo là `3.0` (`EMPTY_LABEL;FIXED_DROP_LINE`), `2.0` (`BOX_OUT_OF_IMAGE;DUP_BOX`…), rồi các warn `1.0 / 0.5 / 0.3`. **52 ảnh** trong hàng đợi đều được duyệt (52 < 80) — không ảnh nào bị bỏ sót.

**Với ngân sách 80 ảnh**: duyệt được **52 / 52** ảnh (**100%**). Tỉ lệ warn cuối = **0%** < ngưỡng 10% → quyết định **RELEASE**.

## 3. Calibration (2 ngưỡng)
| Tham số | Khởi đầu | Đã chọn | Precision / Recall (trước → sau) | Lý do (đánh đổi) |
|---|---|---|---|---|
| `duplicates.phash_dist` | 2 | 16 | EXACT `1.00/1.00 → 1.00/1.00`; NEAR `1.00/0.50 → 1.00/0.83` | F1 NEAR tăng `0.67 → 0.91` mà precision vẫn `1.00`; ở 20 thì EXACT recall tụt còn `0.33` nên không chọn. |
| `second_opinion.conf` | 0.25 | 0.5 | `0.00/0.00 → 0.00/0.00` (mọi ngưỡng 0.1 → 0.8) | Không có `preds` nên không tối ưu được bằng số liệu; giữ 0.5 như điểm cân bằng lý thuyết giữa precision/recall. |

**Tập calib không cung cấp `preds` (model predictions) cho `second_opinion`**. Do đó, **cả 7 ngưỡng `conf` từ 0.1 → 0.8 đều cho 0/0/0** — không thể tối ưu ngưỡng này bằng calib.

**Bằng chứng**:
1. Bảng sweep `second_opinion.conf` trong `submission/calibration/calibration_report.md:36-46` — mọi ô của 4 mã `POSSIBLE_*` đều `0.00/0.00/0.00`.
2. Trong pipeline thật chạy trên 224 ảnh raw: **không có bất kỳ mã `POSSIBLE_*` nào** trong `02_error_list_v1.csv` / `04_error_list_v2.csv` / `05_review_queue.csv`. Nguyên nhân: `checks/__init__.py:17-21` và `calibrate.py:52` chỉ gọi `run(ds, cfg)`, không truyền `preds`; `second_opinion.run` nhận `preds = {}` và `continue` ngay khi ảnh không có preds (`second_opinion.py:109-110,146-148`). Dù `data/cache/checker_preds.json` có sẵn 261 khóa, `model_cache.get_preds` không được gọi ở đâu cả.

**Hệ quả**: giữ `conf = 0.5` là quyết định **theo lý thuyết**, không phải theo calibration. Cần nối `preds` vào `qc()` để module này hoạt động thực sự, rồi sweep lại ngưỡng.

## 4. Cửa chặn và điều pipeline KHÔNG bắt được
**(a) Chứng minh gate chặn đúng.** Đã chạy `python -m pipeline.run --budget 0 --reports work/rep_nb --export-dir work/exp_nb`:

- `work/rep_nb/08_summary.json:3-13` → `decision = Rework`, `n_errors = 12` (6 `MISSING_LABEL_FILE` + 6 `MISSING_LABEL` trên cùng 6 ảnh `img_0031, 0053, 0126, 0135, 0170, 0219`), `n_warn_images = 36`, `warn_ratio = 0.1865` (36/193) ≥ ngưỡng `0.1`.
- `06_review_log.csv` chỉ có header, `reviewed = 0` vì `budget = 0` → `human_review_sim.py:25` lát `rows[:0]`; `03b_fixes_after_review.csv` vẫn giữ 6 dòng `needs_label`; `07_error_list_v3.csv` giữ nguyên 12 error như v2 và `07_error_list_v3.txt:69` kết luận `REWORK`.
- Gate trả **Rework** ngay khi còn error, bất kể đã duyệt hay chưa (`pipeline/steps/gate.py:89-96,103-105`); **exit code 20** theo `pipeline/run.py:26,187` và `docs/PIPELINE.md:66-69` (không có file log riêng, nhưng ánh xạ là cố định trong code).
- `exported: null` trong summary và **`work/exp_nb` không tồn tại** chứng tỏ pipeline **không xuất dataset** (`pipeline/run.py:174-182`). `work/run/ds` được dựng lại từ raw mỗi lần chạy (`run.py:88-95`) nên không sửa tay. Đây là phép thử đối chứng với lô thật: budget 80 → 52 ảnh được duyệt → 0 warn chưa duyệt → Release.

**(b) Những điểm pipeline KHÔNG bắt được (có bằng chứng):**

1. **Box lỏng nhẹ / IoU mờ**: `label_format.py:3` chỉ kiểm cú pháp, không phán xét ngữ nghĩa. `second_opinion.py:183-184` bỏ qua pred nếu `best_same ≥ iou_match = 0.5` (`configs/pipeline.yaml:30-31`), nên box phình vừa phải vẫn “sạch” — và hiện module này còn không chạy vì thiếu `preds` (mục 3). Ảnh `img_0128`, `img_0196` chỉ bị bắt ở mức `BOX_HUGE` rõ rệt, không phải box lỏng nhẹ.
2. **Nhầm lớp mà cả checker cũng nhầm**: nếu nhãn và pred cùng trả một lớp sai thì dòng 183-184 skip; ngược lại, checker sai và tự tin sẽ sinh `POSSIBLE_WRONG_CLASS` giả. Calib không đại diện cho raw: `docs/CALIBRATION.md:39` ghi calib nghiêng `person→car`, raw nghiêng `car/truck/bus` và calib không có `LEAK_TEST`; ánh xạ lớp của checker còn là phỏng đoán cứng tại `model_cache.py:15`.
3. **Ảnh trùng biến đổi mạnh**: phát hiện chỉ bằng khoảng cách Hamming pHash ≤ `duplicates.phash_dist = 16` (`duplicates.py:31,55-57`). `docs/CALIBRATION.md:40` nêu rõ ca đã biết “cắt 80% rồi thu nhỏ” không ngưỡng nào bắt được; ở 16 calib vẫn còn 1 FN `DUP_IMAGE_NEAR` (`calibration_report.md:33`), trong khi tăng lên 20 lại gộp nhầm ảnh khác.
4. **Ngưỡng cấu hình không được đọc đúng**: `label_format.py:43-49,62-67` thiếu alias cho `min_box_px`, `max_box_frac`, `dup_box_iou` (`configs/pipeline.yaml:17-19`), nên thực tế dùng `huge_area = 0.9` và `dup_iou = 0.8` thay vì `0.95` / `0.9` khai báo. Cảnh báo vì thế có thể lệch ngay cả khi pipeline “sạch”. Nói cách khác, gate chỉ chặn tín hiệu mà checker phát ra; checker im lặng thì không có tín hiệu để chặn.

## 5. Train: raw vs clean (tùy chọn, chấm riêng)
**Chưa chạy.** `submission/results/` chỉ có `.gitkeep`: không có `comparison.md`, không có file `*_seed*.json`, không có trọng số hay thư mục `runs/`. `notebooks/day09_train_compare.ipynb` còn `execution_count: null`, `outputs: []` và `REPO_URL` vẫn là placeholder. Vì không có kết quả train nên không có mAP50 để so với độ lệch chuẩn giữa các seed; theo template và `RUBRIC.md:37`, ghi “chưa chạy” rồi bỏ qua phần còn lại của mục này (không bịa số liệu, `RULES.md:22`).

## 6. Quyết định và ghi chú AI
**(a) Quyết định cuối: Release.** Điều kiện giữ quyết định này: `n_errors = 0`, 52/52 ảnh hàng đợi đã duyệt, warn chưa duyệt `0% < 10%` và export `clean_v1` tồn tại. Nếu xuất hiện lại error bắt buộc (ví dụ 6 ảnh thiếu nhãn chưa được gán) → **Rework**; nếu warn chưa duyệt chiếm ≥ 10% → **Hold**. Để chuyển Hold/Rework về Release: gán nhãn đủ 6 ảnh, duyệt hết ảnh còn warn trong ngân sách 80, nối `preds` vào checker, sửa alias đọc ngưỡng, rồi chạy lại gate và export.

**(b) Ghi chú dùng AI (DeepSeek):**
- Prompt chính: dán docstring của `pipeline/checks/label_format.py` + 1 unit test vào DeepSeek, nhờ viết hàm Python chỉ đọc `labels/` (YOLO `id cx cy w h`), in `tên file, số dòng, lý do` cho box tràn ảnh, giải thích từng dòng bằng tiếng Việt, không sửa/xóa file (theo mẫu `docs/PROMPTING.md:4-8`).
- Cách kiểm code AI: đọc diff từng dòng, chạy `python -m pytest tests/`, chạy pipeline trên `data/calib` có lỗi chèn sẵn rồi đối chiếu mã với `submission/calibration/calibration_report.md`; không cho AI đụng vào file giảng viên.
- Lần AI sai: DeepSeek báo `pipeline/common.py` thiếu `Issue.row()` và sinh `pipeline/checks/_patch_issue.py` để patch runtime; thực tế `pipeline/common.py:33` đã định nghĩa sẵn method này, nên patch chỉ là no-op — tôi phát hiện bằng `grep "def row"` và ghi chú lại, không dùng nhận định sai đó để sửa file gốc.

*(Thưởng +2, tùy chọn)* CVAT: **chưa làm** — repo không có export XML/zip hay ảnh chụp phiên lọc, nên không đối chiếu được `width < 20` với các dòng `BOX_TINY` cũng như không có nhận xét thực tế về `can_review`; không nhận +2.
