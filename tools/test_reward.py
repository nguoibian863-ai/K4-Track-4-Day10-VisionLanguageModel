"""Kiểm thử hàm reward IoU cho E5 GRPO."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.labcommon import CLASSES
from scripts.score import parse_output, iou_xyxy


def reward_fn(completion, gt_boxes):
    pred = parse_output(completion)
    if pred is None:
        return 0.0
    if not gt_boxes and not pred:
        return 1.0

    # Ghép đôi tham lam theo IoU giảm dần, cùng lớp, ngưỡng IoU >= 0.5
    ious = []
    for lab in set(CLASSES) | {p["label"] for p in pred}:
        g = [x["bbox"] for x in gt_boxes if x["label"] == lab]
        p = [x["bbox"] for x in pred if x["label"] == lab]
        pairs = sorted(((iou_xyxy(a, b), i, j) for i, a in enumerate(g) for j, b in enumerate(p)), reverse=True)
        used_g, used_p = set(), set()
        for v, i, j in pairs:
            if v < 0.5:
                break
            if i in used_g or j in used_p:
                continue
            used_g.add(i)
            used_p.add(j)
            ious.append(v)

    total_iou = sum(ious)
    denom = max(len(gt_boxes), len(pred), 1)
    return 0.2 + 0.8 * (total_iou / denom)


def test_reward():
    gt = [{"label": "pothole", "bbox": (100.0, 100.0, 300.0, 300.0)}]

    # 1. Dự đoán hoàn hảo (IoU = 1.0)
    perf = '[{"label": "pothole", "bbox_2d": [100, 100, 300, 300]}]'
    r_perf = reward_fn(perf, gt)
    print("1. Perfect match reward:", r_perf)
    assert abs(r_perf - 1.0) < 1e-4

    # 2. JSON hỏng (malformed)
    broken = "not valid json text"
    r_broken = reward_fn(broken, gt)
    print("2. Broken JSON reward:", r_broken)
    assert r_broken == 0.0

    # 3. Sai lớp (crack thay vì pothole)
    wrong_cls = '[{"label": "crack", "bbox_2d": [100, 100, 300, 300]}]'
    r_wrong = reward_fn(wrong_cls, gt)
    print("3. Wrong class reward:", r_wrong)
    assert abs(r_wrong - 0.2) < 1e-4

    # 4. Cả gt và pred đều rỗng
    r_empty = reward_fn("[]", [])
    print("4. Both empty reward:", r_empty)
    assert r_empty == 1.0

    # 5. Thừa box bịa đặt (False Positive bị phạt qua mẫu số)
    fp = '[{"label": "pothole", "bbox_2d": [100, 100, 300, 300]}, {"label": "crack", "bbox_2d": [500, 500, 600, 600]}]'
    r_fp = reward_fn(fp, gt)
    print("5. FP penalty reward:", r_fp)
    assert abs(r_fp - (0.2 + 0.8 * (1.0 / 2))) < 1e-4

    print("=> Tất cả unit test cho reward_fn thành công 100%!")


if __name__ == "__main__":
    test_reward()
