import json
from pathlib import Path

files = [
    ("E0", "zero-shot", "e0.json"),
    ("E1", "LoRA LLM", "e1.json"),
    ("E2", "+ projector", "e2.json"),
    ("E3", "+ LoRA ViT", "e3.json"),
    ("E4", "rank r=4", "e4_rank_4.json"),
    ("E4", "rank r=64", "e4_rank_64.json"),
    ("E5", "GRPO (reward IoU)", "e5.json"),
]

for code, setup, fname in files:
    d = json.load(open(Path("report/evidence") / fname, encoding="utf-8"))
    m = d["micro"]
    ci = d.get("f1_ci95", [0, 0])
    pc = d["per_class"]
    meta = d.get("meta", {})
    t_params = meta.get("trainable_params", 0)
    params_str = f"{t_params:,}" if t_params else "0"
    vram_str = f"{meta['peak_vram_gb']:.1f}" if "peak_vram_gb" in meta else "-"
    time_str = f"{meta['train_minutes']:.1f}" if "train_minutes" in meta else "-"
    
    line = f"| {code} | {setup} | {params_str} | {vram_str} | {time_str} | {m['f1']:.3f} | {ci[0]:.3f}–{ci[1]:.3f} | {d['fp_per_image']:.2f} | {pc['pothole']['f1']:.3f} | {pc['crack']['f1']:.3f} | {pc['manhole']['f1']:.3f} |"
    print(line)
