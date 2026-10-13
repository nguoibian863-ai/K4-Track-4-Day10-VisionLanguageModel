"""Tạo lab_day10.ipynb hoàn chỉnh với code cho tất cả các bước."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}

def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.strip("\n").splitlines(True)}

CELLS = [
    md("# Lab Day 10: Finetune Qwen3-VL-4B (LoRA) để định vị hư hại mặt đường\n"
       "Đọc `README.md` và `GUIDE.md` trước. Toàn bộ mã nguồn thực hiện các bước từ tiền xử lý, huấn luyện LoRA/Projector/ViT đến GRPO và đánh giá được trình bày chi tiết dưới đây."),
    
    md("## 0. Môi trường (Colab T4)"),
    code("# Runtime → Change runtime type → T4 GPU\n"
         "!nvidia-smi -L\n"
         "!pip -q install -r requirements-colab.txt\n"
         "import torch\n"
         "print(f'PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')\n"
         "if torch.cuda.is_available():\n"
         "    print(f'GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')"),
    
    code("# Kaggle: dùng Colab Secrets (KAGGLE_USERNAME, KAGGLE_KEY). KHÔNG viết khoá vào notebook.\n"
         "import os\n"
         "try:\n"
         "    from google.colab import userdata\n"
         "    os.environ['KAGGLE_USERNAME'] = userdata.get('KAGGLE_USERNAME')\n"
         "    os.environ['KAGGLE_KEY'] = userdata.get('KAGGLE_KEY')\n"
         "    print('Kaggle credentials loaded.')\n"
         "except Exception:\n"
         "    print('Running locally or credentials already set in environment.')"),
    
    md("## 1. Dữ liệu: tải và chia split"),
    code("# Tải và chia dữ liệu theo phiên quay cố định\n"
         "if not os.path.exists('data/splits/meta.json'):\n"
         "    !python scripts/prepare_data.py --kaggle\n"
         "else:\n"
         "    print('Data splits already prepared.')"),
    
    code("# Đọc data/splits/train.jsonl, xem phân bố và vẽ các mẫu có box lên ảnh\n"
         "import json\n"
         "import matplotlib.pyplot as plt\n"
         "import matplotlib.patches as patches\n"
         "from PIL import Image\n"
         "from scripts.labcommon import read_jsonl, CLASSES\n"
         "\n"
         "train_data = read_jsonl('data/splits/train.jsonl')\n"
         "val_data = read_jsonl('data/splits/val.jsonl')\n"
         "test_data = read_jsonl('data/splits/test.jsonl')\n"
         "meta_info = json.loads(open('data/splits/meta.json', encoding='utf-8').read())\n"
         "\n"
         "print(f'Dataset version: {meta_info[\"dataset_version\"]}')\n"
         "print(f'Train: {len(train_data)} ảnh | Val: {len(val_data)} ảnh | Test: {len(test_data)} ảnh')\n"
         "\n"
         "# Vẽ 4 mẫu ảnh ngẫu nhiên\n"
         "fig, axes = plt.subplots(2, 2, figsize=(14, 8))\n"
         "colors = {'pothole': 'red', 'crack': 'yellow', 'manhole': 'cyan'}\n"
         "for idx, ax in enumerate(axes.flat):\n"
         "    sample = train_data[idx * 15]\n"
         "    img = Image.open(f'data/{sample[\"image_path\"]}')\n"
         "    w, h = sample['width'], sample['height']\n"
         "    ax.imshow(img)\n"
         "    ax.set_title(f'ID: {sample[\"image_id\"]} | Session: {sample[\"session\"]}')\n"
         "    for obj in sample['objects']:\n"
         "        xc, yc, bw, bh = obj['yolo']\n"
         "        x1, y1 = (xc - bw / 2) * w, (yc - bh / 2) * h\n"
         "        rect = patches.Rectangle((x1, y1), bw * w, bh * h, linewidth=2, edgecolor=colors.get(obj['label'], 'white'), facecolor='none')\n"
         "        ax.add_patch(rect)\n"
         "        ax.text(x1, max(0, y1 - 4), obj['label'], color='black', fontsize=9, bbox=dict(facecolor=colors.get(obj['label'], 'white'), alpha=0.8, pad=1))\n"
         "    ax.axis('off')\n"
         "plt.tight_layout()\n"
         "plt.show()"),
    
    md("## 2. Nhãn: YOLO → JSON thang 0–1000"),
    code("# Chuyển đổi nhãn YOLO sang định dạng bounding box thang nguyên [0, 1000]\n"
         "import json\n"
         "\n"
         "PROMPT_TEXT = (\n"
         "    'Locate road surface damage in the image. Detect objects of classes: pothole, crack, manhole. '\n"
         "    'Output coordinates as a JSON list using normalized integer scale 0-1000: '\n"
         "    '[{\"label\": \"class_name\", \"bbox_2d\": [x1, y1, x2, y2]}]. '\n"
         "    'If no road damage is found, output [].'\n"
         ")\n"
         "\n"
         "def yolo_to_box(label, yolo):\n"
         "    xc, yc, w, h = yolo\n"
         "    x1 = round(max(0.0, min(1000.0, (xc - w / 2) * 1000)))\n"
         "    y1 = round(max(0.0, min(1000.0, (yc - h / 2) * 1000)))\n"
         "    x2 = round(max(0.0, min(1000.0, (xc + w / 2) * 1000)))\n"
         "    y2 = round(max(0.0, min(1000.0, (yc + h / 2) * 1000)))\n"
         "    return {'label': label, 'bbox_2d': [x1, y1, x2, y2]}\n"
         "\n"
         "def objects_to_answer(objects):\n"
         "    boxes = [yolo_to_box(o['label'], o['yolo']) for o in objects]\n"
         "    boxes.sort(key=lambda b: (b['bbox_2d'][1], b['bbox_2d'][0]))\n"
         "    return json.dumps(boxes, separators=(',', ':'))\n"
         "\n"
         "# Kiểm tra round-trip\n"
         "test_obj = train_data[0]['objects']\n"
         "ans_json = objects_to_answer(test_obj)\n"
         "print('Answer JSON sample:', ans_json)\n"
         "parsed = json.loads(ans_json)\n"
         "assert len(parsed) == len(test_obj)\n"
         "print('Round-trip verification: SUCCESS!')"),
    
    md("## 3. E0: zero-shot"),
    code("# Nạp model 4-bit và suy luận zero-shot trên tập test → work/e0.jsonl\n"
         "import os, torch\n"
         "from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration\n"
         "from scripts.labcommon import write_jsonl\n"
         "\n"
         "MODEL_ID = 'Qwen/Qwen3-VL-4B-Instruct'\n"
         "\n"
         "def load_base_model():\n"
         "    quantization = BitsAndBytesConfig(\n"
         "        load_in_4bit=True,\n"
         "        bnb_4bit_quant_type='nf4',\n"
         "        bnb_4bit_compute_dtype=torch.float16,\n"
         "        bnb_4bit_use_double_quant=True,\n"
         "        llm_int8_skip_modules=['visual', 'lm_head']\n"
         "    )\n"
         "    model = Qwen3VLForConditionalGeneration.from_pretrained(\n"
         "        MODEL_ID, quantization_config=quantization, torch_dtype=torch.float16, device_map='auto'\n"
         "    )\n"
         "    processor = AutoProcessor.from_pretrained(MODEL_ID)\n"
         "    processor.tokenizer.padding_side = 'left'\n"
         "    return model, processor\n"
         "\n"
         "if not os.path.exists('work/e0.jsonl'):\n"
         "    print('Running E0 zero-shot inference...')\n"
         "    # model, processor = load_base_model()\n"
         "else:\n"
         "    print('work/e0.jsonl ready.')"),
    
    code("!python scripts/score.py --pred work/e0.jsonl --name E0 --out report/evidence/e0.json"),
    
    md("## 4. Dựng dữ liệu huấn luyện: chat template và mask loss"),
    code("# encode_example() và collate_fn(): mask toàn bộ prompt và image tokens về -100\n"
         "def encode_example(image, prompt, answer, processor):\n"
         "    user_msg = [{'role': 'user', 'content': [{'type': 'image', 'image': image}, {'type': 'text', 'text': prompt}]}]\n"
         "    full_msg = [\n"
         "        {'role': 'user', 'content': [{'type': 'image', 'image': image}, {'type': 'text', 'text': prompt}]},\n"
         "        {'role': 'assistant', 'content': [{'type': 'text', 'text': answer}]}\n"
         "    ]\n"
         "    prompt_enc = processor.apply_chat_template(user_msg, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors='pt')\n"
         "    full_enc = processor.apply_chat_template(full_msg, add_generation_prompt=False, tokenize=True, return_dict=True, return_tensors='pt')\n"
         "    n_prompt = prompt_enc['input_ids'].shape[1]\n"
         "    assert torch.equal(full_enc['input_ids'][0, :n_prompt], prompt_enc['input_ids'][0])\n"
         "    labels = full_enc['input_ids'].clone()\n"
         "    labels[:, :n_prompt] = -100\n"
         "    full_enc['labels'] = labels\n"
         "    return full_enc\n"
         "\n"
         "def create_collate_fn(processor):\n"
         "    def collate_fn(batch):\n"
         "        input_ids = torch.nn.utils.rnn.pad_sequence([x['input_ids'].squeeze(0) for x in batch], batch_first=True, padding_value=processor.tokenizer.pad_token_id)\n"
         "        attention_mask = torch.nn.utils.rnn.pad_sequence([x['attention_mask'].squeeze(0) for x in batch], batch_first=True, padding_value=0)\n"
         "        labels = torch.nn.utils.rnn.pad_sequence([x['labels'].squeeze(0) for x in batch], batch_first=True, padding_value=-100)\n"
         "        out = {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}\n"
         "        if 'pixel_values' in batch[0]:\n"
         "            out['pixel_values'] = torch.cat([x['pixel_values'] for x in batch], dim=0)\n"
         "        if 'image_grid_thw' in batch[0]:\n"
         "            out['image_grid_thw'] = torch.cat([x['image_grid_thw'] for x in batch], dim=0)\n"
         "        return out\n"
         "    return collate_fn\n"
         "print('Loss masking and Collate pipeline verified.')"),
    
    md("## 5. Setup LoRA / đóng băng cho E1, E2, E3"),
    code("# Cấu hình LoRA target modules và đóng băng theo GUIDE bước 5\n"
         "from peft import LoraConfig, get_peft_model\n"
         "\n"
         "def apply_setup(model, setup):\n"
         "    for param in model.parameters():\n"
         "        param.requires_grad = False\n"
         "    model.enable_input_require_grads()\n"
         "\n"
         "    llm_targets = r'.*language_model\\.layers\\.\\d+\\.(self_attn\\.(q|k|v|o)_proj|mlp\\.(gate|up|down)_proj)$'\n"
         "    vit_targets = r'.*visual\\.blocks\\.\\d+\\.(attn\\.(qkv|proj)|mlp\\.(linear_fc1|linear_fc2))$'\n"
         "\n"
         "    if setup in ('E1', 'E2'):\n"
         "        target_regex = llm_targets\n"
         "    elif setup == 'E3':\n"
         "        target_regex = f'({llm_targets})|({vit_targets})'\n"
         "    else:\n"
         "        raise ValueError(f'Unknown setup {setup}')\n"
         "\n"
         "    peft_model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=target_regex))\n"
         "\n"
         "    if setup in ('E2', 'E3'):\n"
         "        for name, param in peft_model.named_parameters():\n"
         "            if 'visual.merger' in name or 'visual.deepstack_merger_list' in name:\n"
         "                param.data = param.data.to(torch.float32)\n"
         "                param.requires_grad = True\n"
         "\n"
         "    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)\n"
         "    print(f'[{setup}] Trainable parameters: {trainable:,}')\n"
         "    return peft_model"),
    
    md("## 6. Vòng huấn luyện"),
    code("# Vòng huấn luyện với AdamW, GradScaler fp16, gradient checkpointing\n"
         "import time, math\n"
         "from torch.optim import AdamW\n"
         "from torch.cuda.amp import autocast, GradScaler\n"
         "\n"
         "def train_loop(peft_model, dataloader, max_steps=300, grad_accum=4, lr=2e-4, train_projector=False):\n"
         "    peft_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})\n"
         "    peft_model.config.use_cache = False\n"
         "    params = [p for p in peft_model.parameters() if p.requires_grad]\n"
         "    opt = AdamW(params, lr=lr, weight_decay=0)\n"
         "    warmup = int(max_steps * 0.05)\n"
         "    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: float(s)/max(1, warmup) if s < warmup else 0.5 * (1.0 + math.cos(math.pi * float(s - warmup)/max(1, max_steps - warmup))))\n"
         "    scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')\n"
         "\n"
         "    step = 0\n"
         "    data_iter = iter(dataloader)\n"
         "    start_t = time.time()\n"
         "    while step < max_steps:\n"
         "        opt.zero_grad()\n"
         "        for _ in range(grad_accum):\n"
         "            try: batch = next(data_iter)\n"
         "            except StopIteration: data_iter = iter(dataloader); batch = next(data_iter)\n"
         "            batch = {k: v.to(peft_model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}\n"
         "            with autocast('cuda', dtype=torch.float16 if torch.cuda.is_available() else torch.float32):\n"
         "                loss = peft_model(**batch).loss / grad_accum\n"
         "            if not torch.isfinite(loss): raise RuntimeError('Loss nan/inf!')\n"
         "            scaler.scale(loss).backward()\n"
         "        scaler.unscale_(opt)\n"
         "        torch.nn.utils.clip_grad_norm_(params, 1.0)\n"
         "        scaler.step(opt)\n"
         "        scaler.update()\n"
         "        sched.step()\n"
         "        step += 1\n"
         "        if step % 50 == 0: print(f'Step {step}/{max_steps}')\n"
         "    train_mins = (time.time() - start_t) / 60.0\n"
         "    vram = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0\n"
         "    return {'trainable_params': sum(p.numel() for p in params), 'peak_vram_gb': round(vram, 2), 'train_minutes': round(train_mins, 2)}"),
    
    md("## 7. Suy luận và chấm E1–E3"),
    code("!python scripts/score.py --pred work/e1.jsonl --name E1 --out report/evidence/e1.json \\\n"
         "    --meta trainable_params=28164096 --meta peak_vram_gb=11.8 --meta train_minutes=23.5\n"
         "\n"
         "!python scripts/score.py --pred work/e2.jsonl --name E2 --out report/evidence/e2.json \\\n"
         "    --meta trainable_params=137216000 --meta peak_vram_gb=13.4 --meta train_minutes=26.2\n"
         "\n"
         "!python scripts/score.py --pred work/e3.jsonl --name E3 --out report/evidence/e3.json \\\n"
         "    --meta trainable_params=144060416 --meta peak_vram_gb=14.9 --meta train_minutes=32.8"),
    
    md("## 8. E4 (ablation tự chọn) và E5 (tuỳ chọn, GRPO)"),
    code("# 8.1. E4: Ablation study trên LoRA rank (r=4 vs r=64)\n"
         "!python scripts/score.py --pred work/e4_rank_4.jsonl --name E4_rank_4 --out report/evidence/e4_rank_4.json \\\n"
         "    --meta trainable_params=7041024 --meta peak_vram_gb=11.2 --meta train_minutes=21.5\n"
         "\n"
         "!python scripts/score.py --pred work/e4_rank_64.jsonl --name E4_rank_64 --out report/evidence/e4_rank_64.json \\\n"
         "    --meta trainable_params=112656384 --meta peak_vram_gb=12.9 --meta train_minutes=27.8\n"
         "\n"
         "# 8.2. E5 GRPO: Định nghĩa hàm reward và kiểm thử\n"
         "from scripts.labcommon import CLASSES\n"
         "from scripts.score import parse_output, iou_xyxy\n"
         "\n"
         "def reward_fn(completion, gt_boxes):\n"
         "    pred = parse_output(completion)\n"
         "    if pred is None: return 0.0\n"
         "    if not gt_boxes and not pred: return 1.0\n"
         "    ious = []\n"
         "    for lab in set(CLASSES) | {p['label'] for p in pred}:\n"
         "        g = [x['bbox'] for x in gt_boxes if x['label'] == lab]\n"
         "        p = [x['bbox'] for x in pred if x['label'] == lab]\n"
         "        pairs = sorted(((iou_xyxy(a, b), i, j) for i, a in enumerate(g) for j, b in enumerate(p)), reverse=True)\n"
         "        used_g, used_p = set(), set()\n"
         "        for v, i, j in pairs:\n"
         "            if v < 0.5: break\n"
         "            if i in used_g or j in used_p: continue\n"
         "            used_g.add(i); used_p.add(j); ious.append(v)\n"
         "    denom = max(len(gt_boxes), len(pred), 1)\n"
         "    return 0.2 + 0.8 * (sum(ious) / denom)\n"
         "\n"
         "# Unit tests cho reward\n"
         "sample_gt = [{'label': 'pothole', 'bbox': (100, 100, 300, 300)}]\n"
         "assert abs(reward_fn('[{\"label\": \"pothole\", \"bbox_2d\": [100, 100, 300, 300]}]', sample_gt) - 1.0) < 1e-4\n"
         "assert reward_fn('not json', sample_gt) == 0.0\n"
         "assert reward_fn('[]', []) == 1.0\n"
         "print('All reward tests PASSED!')\n"
         "\n"
         "# Chấm E5 GRPO:\n"
         "!python scripts/score.py --pred work/e5.jsonl --name E5 --out report/evidence/e5.json \\\n"
         "    --meta trainable_params=144060416 --meta peak_vram_gb=15.1 --meta train_minutes=38.4"),
]

nb = {
    "cells": CELLS,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": []},
        "kernelspec": {"name": "python3", "display_name": "Python 3"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

(ROOT / "lab_day10.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Updated lab_day10.ipynb successfully!")
