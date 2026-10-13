"""Student implementation for the Day 10 Colab notebook (E0--E4).

The dataset, model and GPU are intentionally loaded only when called by the notebook.
Evidence is produced exclusively by the supplied scripts/score.py.
"""

import gc
import json
import math
import random
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
WORK = ROOT / "work"
MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
CLASSES = ("pothole", "crack", "manhole")


def read_split(split):
    with (DATA / "splits" / f"{split}.jsonl").open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def inspect_splits():
    records = {split: read_split(split) for split in ("train", "val", "test")}
    sessions = {split: {row["session"] for row in rows} for split, rows in records.items()}
    assert not any(sessions[a] & sessions[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test")))
    for split, rows in records.items():
        counts = Counter(obj["label"] for row in rows for obj in row["objects"])
        print(split, "images", len(rows), "boxes", dict(counts), "sessions", len(sessions[split]))
    return records


def yolo_to_box(label, yolo, width=640, height=360, coord="norm1000"):
    xc, yc, w, h = map(float, yolo)
    sx, sy = (1000, 1000) if coord == "norm1000" else (width, height)
    bb = [round((xc - w / 2) * sx), round((yc - h / 2) * sy),
          round((xc + w / 2) * sx), round((yc + h / 2) * sy)]
    bb = [max(0, min((sx if i % 2 == 0 else sy), v)) for i, v in enumerate(bb)]
    return {"label": label, "bbox_2d": bb}


def objects_to_answer(row, coord="norm1000"):
    boxes = [yolo_to_box(o["label"], o["yolo"], row["width"], row["height"], coord)
             for o in row["objects"]]
    boxes.sort(key=lambda o: (o["bbox_2d"][1], o["bbox_2d"][0]))
    return json.dumps(boxes, ensure_ascii=False, separators=(",", ":"))


def prompt_for(coord="norm1000"):
    scale = "integer coordinates normalized to 0-1000 for both axes" if coord == "norm1000" else "integer pixel coordinates in the original image"
    return ("Find every road damage object in this image. Classes: pothole, crack, manhole. "
            f"Use {scale}. Return ONLY a JSON array of objects with exactly the keys "
            "label and bbox_2d; bbox_2d is [x1,y1,x2,y2]. Return [] if there are none.")


def image_for(row):
    return Image.open(DATA / row["image_path"]).convert("RGB")


def draw_examples(rows, count=6):
    from IPython.display import display
    for row in rows[:count]:
        im = image_for(row)
        d = ImageDraw.Draw(im)
        for o in row["objects"]:
            bb = yolo_to_box(o["label"], o["yolo"], row["width"], row["height"], "pixel")["bbox_2d"]
            d.rectangle(bb, outline="lime", width=2)
            d.text((bb[0], bb[1]), o["label"], fill="lime")
        print(row["image_id"], objects_to_answer(row))
        display(im)


def load_model():
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
    if not torch.cuda.is_available():
        raise RuntimeError("This lab requires a CUDA GPU (Colab T4).")
    print(f"Loading model {MODEL_ID}...", flush=True)
    started = time.monotonic()
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.float16,
                              bnb_4bit_use_double_quant=True,
                              llm_int8_skip_modules=["visual", "lm_head"])
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, quantization_config=quant, dtype=torch.float16, device_map={"": 0})
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer.padding_side = "left"
    print(f"Model ready in {(time.monotonic() - started):.1f}s", flush=True)
    return model, processor


def messages(row, answer=None, coord="norm1000"):
    user = {"role": "user", "content": [{"type": "image", "image": image_for(row)},
                                        {"type": "text", "text": prompt_for(coord)}]}
    return [user] if answer is None else [user, {"role": "assistant", "content": [{"type": "text", "text": answer}]}]


def predict(model, processor, rows, output, coord="norm1000", batch_size=4):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    processor.tokenizer.padding_side = "left"
    model.eval()
    started = time.monotonic()
    total = len(rows)
    with output.open("w", encoding="utf-8") as f, torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            batch_end = min(start + len(batch), total)
            print(f"{output.name}: processing images {start + 1}-{batch_end}/{total}", flush=True)
            prompts = [messages(row, coord=coord) for row in batch]
            inputs = processor.apply_chat_template(prompts, tokenize=True,
                        add_generation_prompt=True, return_dict=True, return_tensors="pt", padding=True)
            inputs = inputs.to("cuda")
            outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            raw = processor.batch_decode(outputs[:, inputs["input_ids"].shape[1]:],
                                         skip_special_tokens=True)
            for row, answer in zip(batch, raw):
                f.write(json.dumps({"image_id": row["image_id"], "raw": answer}, ensure_ascii=False) + "\n")
            f.flush()
            elapsed = time.monotonic() - started
            eta = elapsed * (total - batch_end) / batch_end if batch_end else 0
            print(f"{output.name}: {batch_end}/{total} done; elapsed {elapsed / 60:.1f} min; ETA {eta / 60:.1f} min", flush=True)
    return output


def encode_example(processor, row, coord="norm1000"):
    answer = objects_to_answer(row, coord)
    msg = messages(row, coord=coord)
    common = dict(tokenize=True, return_dict=True, return_tensors="pt", padding=False)
    prompt = processor.apply_chat_template(msg, add_generation_prompt=True, **common)
    full = processor.apply_chat_template(messages(row, answer=answer, coord=coord),
                                         add_generation_prompt=False, **common)
    n = prompt["input_ids"].shape[1]
    if not torch.equal(prompt["input_ids"][0], full["input_ids"][0, :n]):
        raise ValueError("Chat template prompt is not a prefix of the labeled conversation")
    labels = full["input_ids"].clone()
    labels[:, :n] = -100
    if not (labels != -100).any():
        raise ValueError("No assistant tokens remain after masking")
    result = {k: v.squeeze(0) if k in ("input_ids", "attention_mask", "mm_token_type_ids") else v
              for k, v in full.items()}
    result["labels"] = labels.squeeze(0)
    return result


def collate(batch, pad_id):
    max_len = max(len(b["input_ids"]) for b in batch)
    out = {}
    for key, fill in (("input_ids", pad_id), ("attention_mask", 0),
                      ("mm_token_type_ids", 0), ("labels", -100)):
        if key not in batch[0]:
            continue
        out[key] = torch.stack([torch.nn.functional.pad(b[key], (0, max_len - len(b[key])), value=fill)
                                for b in batch])
    for key in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
        if key in batch[0]:
            out[key] = torch.cat([b[key] for b in batch], dim=0)
    return out


def _target_names(model, include_vit):
    llm = re.compile(r"(?:^|\.)language_model\.layers\.\d+\."
                     r"(?:self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)$")
    vit = re.compile(r"(?:^|\.)visual\.blocks\.\d+\."
                     r"(?:attn\.(?:qkv|proj)|mlp\.(?:linear_fc1|linear_fc2))$")
    names = [name for name, _ in model.named_modules() if llm.search(name) or (include_vit and vit.search(name))]
    expected = 348 if include_vit else 252
    if len(names) != expected:
        raise RuntimeError(f"Found {len(names)} LoRA targets, expected {expected}. Inspect model.named_modules().")
    return names


def _projector_params(model):
    return {name: p for name, p in model.named_parameters()
            if "visual.merger." in name or "visual.deepstack_merger_list." in name}


def apply_setup(model, setup, rank=16):
    from peft import LoraConfig, get_peft_model
    for p in model.parameters():
        p.requires_grad_(False)
    model.enable_input_require_grads()
    names = _target_names(model, include_vit=(setup == "E3"))
    model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=2 * rank,
                           lora_dropout=0.05, target_modules=names, bias="none"))
    if setup in ("E2", "E3"):
        projectors = _projector_params(model)
        if not projectors:
            raise RuntimeError("Projector parameters were not found")
        for p in projectors.values():
            p.data = p.data.to(torch.float32)
            p.requires_grad_(True)
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(setup, "LoRA modules", len(names), "trainable parameters", count)
    return model, count


def save_checkpoint(model, folder, run_meta=None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(folder / "adapter")
    projectors = _projector_params(model)
    if any(p.requires_grad for p in projectors.values()):
        torch.save({name: p.detach().cpu() for name, p in projectors.items()}, folder / "projector.pt")
    if run_meta is not None:
        (folder / "run.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")


def restore_checkpoint(model, folder):
    from peft import PeftModel
    folder = Path(folder)
    model = PeftModel.from_pretrained(model, folder / "adapter")
    path = folder / "projector.pt"
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        current = _projector_params(model)
        if saved.keys() != current.keys():
            raise RuntimeError("Saved projector keys differ from model projector keys")
        with torch.no_grad():
            for name, p in current.items():
                p.copy_(saved[name].to(device=p.device, dtype=p.dtype))
    return model


def train(model, processor, rows, setup, folder, rank=16, seed=42,
          max_steps=300, batch_size=2, grad_accum=4, lr=2e-4):
    from torch.amp import GradScaler, autocast
    from transformers import get_cosine_schedule_with_warmup
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    processor.tokenizer.padding_side = "right"
    model, trainable = apply_setup(model, setup, rank)
    # GradScaler cannot unscale gradients belonging to FP16 trainable weights.
    # Keep the frozen base model in its original dtype, but optimize trainable
    # LoRA and projector weights in FP32.
    for p in model.parameters():
        if p.requires_grad and p.dtype == torch.float16:
            p.data = p.data.float()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0)
    scheduler = get_cosine_schedule_with_warmup(optimizer, max(1, int(max_steps * 0.05)), max_steps)
    scaler = GradScaler("cuda")
    rng = random.Random(seed)
    order = list(range(len(rows)))
    rng.shuffle(order)
    cursor = 0
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for step in range(1, max_steps + 1):
        losses = []
        for _ in range(grad_accum):
            if cursor + batch_size > len(order):
                rng.shuffle(order)
                cursor = 0
            chosen = [rows[i] for i in order[cursor:cursor + batch_size]]
            cursor += batch_size
            examples = [encode_example(processor, row) for row in chosen]
            batch = {k: v.to("cuda") for k, v in collate(examples, processor.tokenizer.pad_token_id).items()}
            with autocast("cuda", dtype=torch.float16):
                loss = model(**batch).loss / grad_accum
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at step {step}")
            scaler.scale(loss).backward()
            losses.append(float(loss.detach()) * grad_accum)
        # Some PEFT/projector weights can still produce FP16 gradients during
        # backward even after their weights were promoted before the forward.
        # GradScaler requires FP32 gradients for optimizer-owned parameters.
        for p in params:
            if p.grad is not None and p.grad.dtype == torch.float16:
                p.data = p.data.float()
                p.grad = p.grad.float()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if step == 1 or step % 10 == 0:
            print(f"{setup} step {step}/{max_steps} loss={sum(losses)/len(losses):.4f}", flush=True)
        if step % 100 == 0:
            save_checkpoint(model, folder)
    run_meta = {"trainable_params": trainable,
                "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
                "train_minutes": (time.monotonic() - start) / 60,
                "steps": max_steps, "rank": rank, "setup": setup, "seed": seed,
                "batch_size": batch_size, "grad_accum": grad_accum}
    save_checkpoint(model, folder, run_meta)
    return model, run_meta


def score(prediction, name, output, run_meta=None, split="test", coord="norm1000"):
    cmd = [sys.executable, str(ROOT / "scripts" / "score.py"), "--pred", str(prediction),
           "--name", name, "--out", str(output), "--split", split, "--coord", coord]
    if run_meta:
        for key in ("trainable_params", "peak_vram_gb", "train_minutes"):
            cmd += ["--meta", f"{key}={run_meta[key]}"]
    subprocess.run(cmd, check=True, cwd=ROOT)
    return json.loads(Path(output).read_text(encoding="utf-8"))


def release_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()


def run_e0(test_rows):
    model, processor = load_model()
    batch_size = 1 if torch.cuda.get_device_properties(0).total_memory < 8 * 1024**3 else 4
    prediction = predict(model, processor, test_rows, WORK / "e0.jsonl", batch_size=batch_size)
    result = score(prediction, "E0", ROOT / "report/evidence/e0.json")
    release_model(model)
    return result


def run_trained(name, train_rows, test_rows, setup=None, rank=16):
    setup = setup or name
    model, processor = load_model()
    folder = WORK / name.lower()
    small_gpu = torch.cuda.get_device_properties(0).total_memory < 8 * 1024**3
    model, run_meta = train(model, processor, train_rows, setup, folder, rank=rank,
                            batch_size=1 if small_gpu else 2, grad_accum=8 if small_gpu else 4)
    # Reload from the saved adapter and projector to verify the actual inference artifact.
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    model, processor = load_model()
    model = restore_checkpoint(model, folder)
    prediction = predict(model, processor, test_rows, WORK / f"{name.lower()}.jsonl",
                         batch_size=1 if small_gpu else 4)
    result = score(prediction, name, ROOT / "report/evidence" / f"{name.lower()}.json", run_meta)
    release_model(model)
    return result


def save_illustrations(rows, prediction_file, prefix, count=6):
    """Save examples with green ground truth and red predictions for the report."""
    from scripts.score import iou_xyxy, parse_output
    raw = {}
    with Path(prediction_file).open(encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            raw[item["image_id"]] = item["raw"]
    good, bad = [], []
    for row in rows:
        parsed = parse_output(raw[row["image_id"]])
        gt = [yolo_to_box(o["label"], o["yolo"], row["width"], row["height"], "pixel")
              for o in row["objects"]]
        pred = [] if parsed is None else [
            {"label": o["label"], "bbox_2d": [round(o["bbox"][0] * row["width"] / 1000),
             round(o["bbox"][1] * row["height"] / 1000),
             round(o["bbox"][2] * row["width"] / 1000),
             round(o["bbox"][3] * row["height"] / 1000)]} for o in parsed]
        matches = sum(any(p["label"] == g["label"] and
                          iou_xyxy(p["bbox_2d"], g["bbox_2d"]) >= 0.5 for p in pred)
                      for g in gt)
        (good if matches == len(gt) and len(pred) == len(gt) and parsed is not None else bad).append((row, gt, pred))
    chosen = (good[:count // 2] + bad[:count - min(count // 2, len(good))])[:count]
    if len(chosen) < count:
        chosen += (good[count // 2:] + bad[count:])[:count - len(chosen)]
    outdir = ROOT / "report" / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, (row, gt, pred) in enumerate(chosen, 1):
        im = image_for(row)
        draw = ImageDraw.Draw(im)
        for o in gt:
            bb = o["bbox_2d"]
            draw.rectangle(bb, outline="lime", width=2)
            draw.text((bb[0], bb[1]), "GT " + o["label"], fill="lime")
        for o in pred:
            bb = o["bbox_2d"]
            draw.rectangle(bb, outline="red", width=2)
            draw.text((bb[0], max(0, bb[1] - 12)), "Pred " + o["label"], fill="red")
        path = outdir / f"{prefix}_{i:02d}.png"
        im.save(path)
        paths.append(path)
        print(path, row["image_id"], "GT", len(gt), "Pred", len(pred))
    return paths
