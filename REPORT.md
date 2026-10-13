Ho ten: Nguyễn Xuân Thành.
`dataset_version`: `d10-21e4070301` (prepared locally from the Kaggle dataset).

**Status:** E0 and E1 completed on Colab, and their scorer-generated evidence was recovered from Drive. E2-E4 remain pending because the GPU runtime has been unreliable. The local RTX 3050 Laptop 4 GB attempt did not complete E0. Experimental values below come from the recovered evidence; no pending results are inferred.

Local environment: Python 3.10.9, PyTorch 2.5.1+cu121, Transformers 4.57.2, RTX 3050 Laptop GPU 4 GB, 15.4 GB system RAM. The Colab-pinned Transformers 5.19.0 failed to import with this PyTorch build; the local environment uses 4.57.2 as documented in `requirements-local.txt`. The dataset split has 1422/276/311 train/val/test images with no session overlap.
## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | zero-shot | 0 | Not recorded | Not recorded | 0.0795 | 0.0552–0.1080 | 0.842 | 0.1615 | 0.0246 | 0.1379 |
| E1 | LoRA LLM | 33,030,144 | 5.645 | 44.999 | 0.2806 | 0.2455–0.3207 | 1.219 | 0.1991 | 0.2143 | 0.5197 |
| E2 | + projector | Pending run | Pending run | Pending run | Pending run | Pending run | Pending run | Pending run | Pending run | Pending run |
| E3 | + LoRA ViT | Pending run | Pending run | Pending run | Pending run | Pending run | Pending run | Pending run | Pending run | Pending run |
| E4 rank 4 | Best E1-E3 setup, r=4 | Pending run | Pending run | Pending run | Pending run | Pending run | Pending run | Pending run | Pending run | Pending run |
| E4 rank 64 | Same setup, r=64 | Pending run | Pending run | Pending run | Pending run | Pending run | Pending run | Pending run | Pending run | Pending run |

## 10 câu hỏi (trả lời bằng **số của bạn**, không chép lại tài liệu)

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?
Per lab specifications, LLM plus embeddings are about 4.0B parameters, ViT about 415M, merger about 27M, and DeepStack mergers about 82M; these are reference figures pending model inspection. E1 trains LLM LoRA, E2 also trains the full projector, and E3 also applies ViT LoRA. The code asserts 252/348 LoRA target modules.

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?
E0 parse_valid_rate was 0.9132 (284/311 valid outputs). Rechecking all 311 saved outputs with the supplied scorer found two frequent failure patterns: 252 images had no true positive at IoU 0.5, and 27 outputs could not be parsed as valid detection arrays. The 252 images are a broad outcome category that includes wrong class, missed objects, and misplaced boxes; it is not a single root cause. A raw malformed example from `vlcsnap-00206` is `[{"label":"manhole","bbox_2d":[309,675,396,714]}\n}`: the closing `]` is missing. E0 also produced 182 false-positive crack boxes and missed 293 of 299 crack boxes.

### Câu 3 — Thang 0–1000: nếu chấm output theo pixel (hoặc ngược lại) thì F1 ra sao và vì sao? Bạn đã kiểm chứng điều này thế nào?
YOLO center/size coordinates are converted to corners and scaled to 0-1000 on each axis. I rescored the same 311 saved E1 outputs with the supplied scorer, changing only `--coord norm1000` to `--coord pixel`. F1 fell from 0.2806 (CI95 0.2455–0.3207) to 0.0000 (CI95 0–0). The scorer divided coordinates already normalized to 0–1000 by each image's pixel dimensions, so the predicted boxes moved and shrank relative to ground truth. This is a scoring diagnostic, not a separate model run; diagnostic output is in `work/e1_wrongcoord.json`.

### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời? Điều gì xảy ra nếu tính trên cả prompt và token ảnh?
Only assistant answer tokens and the end token contribute to loss. Prompt and image tokens are input conditions, not targets. Training on them would spend gradient capacity reconstructing fixed inputs. encode_example checks the prompt prefix and masks it with -100.

### Câu 5 — Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit, và bạn đã xử lý thế nào? Vì sao projector phải ở float32?
The LLM is quantized to 4-bit while visual and lm_head are skipped. Full projector training requires floating-point parameters. E2/E3 cast the projector to float32 for more stable updates and save it separately, because PEFT adapters only include LoRA weights.

### Câu 6 — Split theo phiên quay: vì sao chia ngẫu nhiên sẽ cho F1 cao giả? Số ảnh và số box từng lớp ở mỗi split là bao nhiêu?
Adjacent video frames look similar; random image splits can leak near-duplicates and inflate test F1. inspect_splits asserts no session overlap. The prepared data contains train 1422 images (1074 pothole, 1750 crack, 691 manhole), val 276 (66, 470, 103), test 311 (121, 299, 163).

### Câu 7 — So sánh E1/E2/E3: mở projector và LoRA trên ViT có đáng không (F1, CI95, VRAM, thời gian)?
E1 reached F1 0.2806 (CI95 0.2455–0.3207) with 5.645 GB peak VRAM and 44.999 training minutes. E2 and E3 remain unmeasured, so the extra cost and benefit of projector and ViT updates cannot yet be assessed.

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?
The scorer bootstraps by image. For 311 test images, E0 CI95 is 0.0552–0.1080 (width 0.0528) and E1 CI95 is 0.2455–0.3207 (width 0.0752). These marginal intervals are separated, supporting an E1 improvement over E0 on this test split. They are not a paired confidence interval for the difference; E2/E3 comparisons need their own evidence. The test split has only 121 pothole boxes, so conclusions about that class are less stable.

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.
In E1, pothole was weakest by F1: 0.1991 versus crack 0.2143 and manhole 0.5197. Potholes had the fewest test boxes (121), the smallest median test box area (0.0042 of the image, versus 0.0198 for crack and 0.0067 for manhole), and 99/121 were missed. These observations are consistent with small-object difficulty, but do not establish its cause. Crack had more false positives (276) despite slightly higher F1. The six E1 overlays below show green ground truth and red predictions. Images 1–3 are exact detections; images 4–6 show errors. They are selected examples, not a random sample.

![E1 example 1](report/figures/e1_01.png) ![E1 example 2](report/figures/e1_02.png) ![E1 example 3](report/figures/e1_03.png)

![E1 example 4](report/figures/e1_04.png) ![E1 example 5](report/figures/e1_05.png) ![E1 example 6](report/figures/e1_06.png)

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?
E4 varies only LoRA rank: 4 versus 64, alpha equals twice rank. It keeps the selected base setup, data, prompt, seed and steps fixed. Results and CI95 are pending the actual runs.

## Nhìn lại

Colab produced valid E0 and E1 evidence. All six annotated images were generated offline from saved E1 predictions. E2–E4 have no results, so the full experimental comparison and final checker cannot pass yet. No metrics have been filled in for unrun stages.
