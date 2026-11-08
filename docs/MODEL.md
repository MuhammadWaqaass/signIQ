# Models

The service uses two models. Both are loaded once by the worker at startup.

| | Detector | Similarity |
|---|---|---|
| Task | Find signatures on a document | Score how alike two signatures are |
| Architecture | LibreYOLO9m (YOLOv9-medium) | Siamese ConvNeXt-Tiny |
| Library | `libreyolo` | PyTorch, torchvision |
| Weights | `best.pt`, about 77 MB | `model/best.pt`, 112 MB |
| Hub location | **TODO: upload and fill in** | `izamscientist/signature-similarity-convnext-tiny` |
| Input | Document image | Two signature images |
| Output | Boxes with confidence | Score from 0 to 1 |

---

## 1. Signature detector

### Purpose
Locates every signature on a document page and returns bounding boxes.

### Training
| Item | Value |
|---|---|
| Model | LibreYOLO9m, starting from `LibreYOLO9m.pt` |
| Library | `libreyolo` (permissive license; chosen because Ultralytics is AGPL) |
| Dataset | Merged signature dataset in YOLO format (train / valid / test splits) |
| Image size | 640 |
| Epochs | Up to 40 |
| Batch size | 16 |
| Early stopping | Patience 10 |
| Validation | After every epoch |
| Hardware | Kaggle GPU |
| Final test set | 2,718 images with 3,472 signature annotations |

The notebook also contains earlier experiments with Ultralytics (YOLO26 variants) and RF-DETR. Those were exploration only. The deployed detector is the LibreYOLO model, and no Ultralytics code runs in production.

### Inference settings
| Setting | Value | Why |
|---|---|---|
| Confidence | 0.385 | Peak F1 point found during evaluation |
| IoU (NMS) | 0.5 | Merges overlapping boxes |
| Image size | 640 | Matches training |

Both are configurable through `DETECT_CONF` and `DETECT_IOU` in `.env`.

### Usage
```python
from libreyolo import LibreYOLO

model = LibreYOLO("weights/best.pt")
results = model.predict(source="document.jpg", conf=0.385, iou=0.5, save=False)
for box in results[0].boxes:
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    confidence = float(box.conf[0])
```

### Performance on the test set
The notebook evaluates the model on the test split but the final numbers are not saved in it. Fill these in from your evaluation run.

| Metric | Value |
|---|---|
| mAP50 | **TODO** |
| mAP50-95 | **TODO** |
| Precision | **TODO** |
| Recall | **TODO** |
| F1 at conf 0.385 | **TODO** |

### Known limitations
- The dataset has many near-duplicate images (the data check flagged tens of thousands of near-identical pairs in train, plus pairs in validation and test). This can make test scores look better than real-world performance. Note this honestly in the presentation.
- Performance on heavily rotated, low-resolution or photographed documents has not been measured.

---

## 2. Signature similarity model

Source: the model card at `izamscientist/signature-similarity-convnext-tiny`.

### Purpose
Given two signatures, estimates how likely it is that the same person wrote both. It works on writers it never saw in training (writer-independent).

### Architecture
- Backbone: ConvNeXt-Tiny pretrained on ImageNet, 1,000-class layer removed.
- Head: dropout 0.2, then a linear layer producing a 256-number embedding.
- Siamese: both images pass through the same network. Embeddings are normalised and the score is the cosine similarity rescaled to 0 to 1.
- Loss: contrastive loss with margin 0.3.

### Training data
HMS (Bangla, English, "S"), CEDAR, and BHSig260 (Bengali, Hindi). Split by writer: 452 training writers (14,556 images), 96 validation writers (3,118), 99 test writers (3,199). Pairs include genuine-vs-genuine (same person), genuine-vs-skilled-forgery, and different-writer.

### Preprocessing
Handled inside the repo's `inference.py`: grayscale, ink isolation, box-line removal, tight crop, square padding, resize to 224 x 224.

### Thresholds (chosen on validation data only)
| Name | Value | Meaning |
|---|---|---|
| main | 0.7785 | Single yes/no cut: at or above means same person |
| accept | 0.9106 | At or above is ACCEPT (false accepts about 1% or less on validation) |
| reject | 0.5376 | Below is REJECT (false rejects about 1% or less on validation) |

Scores between reject and accept are REVIEW and need a person.

### Performance (test set, 3,000 pairs, 99 unseen writers)
| Metric | Value |
|---|---|
| Accuracy | 81.5% (95% range 78.8% to 84.2%) |
| F1 | 0.821 |
| ROC-AUC | 0.902 |
| EER | 18.5% |
| False accept rate | 21.8% |
| False reject rate | 15.2% |

Weaker groups: HMS English (76.4% accuracy) and HMS Bengali (79.5%).

### Usage
```python
from huggingface_hub import snapshot_download
import sys

root = snapshot_download("izamscientist/signature-similarity-convnext-tiny")
sys.path.insert(0, root)
import inference

model = inference.load_model()
score = inference.compare(model, "signature_a.png", "signature_b.png")
verdict = inference.decide(score)   # ACCEPT / REJECT / REVIEW
```

### Limitations
- Not a sole decision-maker. At the main threshold, about 1 in 5 different-person pairs (including forgeries) are accepted.
- Trained on scans; phone photos or unusual paper may score lower.
- The test set is public, so a new model needs new writers for an honest comparison.
- Each data source keeps its own license terms, and the model card lists the model license as "other". State this in the presentation.

---

## 3. How the models are used together
1. The detector finds signature boxes on the document.
2. Each box is cropped and saved.
3. The similarity model compares each crop to the reference signature.
4. The API returns box, confidence, similarity score and verdict for each one.

## 4. Versioning and updates
- Both models are stored on the Hugging Face Hub. A new version is a new upload.
- The training pipeline (planned) should: train on new data, log the run to Comet or MLflow, compare metrics against the current model, and upload new weights only if they are better.
- The worker reads the repo names from `.env` (`DETECTOR_REPO`, `SIMILARITY_REPO`). Restarting the worker loads the latest version.
- Record the Hub revision (commit hash) of each model used in a release, so results can be reproduced.

## 5. Config reference
| Variable | Purpose |
|---|---|
| `DETECTOR_LOCAL_PATH` | Local detector weights, used if set |
| `DETECTOR_REPO`, `DETECTOR_FILENAME` | Hub location of the detector |
| `SIMILARITY_REPO` | Hub location of the similarity model |
| `DETECT_CONF`, `DETECT_IOU` | Detector thresholds |
| `HF_TOKEN` | Needed only for private repos |
