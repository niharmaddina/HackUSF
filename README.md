# Skin-lesion classifier (HackUSF 2025)

Built in 24 hours at HackUSF, April 2025. A dermatoscopic image goes in; three fine-tuned CNNs
(ResNet50, DenseNet121, MobileNetV3-Large) each classify it into one of seven lesion types, and a
majority vote picks the answer. A Next.js app wraps the classifier with an upload flow, a history
page and plain-language guidance per class.

This is hackathon code, kept as it was at the end of the event, plus this README.

## Classes

| Code | Lesion | Risk label shown to the user |
|---|---|---|
| akiec | actinic keratosis / intraepithelial carcinoma | high |
| bcc | basal cell carcinoma | high |
| bkl | benign keratosis-like lesion | low |
| df | dermatofibroma | low |
| mel | melanoma | very high |
| nv | melanocytic nevus | low |
| vasc | vascular lesion | low |

## Results

Source images: 10,015 dermatoscopic images across the seven classes (the class counts in
`skin-lesion-mvp/ml-service/models/README.md` match the HAM10000 collection). Classes were balanced
to 6,705 images each by augmentation, then split 70/15/15 into 32,853 / 7,041 / 7,041 images. Each
model was fine-tuned from ImageNet weights for 5 epochs at batch size 128, 224x224 input.

| Model | Test accuracy | Macro F1 | Training time |
|---|---|---|---|
| MobileNetV3-Large | 84.79 % | 0.84 | 13 min |
| DenseNet121 | 94.62 % | 0.95 | 40 min |
| ResNet50 | 95.20 % | 0.95 | 36 min |
| Majority-vote ensemble | 95.46 % | 0.954 | n/a |

Melanoma and benign keratosis are the hardest classes for every model (ensemble recall 0.87 and
0.90); dermatofibroma and vascular lesions are near-perfect. Full per-class tables are in
`models/README.md`.

Read these numbers with one caveat: augmentation was applied before the split, so augmented
copies of the same source image can sit in both train and test. The test accuracy is therefore
optimistic. A clean evaluation would split on source image first and augment only the training
set.

## Running the classifier

The three weight files are committed in `skin-lesion-mvp/ml-service/models/` (94, 28 and 13 MB).

```bash
cd skin-lesion-mvp/ml-service
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py -f cancer_images/melanoma.jpeg
```

The CLI prints each model's class and confidence, the ensemble vote and the share of models that
agreed, plus averaged class probabilities. Three sample images are in `cancer_images/`. A
`Dockerfile` in the same directory builds the service image.

Checked on 2026-10-11 with torch 2.14 / torchvision 0.29: all three models load and the three
samples give dermatofibroma (correct, 2 of 3 votes), nevus for the melanoma sample (2 of 3) and
nevus for the basal-cell sample (3 of 3). The samples are ordinary photos, not dermatoscopic test
images, so two misses out of three is a fair picture of how far the test accuracy transfers.
The only code change since the event is the MobileNetV3 head in `app.py`, rebuilt to match the
saved checkpoint (960 to 256 to 7) so that model loads instead of being skipped.

The web app in `skin-lesion-mvp/next-app` is a Next.js 15 project (`pnpm install && pnpm dev`);
it expects the classifier to be reachable as an HTTP service, which was wired up during the event
and is not reproduced here.

## Layout

```
skin-lesion-mvp/
  ml-service/
    app.py             CLI: load the three models, preprocess, predict, majority vote
    models/            fine-tuned weights and the training notes (models/README.md)
    cancer_images/     three sample inputs
    requirements.txt   torch 2.2, torchvision 0.17, FastAPI, Pillow
    Dockerfile
  next-app/            Next.js UI (upload, analyze, history, education pages)
```

## Limitations

- Not a medical device and not validated clinically. The risk labels and recommendations in the
  UI are illustrative.
- Test accuracy is inflated by the augment-then-split issue described above.
- Weights live in git history, which makes the repository 136 MB to clone.
- Five epochs on a balanced set; no hyperparameter search, no calibration, no explainability.

## License

MIT, see `LICENSE`.
