# GliomaScan-HC

GliomaScan-HC is a research-oriented glioma classification pipeline built around the paper:
Pilaoon, Hamamoto, Maneerat, "Glioma Brain Tumor Classification Using Convolution Neural Network and Majority Voting", IEEE Access, 2025.

This repository implements the paper's essential workflow with a practical software extension: a hybrid camera ingest path for MRI screenshots, photos, DICOM, and NIfTI slices. The hybrid camera is an extension beyond the paper and is explicitly not a clinical device.

## Assumptions
- The classifier has four output classes: glioma, meningioma, notumor, and pituitary.
- The Kaggle MRI dataset does not specify MRI sequence; these labels are not clinical diagnoses.
- Dataset images must be obtained by the user; no dataset is auto-downloaded.
- The project is built for Python 3.10/3.11 and TensorFlow 2.15.x.
- When real weights are missing, the API returns HTTP 503 and explains the training command.
- Batch sizes and training hyperparameters follow the paper reproduction defaults unless a fast preset is chosen.

## Project purpose
- Adapt the downloaded Kaggle CNN workflow for four-class brain MRI image classification.
- Add a hybrid ingest workflow for camera, upload, DICOM, and NIfTI inputs.
- Provide a local FastAPI backend and a lightweight HTML/JS frontend.
- Keep the code open, typed, and testable.

## Directory layout

```text
glioma-hybrid-cam/
├── README.md
├── requirements.txt
├── requirements-dev.txt
├── .gitignore
├── .env.example
├── glioma_hc/
│   ├── __init__.py
│   ├── config.py
│   ├── preprocessing/
│   │   └── pipeline.py
│   ├── hybrid_camera/
│   │   ├── __init__.py
│   │   ├── cli.py
│   │   ├── quality.py
│   │   ├── ingest.py
│   │   ├── study.py
│   │   └── camera_sim.py
│   ├── dataset_builder/
│   │   ├── __init__.py
│   │   ├── build_dataset.py
│   │   ├── extract_slices.py
│   │   └── make_smoke_data.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── model_zoo.py
│   │   ├── train_all.py
│   │   └── ensemble.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluate.py
│   │   ├── benchmark.py
│   │   ├── robustness.py
│   │   ├── paper_comparison.py
│   │   └── stats_tests.py
│   ├── explainability/
│   │   ├── gradcam.py
│   │   ├── lime_explainer.py
│   │   └── hybrid_cam.py
│   └── serving/
│       ├── __init__.py
│       ├── inference.py
│       └── schemas.py
├── backend/
│   └── main.py
├── frontend/
│   ├── index.html
│   ├── style.css
│   ├── app.js
│   └── assets/
├── data/
│   ├── README.md
│   ├── raw/
│   └── processed/
├── models/
├── reports/
├── scripts/
│   └── run_all_quick.py
├── notebooks/
│   └── train_on_colab.ipynb
├── tests/
│   ├── test_preprocessing.py
│   ├── test_camera_quality.py
│   ├── test_split_and_augmentation.py
│   ├── test_api.py
│   └── test_model_zoo.py
└── pytest.ini
```

## Setup

```bash
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If the project virtual environment has not been created yet, create it with
`python -m venv .venv` first. In PowerShell, activate it with
`.venv\Scripts\Activate.ps1`. The project commands below use that environment.

## Quick smoke check

```bash
.\.venv\Scripts\python.exe -m pytest -q
```

## Eight-model workflow

```bash
.\.venv\Scripts\python.exe -m glioma_hc.dataset_builder.import_kaggle_mri --source "PATH_TO_EXTRACTED_DATASET" --output data/processed
.\.venv\Scripts\python.exe -m glioma_hc.training.train_all --data-dir data/processed --epochs 10 --batch-size 8 --seed 42
.\.venv\Scripts\python.exe -m glioma_hc.evaluation.ensemble_evaluate --data-dir data/processed --split val --output reports/ensemble_validation.json
.\.venv\Scripts\python.exe -m glioma_hc.evaluation.ensemble_evaluate --data-dir data/processed --split test --output reports/weighted_soft_ensemble_test.json
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

The training command defaults to the required eight models: ResNet18, ResNet50, ResNet101, GoogLeNet, EfficientNet-B0, MobileNetV2, InceptionV3, and DenseNet121. It reports per-class split counts, uses ImageNet initialization for available Keras application backbones, uses the locally implemented ResNet18 and GoogLeNet architectures where the Keras applications API has no matching pretrained model, applies class weights when requested, and saves per-epoch training/validation loss and accuracy. The default training is frozen-backbone feature extraction. Add `--fine-tune` to first train the classification head and then selectively unfreeze the final 20% of each pretrained backbone at the lower `--fine-tune-learning-rate`; batch-normalization layers remain frozen. `--learning-rate`, `--batch-size`, and `--seed` are configurable. This is fresh training from ImageNet initialization (where available), not optimizer-state continuation from existing `.keras` files. Checkpoints are written to new seed-named paths and never overwrite an existing checkpoint; an existing summary is backed up before it is replaced. Use `--model-dir` and `--summary-output` to isolate an experiment. Application ImageNet weights may download the first time, and training all eight models may take substantial time and memory; a GPU is recommended.

Download and extract the [Brain Tumor MRI dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) separately; the extracted directory must contain `Training/` and `Testing/`, each with the four class folders. The importer keeps its four source labels (`glioma`, `meningioma`, `notumor`, and `pituitary`) and creates train, validation, and held-out test splits. The source `glioma` label is not specifically a glioblastoma diagnosis.

All model paths use the same deterministic image pipeline: RGB conversion, 300x300 common sizing, 3x3 median denoising, luminance-channel CLAHE, resizing to each architecture's input size, and architecture-specific input preprocessing. Small random rotations, translations, and brightness adjustments are enabled only for training batches. Horizontal flipping is omitted because left/right orientation has not been validated as label-preserving for this dataset. Automated skull/background removal is also omitted because this dataset has no validated brain masks; the UI discloses this.

Training compares hard voting, mean-probability soft voting, and validation-accuracy-weighted soft voting on validation data, then compares those with individual models. The active deployment uses the recovered eight-CNN weighted-soft-voting ensemble selected on validation. The existing one-epoch checkpoints are used as saved; no model was retrained to restore this selection.

The active held-out ensemble report is `reports/weighted_soft_ensemble_test.json`. Evaluation reports include accuracy, macro precision/recall/F1, per-class precision/sensitivity/specificity/F1/AUC, Cohen's kappa, confusion matrix, model/checkpoint metadata, and dataset provenance. The performance UI displays the measured result for the active predictor. `reports/final_test_results.json` and `reports/test_results.json` preserve separate earlier model evaluations.

Commands for evaluating the already-trained checkpoints (do not run the training command again just to recreate existing checkpoints):

```powershell
.\.venv\Scripts\python.exe -m glioma_hc.evaluation.ensemble_evaluate --data-dir data/processed --training-summary reports/training_summary.json --split val --output reports/ensemble_validation.json
.\.venv\Scripts\python.exe -m glioma_hc.evaluation.ensemble_evaluate --data-dir data/processed --training-summary reports/training_summary.json --split test --output reports/weighted_soft_ensemble_test.json
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Running the test evaluator again will replay all eight models across the test split and can take several minutes on a laptop CPU. The measured test metrics documented below were recovered from the completed all-model evaluation; they do not require or imply retraining.

Previously recovered one-epoch checkpoints have these validation results:

| Candidate | Validation accuracy | Validation macro F1 |
|---|---:|---:|
| ResNet18 | 62.77% | 56.63% |
| ResNet50 | 84.82% | 84.32% |
| ResNet101 | 86.79% | 86.39% |
| GoogLeNet | 25.00% | 10.00% |
| EfficientNet-B0 | 82.77% | 82.05% |
| MobileNetV2 | 83.30% | 82.77% |
| InceptionV3 | 83.21% | 83.41% |
| DenseNet121 | 80.71% | 80.15% |
| Weighted soft-voting ensemble | 86.43% | 86.25% |

The recovered eight-model validation comparison selected weighted soft voting among voting strategies at **86.43% accuracy** and **86.25% macro F1**; ResNet101 was its strongest individual member at **86.79% validation accuracy**. The active summary selects weighted soft voting on validation data, without consulting held-out test results.

The active eight-model ensemble was evaluated on the 1,600-image held-out split. It measured **79.19% accuracy**, **80.48% macro precision**, **79.19% macro sensitivity/recall**, **93.06% macro specificity**, **78.11% macro F1**, **95.91% macro one-vs-rest AUC**, and Cohen's kappa **0.7225**. Its test sensitivity was lowest for meningioma (**58.00%**) and glioma (**60.75%**). Metrics are recorded in `reports/weighted_soft_ensemble_test.json`.

For context, saved individual test accuracies were ResNet18 59.13%, ResNet50 79.25%, ResNet101 80.25%, GoogLeNet 25.00%, EfficientNet-B0 76.56%, MobileNetV2 79.25%, InceptionV3 78.63%, and DenseNet121 75.56%. The low ResNet18/GoogLeNet results and the high meningioma/glioma confusion indicate that averaging all models can dilute useful predictions. The evaluation was performed with the existing checkpoints; no new training was performed.

The ensemble report, including per-class metrics, confusion matrix, per-model test comparison, and metadata, is `reports/weighted_soft_ensemble_test.json`. Its confusion-matrix figure is `reports/figures/weighted_soft_ensemble_test_confusion_matrix.png`. The earlier `reports/final_test_results.json` contains the separate ResNet101 comparison. These scores describe the saved one-epoch checkpoints only.

After aggregate reporting, two individual test-fold images were submitted as API smoke checks for the prediction and Hybrid CAM paths. Those individual outputs were exploratory service checks only; they were not used to select models or strategies and are not included in the aggregate report metrics.

The >95% target has not been achieved or guaranteed. The available laptop has an AMD Ryzen 5 7520U CPU, about 5.8 GB total RAM, and no NVIDIA GPU. At the time of evaluation, free RAM was below 1 GB, so no further training was started. First close other applications or restart the laptop; begin only when at least 2 GB is available. The one-epoch validation results point to two bottlenecks: ResNet18 and GoogLeNet perform poorly (62.77% and 25.00% validation accuracy), while the other models range from 80.71% to 86.79%. Weighted voting still gives weak models influence. The stronger members also struggle with meningioma: ResNet101 validation sensitivity was 66.43%, and held-out ensemble test sensitivity was 58.00% for meningioma and 60.75% for glioma, with many confusions between tumor classes. This supports undertraining and class-boundary difficulty as improvement targets, but more epochs do not guarantee 95%.

Before any training, implement and test explicit warm-start support. Then fine-tune ResNet101 first and ResNet50 only if memory permits; use batch size 1–2 and a conservative learning rate. Start with the classification head, then consider unfreezing only the final residual stage. Track validation per-class sensitivity, macro F1, balanced accuracy, and confusion matrices; inspect misclassified images for data or label issues. Reassess whether to exclude or down-weight weak ensemble members using validation only. Do not spend the first limited-RAM experiment training ResNet18 or GoogLeNet from scratch. Require at least 2 GB free RAM before starting; that is a safety threshold, not a guarantee training will fit or complete quickly. Existing `.keras` files contain model weights, but the current trainer does not resume from them; continuing would require loading model weights/model structure, compiling a fresh optimizer at a lower learning rate, and recording explicitly that optimizer state was not restored.

The resume/fine-tune implementation must write to a new, uniquely named checkpoint path (never `saved_models/<existing-name>_fast.keras`), preserve the original `training_summary.json` and test report with backups, and store its run summary separately. Only after the changed candidate has been compared on validation should the ensemble configuration be reconsidered. Keep the test split untouched until that choice is final. The laptop-only experiment is a proposal, not work already performed.

```bash
# Do not run until warm-start support has been implemented and tested:
# .\.venv\Scripts\python.exe -m glioma_hc.training.train_all --models resnet101 --data-dir data/processed --epochs 3 --batch-size 1 --seed 42 --model-dir saved_models/laptop_experiments --summary-output reports/laptop_resnet101_summary.json
# .\.venv\Scripts\python.exe -m glioma_hc.evaluation.evaluate --model saved_models/laptop_experiments/<new_checkpoint>.keras --model-name resnet101 --data-dir data/processed --split val --output reports/laptop_resnet101_validation.json
```

The training command above is intentionally commented out because current code does not yet warm-start from an existing checkpoint. After implementation, review validation results before any final held-out test evaluation.

Hybrid CAM computes Grad-CAM and Guided Backpropagation for the same selected model, image, and ensemble-predicted class. It independently max-normalizes the two non-negative maps and fuses them using a pixelwise product, `H = normalize(G × B)`, where `G` is Grad-CAM and `B` is Guided Backpropagation. This favors locations supported by both maps; it is an attribution visualization, not segmentation, and it does not improve the classifier's accuracy.

The current deployment is the validation-selected saved Kaggle CNN checkpoint, served with its matching `kaggle_notebook_v1` preprocessing and verified held-out report. The eight-model ensemble remains saved for comparison and can be restored by changing the deployment selection only after its validation-based comparison is reviewed.

## Medical disclaimer

This project is for research and educational use only. It does not replace professional medical interpretation and is not a clinically validated diagnostic device.

## Citation

Pilaoon, S., Hamamoto, K., Maneerat, S. (2025). Glioma Brain Tumor Classification Using Convolution Neural Network and Majority Voting. IEEE Access.

## License

MIT. The project is inspired by and built with attribution to the original reference repository. The paper and associated research code remain the basis for this adapted pipeline.
