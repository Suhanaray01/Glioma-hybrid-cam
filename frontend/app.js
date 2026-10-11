const tabs = [...document.querySelectorAll('.tab')];
const panels = [...document.querySelectorAll('.panel')];
const systemState = document.getElementById('systemState');
const systemStateText = document.getElementById('systemStateText');
const fileInput = document.getElementById('fileInput');
const explanationMode = document.getElementById('explainMode');
const analyzeButton = document.getElementById('analyzeBtn');
const inputPreviewFrame = document.getElementById('inputPreviewFrame');
const inputPreview = document.getElementById('inputPreview');
const resultState = document.getElementById('resultState');
const emptyState = document.getElementById('emptyState');
const resultContent = document.getElementById('resultContent');
let previewUrl = null;
let reportUrl = null;

for (const tab of tabs) {
  tab.addEventListener('click', () => {
    const target = tab.dataset.tab;
    for (const item of tabs) {
      const active = item === tab;
      item.classList.toggle('active', active);
      item.setAttribute('aria-selected', String(active));
    }
    for (const panel of panels) panel.classList.toggle('active', panel.id === target);
    if (target === 'performance') loadMetrics();
  });
}

async function loadStatus() {
  try {
    const response = await fetch('/api/health');
    const health = await response.json();
    const ready = response.ok && health.status === 'ok'
      && (health.predictor_ready === true || health.ensemble_ready === true);
    const deployment = health.deployment || {};
    systemState.dataset.ready = String(ready);
    systemState.dataset.readyState = String(ready);
    systemStateText.textContent = ready
      ? deployment.kind === 'single_model'
        ? `${deployment.model === 'kaggle_cnn' ? 'Kaggle CNN' : deployment.model || 'Validated model'} selected`
        : 'Eight-CNN ensemble ready'
      : `${health.models_loaded || 0}/${health.models_required || 8} CNNs trained · ensemble unavailable`;
  } catch {
    systemState.dataset.ready = 'false';
    systemStateText.textContent = 'API unavailable';
  }
}

fileInput.addEventListener('change', () => {
  const file = fileInput.files[0];
  document.getElementById('fileName').textContent = file ? file.name : 'No image selected';
  analyzeButton.disabled = !file;
  resultContent.hidden = true;
  emptyState.hidden = false;
  resultState.textContent = 'Awaiting analysis';
  emptyState.querySelector('p').textContent = file ? 'Ready to analyze the selected image.' : 'Select an image to begin.';
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  inputPreviewFrame.hidden = !file;
  document.getElementById('inputDetails').hidden = !file;
  if (file) {
    previewUrl = URL.createObjectURL(file);
    inputPreview.src = previewUrl;
    inputPreview.onload = () => {
      document.getElementById('inputResolution').textContent =
        `${inputPreview.naturalWidth} × ${inputPreview.naturalHeight} px`;
    };
    document.getElementById('inputModality').textContent = 'User-provided raster; not independently verified';
  } else {
    inputPreview.removeAttribute('src');
    document.getElementById('inputResolution').textContent = '—';
  }
});

function showWarnings(warnings) {
  const list = document.getElementById('warningList');
  list.replaceChildren();
  for (const warning of warnings || []) {
    const item = document.createElement('li');
    item.textContent = warning;
    list.append(item);
  }
}

function renderPrediction(prediction) {
  const classLabels = {
    glioma: 'Glioma (not glioblastoma-specific)',
    meningioma: 'Meningioma',
    notumor: 'No tumor',
    pituitary: 'Pituitary',
  };
  const modelLabels = {
    kaggle_cnn: 'Kaggle CNN',
    resnet18: 'ResNet18',
    resnet50: 'ResNet50',
    resnet101: 'ResNet101',
    googlenet: 'GoogLeNet',
    efficientnet_b0: 'EfficientNet-B0',
    mobilenet_v2: 'MobileNetV2',
    inception_v3: 'InceptionV3',
    densenet121: 'DenseNet121',
  };
  const label = classLabels[prediction.label] || prediction.label;
  const confidence = Number(prediction.confidence);
  document.getElementById('resultLabel').textContent = label;
  document.getElementById('classConfidence').textContent = `${(confidence * 100).toFixed(1)}%`;
  document.getElementById('probabilityFill').style.width = `${Math.max(0, Math.min(100, confidence * 100))}%`;
  const classProbabilityList = document.getElementById('classProbabilityList');
  classProbabilityList.replaceChildren();
  for (const [className, probability] of Object.entries(prediction.class_probabilities || {})) {
    const item = document.createElement('li');
    const classLabel = document.createElement('span');
    classLabel.textContent = classLabels[className] || className;
    const classProbability = document.createElement('strong');
    classProbability.textContent = `${(Number(probability) * 100).toFixed(1)}%`;
    item.append(classLabel, classProbability);
    classProbabilityList.append(item);
  }
  const ensemble = prediction.ensemble || {};
  const predictor = prediction.predictor || {};
  const singleModel = predictor.kind === 'single_model';
  const predictorName = modelLabels[predictor.model] || predictor.model;
  document.getElementById('predictorTitle').textContent = singleModel
    ? 'VALIDATION-SELECTED SINGLE MODEL'
    : 'VALIDATION-SELECTED ENSEMBLE';
  document.getElementById('aggregation').textContent = singleModel
    ? predictorName || 'Single model'
    : ensemble.strategy || '—';
  document.getElementById('modelAgreement').textContent =
    singleModel
      ? 'Not applicable · single model'
      : `${ensemble.models_agreeing ?? 0} / ${ensemble.total_models ?? prediction.models.length}`;
  document.getElementById('inferenceTime').textContent = `${Number(prediction.timing_ms).toFixed(1)} ms`;
  const decisionCard = document.getElementById('decisionCard');
  const noTumor = prediction.label === 'notumor';
  decisionCard.dataset.tumor = String(!noTumor);
  document.getElementById('decisionTitle').textContent = noTumor ? 'NO TUMOR CLASS PREDICTED' : 'TUMOR CLASS PREDICTED';
  document.getElementById('decisionDetail').textContent =
    `${label} · ${formatPercent(confidence)} model confidence. This is a dataset-class prediction, not a diagnosis.`;

  const explanation = prediction.explain || {};
  const hybridMode = explanation.requested === 'hybrid_cam';
  const explanationGrid = document.getElementById('explanationGrid');
  const explanationImages = [
    ['originalResultImage', 'downloadOriginal', prediction.original_image],
    ['gradcamImage', 'downloadGradcam', explanation.gradcam_overlay],
    ['guidedImage', 'downloadGuided', explanation.guided_backprop],
    ['hybridImage', 'downloadHybrid', explanation.hybrid_cam_overlay],
  ];
  explanationGrid.hidden = !prediction.original_image;
  for (const [imageId, linkId, dataUrl] of explanationImages) {
    const image = document.getElementById(imageId);
    const link = document.getElementById(linkId);
    image.hidden = !dataUrl;
    if (dataUrl) {
      image.src = dataUrl;
      link.href = dataUrl;
      link.hidden = false;
    } else {
      image.removeAttribute('src');
      link.removeAttribute('href');
      link.hidden = true;
    }
  }
  document.getElementById('explanationMethod').textContent =
    hybridMode ? 'Grad-CAM + Guided Backpropagation → Hybrid CAM' : 'Model explanation';
  document.getElementById('explanationState').textContent = explanation.status === 'generated'
    ? `Generated for ${modelLabels[explanation.model] || explanation.model} · ${label}`
    : explanation.status === 'disabled' ? 'Not requested' : 'Unavailable';
  document.getElementById('explanationNote').textContent = explanation.status === 'generated'
    ? 'Grad-CAM highlights coarse spatial influence; Guided Backprop highlights positive input gradients. The maps show model attribution only, not tumor segmentation or a medical diagnosis.'
    : explanation.reason || 'Grad-CAM could not be generated for this model.';
  document.getElementById('hybridMethod').textContent = explanation.fusion_method
    ? `Fusion method: ${explanation.fusion_method}.`
    : '';
  const overlapStat = document.getElementById('attributionOverlap');
  overlapStat.hidden = !hybridMode || explanation.attribution_overlap === null || explanation.attribution_overlap === undefined;
  if (!overlapStat.hidden) {
    overlapStat.textContent = `Map overlap: ${(Number(explanation.attribution_overlap) * 100).toFixed(1)}% · not classification precision`;
  }
  const input = prediction.input || {};
  document.getElementById('inputModality').textContent = input.modality || 'User-provided raster; not independently verified';
  document.getElementById('inputResolution').textContent =
    input.original_width && input.original_height ? `${input.original_width} × ${input.original_height} px` : '—';
  document.getElementById('inputCommonSize').textContent =
    input.common_preprocessing_size ? `${input.common_preprocessing_size[1]} × ${input.common_preprocessing_size[0]} px` : '—';
  document.getElementById('inputModelSizes').textContent = Object.entries(input.preprocessed_model_inputs || {})
    .map(([name, size]) => `${modelLabels[name] || name}: ${size[1]} × ${size[0]}`)
    .join('; ') || '—';
  document.getElementById('inputSteps').textContent = (prediction.preprocessing?.steps || []).join(' ');
  document.getElementById('inputSkullRemoval').textContent =
    prediction.preprocessing?.skull_background_removal || 'Not reported';

  const modelRows = (prediction.models || []).map((model) => [
    modelLabels[model.name] || model.name,
    classLabels[model.label] || model.label,
    formatPercent(model.confidence),
    model.agrees_with_ensemble ? 'Yes' : 'No',
  ]);
  appendTable(
    document.getElementById('modelResultsTable'),
    ['CNN architecture', 'Prediction', 'Top-class confidence', singleModel ? 'Selected predictor' : 'Agrees with selected predictor'],
    modelRows,
    true,
  );
  document.getElementById('modelResultsCount').textContent =
    `${modelRows.length} ${modelRows.length === 1 ? 'model' : 'models'}`;
  document.getElementById('agreementDetail').textContent =
    singleModel
      ? `${predictorName || 'The selected model'} predicts ${label}; there is no ensemble vote.`
      : `${ensemble.models_agreeing ?? 0} of ${ensemble.total_models ?? prediction.models.length} models predict ${label}.`;
  document.getElementById('predictorDetail').textContent = singleModel
    ? `${predictorName || 'A single model'} was selected using validation metrics; other models are retained for comparison.`
    : `All eight CNNs vote using ${ensemble.strategy || 'the validation-selected strategy'}.`;

  const evaluation = prediction.evaluation_metrics || {};
  const classMetrics = evaluation.per_class?.[prediction.label] || {};
  const setMetric = (id, value) => {
    document.getElementById(id).textContent =
      value === null || value === undefined || !Number.isFinite(Number(value))
        ? 'N/A'
        : formatPercent(value);
  };
  setMetric('predictionAccuracy', evaluation.accuracy);
  setMetric('predictionRecall', classMetrics.sensitivity ?? classMetrics.recall);
  setMetric('predictionSpecificity', classMetrics.specificity);
  setMetric('predictionPrecision', classMetrics.precision);
  setMetric('predictionF1', classMetrics.f1);
  setMetric('predictionAuc', classMetrics.auc);
  const kappa = evaluation.cohen_kappa;
  document.getElementById('predictionKappa').textContent =
    kappa === null || kappa === undefined || !Number.isFinite(Number(kappa))
      ? 'N/A'
      : Number(kappa).toFixed(3);

  if (reportUrl) URL.revokeObjectURL(reportUrl);
  reportUrl = URL.createObjectURL(new Blob([JSON.stringify(prediction, null, 2)], { type: 'application/json' }));
  document.getElementById('downloadReport').href = reportUrl;
  showWarnings(prediction.warnings);

  emptyState.hidden = true;
  resultContent.hidden = false;
  resultState.textContent = 'Analysis complete';
}

analyzeButton.addEventListener('click', async () => {
  const file = fileInput.files[0];
  if (!file) return;
  if (file.size > 25 * 1024 * 1024) {
    resultState.textContent = 'File exceeds 25 MiB';
    return;
  }

  analyzeButton.disabled = true;
  resultState.textContent = 'Analyzing';
  emptyState.hidden = false;
  emptyState.querySelector('p').textContent = 'Running model inference…';
  resultContent.hidden = true;
  try {
    const formData = new FormData();
    formData.append('files', file);
    formData.append('sequence', 'UNSPECIFIED');
    formData.append('explain', explanationMode.value);
    const response = await fetch('/api/predict', { method: 'POST', body: formData });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || 'Analysis failed');
    renderPrediction(payload);
  } catch (error) {
    emptyState.querySelector('p').textContent = error.message;
    resultState.textContent = 'Analysis failed';
  } finally {
    analyzeButton.disabled = !fileInput.files.length;
  }
});

function formatPercent(value) {
  const numericValue = Number(value);
  return value === null || value === undefined || !Number.isFinite(numericValue)
    ? 'N/A'
    : `${(numericValue * 100).toFixed(2)}%`;
}

function appendTable(table, headers, rows, rowHeader = false) {
  table.replaceChildren();
  const head = document.createElement('thead');
  const headerRow = document.createElement('tr');
  for (const label of headers) {
    const cell = document.createElement('th');
    cell.scope = 'col';
    cell.textContent = label;
    headerRow.append(cell);
  }
  head.append(headerRow);
  table.append(head);

  const body = document.createElement('tbody');
  for (const row of rows) {
    const tableRow = document.createElement('tr');
    for (const [index, value] of row.entries()) {
      const cell = document.createElement(index === 0 && rowHeader ? 'th' : 'td');
      if (index === 0 && rowHeader) cell.scope = 'row';
      cell.textContent = String(value);
      tableRow.append(cell);
    }
    body.append(tableRow);
  }
  table.append(body);
}

async function loadMetrics() {
  const state = document.getElementById('metricsState');
  const content = document.getElementById('metricsContent');
  state.hidden = false;
  content.hidden = true;
  state.querySelector('p').textContent = 'Loading saved evaluation…';
  try {
    const response = await fetch('/api/metrics');
    const payload = await response.json();
    const evaluation = payload.evaluation || {};
    if (!response.ok || payload.report_current !== true || !evaluation.metrics || !Object.keys(evaluation.metrics).length) {
      throw new Error(payload.stale_report_available
        ? 'The saved report is not current for the recorded validation-selected models. Run the final held-out evaluation to show current metrics.'
        : 'No held-out test evaluation report is available for the current validated predictor.');
    }
    const labels = [
      ['accuracy', 'Accuracy'],
      ['balanced_accuracy', 'Balanced accuracy'],
      ['macro_precision', 'Macro precision'],
      ['macro_recall', 'Macro sensitivity / recall'],
      ['macro_specificity', 'Macro specificity'],
      ['macro_f1', 'Macro F1 score'],
      ['macro_auc', 'Macro ROC AUC'],
      ['cohen_kappa', "Cohen's kappa"],
    ];
    const grid = document.getElementById('metricGrid');
    grid.replaceChildren();
    for (const [key, label] of labels) {
      if (evaluation.metrics[key] === undefined) continue;
      const cell = document.createElement('div');
      cell.className = 'metric-cell';
      const name = document.createElement('span');
      name.textContent = label;
      const value = document.createElement('strong');
      value.textContent = key === 'cohen_kappa'
        ? Number(evaluation.metrics[key]).toFixed(3)
        : formatPercent(evaluation.metrics[key]);
      cell.append(name, value);
      grid.append(cell);
    }
    const classMetrics = document.getElementById('classMetrics');
    classMetrics.replaceChildren();
    if (evaluation.metrics.per_class) {
      const heading = document.createElement('h2');
      heading.textContent = 'Per-class performance';
      const table = document.createElement('table');
      table.id = 'perClassMetrics';
      const classRows = Object.entries(evaluation.metrics.per_class).map(([className, metrics]) => [
        className,
        metrics.support,
        formatPercent(metrics.precision),
        formatPercent(metrics.sensitivity ?? metrics.recall),
        formatPercent(metrics.specificity),
        formatPercent(metrics.f1),
        metrics.auc === null ? 'N/A' : formatPercent(metrics.auc),
      ]);
      appendTable(
        table,
        ['Class', 'Images', 'Precision', 'Sensitivity / recall', 'Specificity', 'F1 score', 'ROC AUC'],
        classRows,
        true,
      );
      classMetrics.append(heading, table);
    }

    const classNames = evaluation.label_order || Object.keys(evaluation.metrics.per_class || {});
    const matrix = evaluation.metrics.confusion_matrix || [];
    if (matrix.length && classNames.length === matrix.length) {
      appendTable(
        document.getElementById('confusionMatrix'),
        ['Actual \\ Predicted', ...classNames],
        matrix.map((row, index) => [classNames[index], ...row]),
        true,
      );
    }

    const splitCounts = evaluation.dataset_splits || {};
    const splitRows = ['train', 'val', 'test'].map((split) => [
      split === 'val' ? 'Validation' : split === 'test' ? 'Held-out test' : 'Training',
      ...classNames.map((name) => splitCounts[split]?.[name] ?? '—'),
      classNames.reduce((total, name) => total + (splitCounts[split]?.[name] || 0), 0) || '—',
    ]);
    appendTable(
      document.getElementById('datasetSplitTable'),
      ['Split', ...classNames, 'Total'],
      splitRows,
      true,
    );
    document.getElementById('provenanceLicense').textContent = evaluation.dataset_license || 'Not specified';
    document.getElementById('metricSplit').textContent = evaluation.evaluation_split === 'test'
      ? 'Held-out test set'
      : evaluation.evaluation_split || 'Not specified';
    document.getElementById('testCount').textContent = `${evaluation.evaluation_samples ?? evaluation.test_samples} ${evaluation.evaluation_split === 'val' ? 'validation' : 'held-out test'} images`;
    document.getElementById('metricDataset').textContent = evaluation.dataset || 'Unspecified dataset';
    document.getElementById('metricModel').textContent = evaluation.model || 'Unspecified model';
    const confidence = payload.prediction_confidence || {};
    document.getElementById('confidenceSummary').textContent = confidence.samples
      ? `Selected-prediction confidence summary across ${confidence.samples} images: mean ${(Number(confidence.mean) * 100).toFixed(1)}%, median ${(Number(confidence.median) * 100).toFixed(1)}%, range ${(Number(confidence.minimum) * 100).toFixed(1)}%–${(Number(confidence.maximum) * 100).toFixed(1)}%. Confidence is not calibration or diagnostic certainty.`
      : '';
    for (const [key, elementId] of [
      ['confusion_matrix', 'confusionFigure'],
      ['roc_curves', 'rocFigure'],
    ]) {
      const image = document.getElementById(elementId);
      const figureUrl = payload.figures?.[key];
      image.hidden = !figureUrl;
      if (figureUrl) image.src = figureUrl;
      else image.removeAttribute('src');
    }
    document.getElementById('metricCaveat').textContent = [
      evaluation.modality_note,
      evaluation.class_note,
      'Dataset-split metrics are not clinical performance or a guarantee for other scanners or populations.',
    ].filter(Boolean).join(' ');
    state.hidden = true;
    content.hidden = false;
  } catch (error) {
    state.querySelector('p').textContent = error.message;
  }
}

loadStatus();
