# ML-2 Few-Shot Pathology Classification

This repository contains the core code, public report, submission file, and
evidence tables for the ML-2 5-class few-shot image classification project.

## Selected Method

The submitted model uses a single `UNI2-h` pathology foundation model with:

- `CenterScalePad(scale_to=64)` preprocessing for 32x32 RGB cell images.
- rsLoRA on attention `qkv` and `proj`.
- hard-class weighting for `Class_3` and `Class_4`.
- balanced soft pseudo-label training with 150 pseudo samples per class.
- full-250 single-refit inference without prediction-level TTA or ensemble.

Local 5-fold OOF selection score: `0.8671`.

Teacher/platform hidden feedback: `F1=0.5889`, `Recall=0.7202`, class rank top 5.

## Repository Layout

- `src/ml_final/`: project Python package.
- `configs/`: experiment and final submission configs.
- `scripts/`: experiment, training, prediction, and sync scripts.
- `tests/`: local smoke and plumbing tests.
- `report/`: LaTeX source, compiled PDF, and report figures.
- `artifacts/evidence/`: result tables and figures without model weights.
- `submission.csv`: course submission file.
- `artifacts/submissions/`: named copy of the submission.

## Data And Weights

The full course dataset, model checkpoints, Hugging Face cache, extracted
features, and run directories are intentionally excluded from this GitHub export.
To reproduce training or prediction, place the course data under:

```text
train_few_shot/
test_shuffled/
```

Pretrained model weights follow their upstream model cards and licenses. The
public report records the models and license notes used in the project.

## Report

The report is available at:

```text
report/machine_learning-2.pdf
report/machine_learning-2.tex
```

To rebuild the PDF on macOS with MacTeX:

```bash
cd report
/Library/TeX/texbin/xelatex -interaction=nonstopmode -halt-on-error machine_learning-2.tex
/Library/TeX/texbin/xelatex -interaction=nonstopmode -halt-on-error machine_learning-2.tex
/Library/TeX/texbin/xelatex -interaction=nonstopmode -halt-on-error machine_learning-2.tex
```

## Minimal Local Checks

```bash
python -m compileall src/ml_final tests configs scripts
python -m pytest -q
```

GPU training requires the relevant pathology foundation model weights to be
available locally or through Hugging Face access.
