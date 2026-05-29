# facial-fairness-audit

## Project Overview

`facial-fairness-audit` is a production-oriented foundation for building and auditing a facial verification system through the lens of algorithmic fairness. The project will combine metric learning, disaggregated evaluation, and mitigation workflows to support responsible AI delivery.

## Responsible AI Motivation

Facial verification systems can create unequal outcomes across demographic groups if they are trained or evaluated on imbalanced data. This repository is being structured to support reproducible fairness audits, transparent reporting, and future mitigation work before any high-stakes deployment is considered.

## Architecture Overview

The repository follows a modular `src/` layout with separate packages for data preparation, model components, training, audit, mitigation, evaluation, utilities, and API scaffolding. Containerization is handled at the repository root with Docker and Docker Compose.

## Folder Structure Summary

- `configs/`: YAML configuration placeholders for project, model, training, audit, and mitigation settings.
- `data/`: raw, processed, interim, and audit data locations.
- `notebooks/`: exploratory and analysis notebooks.
- `src/`: implementation modules organized by responsibility.
- `artifacts/`: model outputs, checkpoints, embeddings, and plots.
- `results/`: fairness audit outputs and analysis artifacts.
- `submission/`: final reporting deliverables.
- `tests/`: pipeline and component test placeholders.
- `scripts/`: shell entrypoints for later phase execution.

## Planned Phases

1. Project planning and architecture
2. Repository foundation and environment setup
3. Dataset acquisition and preprocessing
4. Demographic grouping and splitting strategy
5. Pair generation and metric learning model development
6. Training, validation, and threshold selection
7. Initial fairness audit and bias analysis
8. Mitigation, re-training, and re-audit
9. Trade-off analysis, memo writing, and final packaging

## Setup Instructions Placeholder

1. Create and activate a Python virtual environment.
2. Install dependencies from `requirements.txt`.
3. Build the container with Docker Compose.
4. Run the pipeline entrypoint from `main.py` once later phases are implemented.

## Testing and Validation

The project now includes a CPU-safe pytest suite covering data preparation, preprocessing, demographic mapping, split integrity, pair and triplet generation, model initialization, triplet losses, training utilities, threshold analysis, audit outputs, mitigation artifacts, visualizations, JSON validation, and a full end-to-end smoke path.

Run the tests locally:

```bash
python -m pytest
```

Run the end-to-end smoke test only:

```bash
python -m pytest tests/test_end_to_end_pipeline.py
```

Run the same validation inside Docker:

```bash
docker compose run --rm fairness-audit python -m pytest
```

Coverage overview:
- `tests/test_data_pipeline.py`: metadata, processed directories, and schema validation.
- `tests/test_preprocessing.py`: RGB conversion, 224x224 preprocessing, finite tensors, and augmentation smoke checks.
- `tests/test_demographics.py`: age bins, skin-tone mapping, and subgroup construction.
- `tests/test_split_pipeline.py`: 70/15/15 splits, disjoint IDs, and subgroup coverage.
- `tests/test_pair_generation.py`: pair labels, triplet structure, and subgroup metadata.
- `tests/test_models.py` and `tests/test_triplet_network.py`: ResNet18 backbone and embedding behavior.
- `tests/test_losses.py`: scalar triplet losses and subgroup-weighted loss stability.
- `tests/test_training_pipeline.py`: trainer components, checkpoint save path, and summary structure.
- `tests/test_threshold_pipeline.py`, `tests/test_audit_pipeline.py`, and `tests/test_mitigation_pipeline.py`: artifact integrity and report validity.
- `tests/test_visualizations.py`, `tests/test_artifacts.py`, and `tests/test_json_outputs.py`: plot presence, checkpoint robustness, and JSON sanity checks.
- `tests/test_end_to_end_pipeline.py`: smoke path that loads metadata, model, threshold, similarity scoring, and one fairness evaluation step.

## Tech Stack

PyTorch, OpenCV, Fairlearn, NumPy, Pandas, scikit-learn, Matplotlib, Seaborn, SciPy, PyYAML, Jupyter, pytest, Docker, and Docker Compose.

## Ethical Fairness Focus

The future implementation will explicitly measure false accept and false reject rates across gender, age, and skin-tone proxies, then apply mitigation techniques with documented trade-offs. The goal is to prioritize responsible deployment over raw model accuracy.
