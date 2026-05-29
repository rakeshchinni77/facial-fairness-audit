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

The repository now includes a pytest-based validation layer that checks the data pipeline, preprocessing, demographic mappings, split integrity, pair and triplet generation, model wiring, triplet losses, training utilities, threshold analysis, audit outputs, mitigation artifacts, visualizations, JSON validity, and an end-to-end CPU smoke test.

Run the full suite locally:

```bash
python -m pytest
```

Run the smoke test only:

```bash
python -m pytest tests/test_end_to_end_pipeline.py -q
```

Validate inside Docker:

```bash
docker compose run --rm fairness-audit pytest -q
```

Coverage overview:

- Data and preprocessing checks ensure metadata integrity, subgroup enrichment, RGB conversion, normalization, and augmentation stability.
- Model checks verify the ResNet18 backbone, 128-dimensional embeddings, L2 normalization, and triplet-network forwarding.
- Audit checks verify threshold outputs, initial audit metrics, fairness summaries, mitigation reports, and visualization artifacts.
- Quality checks validate all `results/*.json` files and confirm that the PDF memo and plot outputs remain present and non-empty.

## Tech Stack

PyTorch, OpenCV, Fairlearn, NumPy, Pandas, scikit-learn, Matplotlib, Seaborn, SciPy, PyYAML, Jupyter, pytest, Docker, and Docker Compose.

## Ethical Fairness Focus

The future implementation will explicitly measure false accept and false reject rates across gender, age, and skin-tone proxies, then apply mitigation techniques with documented trade-offs. The goal is to prioritize responsible deployment over raw model accuracy.

## Docker Validation

Use the following commands to verify the container build and runtime state:

```bash
docker-compose up -d --build
docker ps
docker logs fairness-audit-container
docker-compose down
```

Expected result:

- Container status: `healthy`

The repository also includes a smoke validation script at [scripts/docker_smoke_test.py](scripts/docker_smoke_test.py) for direct runtime checks inside the project environment.
