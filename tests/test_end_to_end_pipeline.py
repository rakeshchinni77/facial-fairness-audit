from __future__ import annotations

import pandas as pd
import torch

from src.audit.audit_pipeline import load_threshold
from src.audit.embedding_extractor import EmbeddingExtractorConfig, load_embedding_extractor
from src.audit.similarity import pair_similarity
from src.audit.subgroup_evaluator import evaluate_subgroups
from src.data.demographic_mapper import enrich_metadata, load_metadata


def test_end_to_end_smoke_pipeline(project_root):
    metadata = load_metadata(project_root / "data" / "interim" / "metadata.csv")
    enriched = enrich_metadata(metadata)
    assert "subgroup" in enriched.columns, "Enriched metadata should contain subgroup labels"

    validation_pairs = pd.read_csv(project_root / "data" / "processed" / "validation_pairs.csv")
    first_pair = validation_pairs.iloc[0]
    image_a = project_root / str(first_pair["image_a"])
    image_b = project_root / str(first_pair["image_b"])

    extractor = load_embedding_extractor(
        EmbeddingExtractorConfig(
            checkpoint_path=project_root / "artifacts" / "best_model.pth",
            device="cpu",
        )
    )
    embedding_a, embedding_b = extractor.extract_pair_embeddings(image_a, image_b)
    similarity = pair_similarity(embedding_a.unsqueeze(0), embedding_b.unsqueeze(0))

    threshold = load_threshold(project_root / "results" / "threshold_analysis.json")
    predicted_label = int(similarity >= threshold)

    fairness_frame = pd.DataFrame(
        {
            "true_label": [int(first_pair["label"])],
            "predicted_label": [predicted_label],
            "subgroup": [str(first_pair["subgroup"])],
        }
    )
    subgroup_metrics = evaluate_subgroups(fairness_frame)
    assert subgroup_metrics, "End-to-end smoke test should produce subgroup metrics"
    assert torch.isfinite(torch.tensor(similarity)), "Similarity score should be finite"
