from __future__ import annotations

import pandas as pd

from src.audit.audit_pipeline import load_threshold
from src.audit.embedding_extractor import EmbeddingExtractorConfig, load_embedding_extractor
from src.audit.fairness_metrics import compute_confusion_summary
from src.audit.similarity import pair_similarity
from src.audit.subgroup_evaluator import evaluate_subgroups
from tests.helpers import load_csv_strict


def test_end_to_end_smoke_pipeline(project_root, processed_dir, results_dir) -> None:
	metadata = load_csv_strict(processed_dir / "enriched_metadata.csv", required_columns={"age", "gender", "race", "subgroup"})
	assert not metadata.empty, "Metadata must not be empty for the end-to-end smoke test"

	extractor = load_embedding_extractor(
		EmbeddingExtractorConfig(
			checkpoint_path=project_root / "artifacts" / "best_model.pth",
			device="cpu",
			pretrained=False,
		)
	)
	pair_frame = load_csv_strict(processed_dir / "validation_pairs.csv", required_columns={"image_a", "image_b", "label"})
	first_row = pair_frame.iloc[0]
	embedding_a, embedding_b = extractor.extract_pair_embeddings(first_row["image_a"], first_row["image_b"])
	similarity = pair_similarity(embedding_a, embedding_b)
	threshold = load_threshold(results_dir / "threshold_analysis.json")
	predicted_label = int(similarity >= threshold)
	confusion = compute_confusion_summary([int(first_row["label"])], [predicted_label])
	fairness_frame = pd.DataFrame(
		[
			{
				"true_label": int(first_row["label"]),
				"predicted_label": predicted_label,
				"subgroup": str(first_row.get("subgroup", "unknown")),
			}
		]
	)
	subgroup_metrics = evaluate_subgroups(fairness_frame)

	assert similarity is not None, "The model must produce a similarity score"
	assert threshold is not None, "The threshold loader must return a usable operating point"
	assert confusion.support == 1, "The smoke test should evaluate one verification pair"
	assert subgroup_metrics, "The smoke test must run at least one subgroup fairness evaluation step"
