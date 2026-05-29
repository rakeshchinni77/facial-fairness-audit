from __future__ import annotations


EXPECTED_PLOTS = {
	"roc_curve_publication.png",
	"det_curve_publication.png",
	"subgroup_far_chart.png",
	"subgroup_frr_chart.png",
	"fairness_heatmap.png",
	"mitigation_comparison.png",
	"disparity_gap_plot.png",
	"fairness_dashboard.png",
}


def test_plots_directory_contains_expected_pngs(plots_dir) -> None:
	assert plots_dir.exists() and plots_dir.is_dir(), "Plots directory must exist"
	missing = [name for name in EXPECTED_PLOTS if not (plots_dir / name).exists()]
	assert not missing, f"Expected publication plots are missing: {missing}"
	for name in EXPECTED_PLOTS:
		plot_path = plots_dir / name
		assert plot_path.stat().st_size > 0, f"Plot file is empty: {plot_path}"
