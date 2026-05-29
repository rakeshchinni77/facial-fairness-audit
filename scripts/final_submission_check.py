from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
ARTIFACTS_DIR = ROOT / "artifacts"
SUBMISSION_DIR = ROOT / "submission"
REPORT_PATH = SUBMISSION_DIR / "final_submission_report.json"


REQUIRED_JSON_FILES = [
    RESULTS_DIR / "demographics.json",
    RESULTS_DIR / "threshold_analysis.json",
    RESULTS_DIR / "initial_audit.json",
    RESULTS_DIR / "cross_group_metrics.json",
    RESULTS_DIR / "fairness_summary.json",
    RESULTS_DIR / "analysis.json",
    RESULTS_DIR / "mitigated_audit.json",
    RESULTS_DIR / "fairness_comparison.json",
    RESULTS_DIR / "overall_metrics.json",
]

REQUIRED_MODEL_FILES = [
    ARTIFACTS_DIR / "model.pth",
    ARTIFACTS_DIR / "best_model.pth",
    ARTIFACTS_DIR / "mitigated_model.pth",
    ARTIFACTS_DIR / "best_mitigated_model.pth",
]

REQUIRED_OTHER_FILES = [
    ROOT / "README.md",
    ROOT / "Dockerfile",
    ROOT / "docker-compose.yml",
    SUBMISSION_DIR / "deployment_memo.pdf",
]

README_REQUIRED_SECTIONS = [
    "project overview",
    "fairness methodology",
    "mitigation strategy",
    "results",
    "ethical considerations",
    "deployment recommendation",
]

MIN_MODEL_SIZE_BYTES = 1 * 1024 * 1024


def exists_all(paths: List[Path]) -> Tuple[bool, List[str]]:
    missing = [str(p.relative_to(ROOT)) for p in paths if not p.exists()]
    return len(missing) == 0, missing


def validate_models(paths: List[Path]) -> Tuple[bool, Dict[str, float], List[str]]:
    sizes_mb: Dict[str, float] = {}
    failures: List[str] = []

    for path in paths:
        rel = str(path.relative_to(ROOT))
        if not path.exists():
            failures.append(f"missing:{rel}")
            continue

        size_bytes = path.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        sizes_mb[rel] = round(size_mb, 3)

        if size_bytes <= MIN_MODEL_SIZE_BYTES:
            failures.append(f"too_small:{rel}:{size_mb:.3f}MB")

    return len(failures) == 0, sizes_mb, failures


def validate_pdf(path: Path) -> Tuple[bool, int]:
    if not path.exists():
        return False, 0
    size = path.stat().st_size
    return size > 0, size


def validate_readme(readme_path: Path) -> Tuple[bool, List[str]]:
    if not readme_path.exists():
        return False, README_REQUIRED_SECTIONS

    content = readme_path.read_text(encoding="utf-8", errors="replace").lower()
    missing = [section for section in README_REQUIRED_SECTIONS if section not in content]
    return len(missing) == 0, missing


def validate_docker_verification(path: Path) -> Tuple[bool, Dict[str, object]]:
    details: Dict[str, object] = {
        "build_success": False,
        "container_healthy": False,
        "source": str(path.relative_to(ROOT)),
    }
    if not path.exists():
        return False, details

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False, details

    details["build_success"] = bool(payload.get("build_success", False))
    details["container_healthy"] = bool(payload.get("container_healthy", False))
    details["notes"] = payload.get("notes", "")

    return bool(details["build_success"]) and bool(details["container_healthy"]), details


def main() -> int:
    audits_ok, missing_json = exists_all(REQUIRED_JSON_FILES)
    other_ok, missing_other = exists_all(REQUIRED_OTHER_FILES)

    models_ok, model_sizes_mb, model_failures = validate_models(REQUIRED_MODEL_FILES)

    memo_pdf = SUBMISSION_DIR / "deployment_memo.pdf"
    pdf_ok, pdf_size_bytes = validate_pdf(memo_pdf)

    readme_ok, missing_sections = validate_readme(ROOT / "README.md")

    docker_ok, docker_details = validate_docker_verification(RESULTS_DIR / "docker_verification.json")

    report = {
        "repository_ready": bool(
            audits_ok and other_ok and models_ok and pdf_ok and readme_ok and docker_ok
        ),
        "models_verified": models_ok,
        "audit_reports_verified": audits_ok,
        "deployment_memo_verified": pdf_ok,
        "docker_verified": docker_ok,
        "readme_verified": readme_ok,
        "details": {
            "missing_json_files": missing_json,
            "missing_required_files": missing_other,
            "model_sizes_mb": model_sizes_mb,
            "model_validation_failures": model_failures,
            "deployment_memo_size_bytes": pdf_size_bytes,
            "readme_missing_sections": missing_sections,
            "docker_verification": docker_details,
        },
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"REPOSITORY_READY={report['repository_ready']}")
    print(f"MODELS_VERIFIED={report['models_verified']}")
    print(f"AUDITS_VERIFIED={report['audit_reports_verified']}")
    print(f"DOCKER_VERIFIED={report['docker_verified']}")
    print(f"README_VERIFIED={report['readme_verified']}")

    for model_path, size in sorted(model_sizes_mb.items()):
        print(f"MODEL_SIZE_MB {model_path}={size}")

    print(f"DEPLOYMENT_MEMO_BYTES={pdf_size_bytes}")
    print(f"FINAL_REPORT={REPORT_PATH.relative_to(ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
