"""
AEOS Phase 1 Results Audit

Builds reproducible "latest valid run" tables by:
1) Loading the 33-run Phase 1 matrix from orchestrator.py
2) Parsing exp1_*.json files in results/
3) Excluding zero-byte, malformed, failed, or misconfigured runs
4) Selecting latest valid run per model x dataset
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EXP1_RE = re.compile(r"^exp1_(.+)_(tabular|text|vision)_(\d{8}_\d{6})\.json$")
PROVIDER_PREFIXES = ("ollama/", "openai/", "anthropic/", "gemini/")


@dataclass
class AuditRecord:
    file: Path
    model: str
    dataset: str
    timestamp: str
    best_accuracy: float | None
    best_iteration: int | None
    total_iterations: int | None
    waste_count: int | None
    stop_reason: str | None
    valid: bool
    crash_flagged: bool
    reason: str


def _load_exp1_matrix(orchestrator_path: Path) -> list[dict[str, Any]]:
    tree = ast.parse(orchestrator_path.read_text(encoding="utf-8"), filename=str(orchestrator_path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "EXPERIMENT_1_MATRIX":
                return ast.literal_eval(node.value)
    raise ValueError("EXPERIMENT_1_MATRIX not found in orchestrator.py")


def _to_file_model(model: str) -> str:
    return model.replace("/", "_").replace(":", "-")


def _normalize_payload_model(payload_model: str | None) -> str | None:
    if not payload_model:
        return None
    model = payload_model
    for prefix in PROVIDER_PREFIXES:
        if model.startswith(prefix):
            model = model[len(prefix):]
            break
    return _to_file_model(model)


def _coerce_positive_number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and value > 0:
        return float(value)
    return None


def _coerce_positive_int(value: Any) -> int | None:
    if isinstance(value, int) and value > 0:
        return value
    return None


def audit_results(results_dir: Path, expected_pairs: set[tuple[str, str]]) -> list[AuditRecord]:
    records: list[AuditRecord] = []

    for file in sorted(results_dir.glob("exp1_*.json")):
        match = EXP1_RE.match(file.name)
        if not match:
            continue
        file_model, file_dataset, ts = match.groups()

        if file.stat().st_size == 0:
            records.append(
                AuditRecord(file, file_model, file_dataset, ts, None, None, None, None, None, False, False, "zero_byte")
            )
            continue

        try:
            payload = json.loads(file.read_text(encoding="utf-8"))
        except Exception:
            records.append(
                AuditRecord(file, file_model, file_dataset, ts, None, None, None, None, None, False, False, "invalid_json")
            )
            continue

        payload_model = _normalize_payload_model(payload.get("model"))
        payload_dataset = payload.get("dataset")
        stop_reason_raw = payload.get("stop_reason", "") or ""
        crash_flagged = any(term in stop_reason_raw.lower() for term in ["crash", "charmap", "encoding", "traceback"])

        if payload_model and payload_model != file_model:
            reason = "misconfigured_model_mismatch"
            valid = False
        elif payload_dataset and payload_dataset != file_dataset:
            reason = "misconfigured_dataset_mismatch"
            valid = False
        elif (file_model, file_dataset) not in expected_pairs:
            reason = "out_of_phase1_scope"
            valid = False
        else:
            best = _coerce_positive_number(payload.get("best_accuracy"))
            total_iters = _coerce_positive_int(payload.get("total_iterations"))
            iterations_list = payload.get("iterations", [])
            has_successful_iter = any(
                isinstance(it, dict) and it.get("val_accuracy") is not None and it.get("val_accuracy", 0) > 0
                for it in iterations_list
            ) if isinstance(iterations_list, list) else False

            if best is None:
                reason = "non_positive_best_accuracy"
                valid = False
            elif total_iters is None:
                reason = "non_positive_total_iterations"
                valid = False
            elif not has_successful_iter:
                reason = "no_successful_iterations"
                valid = False
            else:
                reason = "ok"
                valid = True

        records.append(
            AuditRecord(
                file=file,
                model=file_model,
                dataset=file_dataset,
                timestamp=ts,
                best_accuracy=payload.get("best_accuracy"),
                best_iteration=payload.get("best_iteration"),
                total_iterations=payload.get("total_iterations"),
                waste_count=payload.get("waste_count"),
                stop_reason=stop_reason_raw or None,
                valid=valid,
                crash_flagged=crash_flagged,
                reason=reason,
            )
        )

    return records


def latest_valid_by_pair(records: list[AuditRecord]) -> dict[tuple[str, str], AuditRecord]:
    by_pair: dict[tuple[str, str], list[AuditRecord]] = defaultdict(list)
    for record in records:
        if record.valid:
            by_pair[(record.model, record.dataset)].append(record)

    latest: dict[tuple[str, str], AuditRecord] = {}
    for pair, items in by_pair.items():
        latest[pair] = max(items, key=lambda r: r.timestamp)
    return latest


def leaderboard_markdown(latest: dict[tuple[str, str], AuditRecord]) -> str:
    models = sorted({model for model, _ in latest})
    lines = [
        "| Model | Tabular | Text | Vision |",
        "|---|---|---|---|",
    ]
    for model in models:
        cells = [f"`{model}`"]
        for dataset in ("tabular", "text", "vision"):
            record = latest.get((model, dataset))
            if record is None:
                cells.append("N/A")
                continue
            acc = float(record.best_accuracy) * 100
            best_it = record.best_iteration if record.best_iteration is not None else "?"
            total_it = record.total_iterations if record.total_iterations is not None else "?"
            flag = " \u2020" if record.crash_flagged else ""
            cells.append(f"{acc:.2f}% (iter {best_it}/{total_it}){flag}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit AEOS exp1 JSON files and build reproducible latest-run tables.")
    parser.add_argument("--aeos-dir", default=str(Path(__file__).parent), help="Path to aeos directory")
    parser.add_argument("--output-json", default=None, help="Optional output JSON path for latest valid runs")
    args = parser.parse_args()

    aeos_dir = Path(args.aeos_dir).resolve()
    results_dir = aeos_dir / "results"
    orchestrator_path = aeos_dir / "orchestrator.py"

    matrix = _load_exp1_matrix(orchestrator_path)
    expected_pairs = {
        (_to_file_model(item["model"]), item["dataset"])
        for item in matrix
    }

    records = audit_results(results_dir, expected_pairs)
    reasons = Counter(record.reason for record in records)
    latest = latest_valid_by_pair(records)

    missing_pairs = sorted(expected_pairs - set(latest.keys()))

    print("AEOS Phase 1 Results Audit")
    print(f"Results dir: {results_dir}")
    print(f"Total exp1 files scanned: {len(records)}")
    print(f"Valid files: {sum(1 for r in records if r.valid)}")
    print(f"Invalid files: {sum(1 for r in records if not r.valid)}")
    print("Invalid reason counts:")
    for reason, count in sorted(reasons.items()):
        if reason == "ok":
            continue
        print(f"  - {reason}: {count}")

    print(f"\nPhase 1 expected combos: {len(expected_pairs)}")
    print(f"Phase 1 combos with >=1 latest valid run: {len(latest)}")
    if missing_pairs:
        print("Missing combos:")
        for model, dataset in missing_pairs:
            print(f"  - {model} x {dataset}")
    else:
        print("Missing combos: 0")

    crash_flagged_valid = [(r.model, r.dataset) for r in records if r.valid and r.crash_flagged]
    if crash_flagged_valid:
        print("\n[WARN] Valid runs with crash/encoding stop-reason (marked † in leaderboard):")
        for m, d in crash_flagged_valid:
            print(f"  - {m} x {d}")

    print("\nLatest Valid Leaderboard (Phase 1 matrix only)")
    print(leaderboard_markdown(latest))

    if args.output_json:
        out_path = Path(args.output_json).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "summary": {
                "total_files": len(records),
                "valid_files": sum(1 for r in records if r.valid),
                "invalid_files": sum(1 for r in records if not r.valid),
                "expected_pairs": len(expected_pairs),
                "pairs_with_valid_latest": len(latest),
                "missing_pairs": [{"model": m, "dataset": d} for m, d in missing_pairs],
                "invalid_reason_counts": {k: v for k, v in sorted(reasons.items()) if k != "ok"},
            },
            "latest_valid_runs": [
                {
                    "model": record.model,
                    "dataset": record.dataset,
                    "file": str(record.file),
                    "timestamp": record.timestamp,
                    "best_accuracy": record.best_accuracy,
                    "best_iteration": record.best_iteration,
                    "total_iterations": record.total_iterations,
                    "waste_count": record.waste_count,
                    "stop_reason": record.stop_reason,
                    "crash_flagged": record.crash_flagged,
                }
                for (_, _), record in sorted(latest.items())
            ],
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nSaved latest valid runs JSON: {out_path}")


if __name__ == "__main__":
    main()
