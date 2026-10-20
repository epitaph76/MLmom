from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from mlmom.contracts import FileRole, SemanticType, TaskType
from mlmom.data_inspector import DataInspector, build_llm_context
from mlmom.llm_report import build_llm_markdown


def write_dictionary(root: Path, rows: list[list[str]]) -> None:
    pd.DataFrame(
        rows,
        columns=["column_name", "role", "type", "unit", "description"],
    ).to_csv(root / "data_dictionary.csv", index=False)


class DataInspectorTests(unittest.TestCase):
    def test_manifest_uses_only_train_dictionary_and_task(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "tables").mkdir()
            train = pd.DataFrame({
                "record_id": range(30),
                "country_code": ["RU", "KZ", "BY"] * 10,
                "amount": range(30),
                "outcome": [0, 1, 2] * 10,
            })
            train.to_csv(root / "tables" / "learning_data.csv", index=False)
            (root / "test.csv").write_text('a,b\n"unterminated', encoding="utf-8")
            (root / "sample_submission.csv").write_text('id,target\n"unterminated', encoding="utf-8")
            write_dictionary(root, [
                ["record_id", "id", "numeric", "identifier", "Row key"],
                ["country_code", "feature", "category", "code", "Country"],
                ["amount", "feature", "numeric", "currency", "Amount"],
                ["outcome", "target", "category", "class", "Outcome class"],
            ])
            (root / "task.md").write_text(
                "Multiclass classification. Metric: macro F1.\n\n"
                "Result file `submission.csv` in UTF-8 without index:\n"
                "```csv\nrecord_id,outcome\n```",
                encoding="utf-8",
            )
            manifest = {
                "dataset_name": "arbitrary_names",
                "train": "tables/learning_data.csv",
                "dictionaries": ["data_dictionary.csv"],
                "task_description": "task.md",
            }
            (root / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            profile = DataInspector().inspect_directory(root)

            self.assertEqual(profile.task.task_type, TaskType.MULTICLASS_CLASSIFICATION)
            self.assertEqual(profile.task.target_columns, ["outcome"])
            self.assertEqual(profile.task.id_columns, ["record_id"])
            self.assertEqual(
                {file.role for file in profile.files},
                {FileRole.TRAIN, FileRole.DATA_DICTIONARY, FileRole.TASK_DESCRIPTION},
            )
            self.assertNotIn("test.csv", {file.file_name for file in profile.files})
            self.assertNotIn("sample_submission.csv", {file.file_name for file in profile.files})

    def test_regression_and_result_contract_are_inferred_without_test(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = pd.DataFrame({
                "id": range(100),
                "x": range(100),
                "price": [i * 1.5 for i in range(100)],
            })
            train.to_csv(root / "train.csv", index=False)
            write_dictionary(root, [
                ["id", "id", "numeric", "identifier", "Row ID"],
                ["x", "feature", "numeric", "units", "Feature"],
                ["price", "target", "numeric", "currency", "Target price"],
            ])
            (root / "task.md").write_text(
                "Predict `price` with RMSE. Smaller is better.\n\n"
                "File `submission.csv`: CSV in UTF-8, comma delimiter, without index.\n"
                "```csv\nid,price\n```\n"
                "For each ID exactly one prediction is required. Missing and infinite values are not allowed. "
                "Duplicate IDs are not allowed. Row order must match the inference input.",
                encoding="utf-8",
            )

            profile = DataInspector().inspect_directory(root)
            result = profile.task.result_format

            self.assertEqual(profile.schema_version, "3.0.0")
            self.assertEqual(profile.task.task_type, TaskType.REGRESSION)
            self.assertEqual(profile.task.target_columns, ["price"])
            self.assertEqual(profile.task.primary_metric, "rmse")
            self.assertEqual(result.file_name, "submission.csv")
            self.assertEqual(result.columns, ["id", "price"])
            self.assertEqual(result.encoding, "utf-8")
            self.assertEqual(result.delimiter, ",")
            self.assertFalse(result.include_index)
            self.assertEqual(result.row_order, "inference_input")
            self.assertFalse(result.missing_predictions_allowed)
            self.assertTrue(result.finite_predictions_required)
            self.assertFalse(result.duplicate_ids_allowed)
            target = profile.target_summaries["price"]
            self.assertEqual(sum(bucket["count"] for bucket in target["distribution"]), 100)

    def test_low_cardinality_numeric_target_is_marked_ambiguous(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pd.DataFrame({"feature": range(120), "target": [0, 1, 2] * 40}).to_csv(
                root / "train.csv", index=False
            )

            profile = DataInspector().inspect_directory(root)

            self.assertEqual(profile.task.target_columns, ["target"])
            self.assertEqual(profile.task.task_type, TaskType.MULTICLASS_CLASSIFICATION)
            self.assertEqual(profile.task.inference_confidence, "low")
            self.assertIn("ambiguous_task_definition", {issue.code for issue in profile.issues})

    def test_datetime_text_and_categories_are_profiled_from_train(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            descriptions = [f"This is a sufficiently long unique text record number {i}" for i in range(40)]
            pd.DataFrame({
                "id": range(40),
                "event_date": pd.date_range("2024-01-01", periods=40).astype(str),
                "description": descriptions,
                "category": ["a", "b"] * 20,
                "target": ["yes", "no"] * 20,
            }).to_csv(root / "train.csv", index=False)

            profile = DataInspector().inspect_directory(root)
            by_name = {column.name: column for column in profile.columns}

            self.assertEqual(by_name["event_date"].semantic_type, SemanticType.DATETIME)
            self.assertEqual(by_name["description"].semantic_type, SemanticType.TEXT)
            self.assertEqual(sum(bucket.count for bucket in by_name["category"].train.distribution), 40)
            self.assertFalse(hasattr(by_name["category"], "test"))

    def test_time_group_structure_and_prompt_reports_are_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pd.DataFrame({
                "id": range(200),
                "building_id": [i // 4 for i in range(200)],
                "period_code": [1, 2, 3, 4] * 50,
                "target": [float(i) for i in range(200)],
            }).to_csv(root / "train.csv", index=False)
            write_dictionary(root, [
                ["id", "id", "numeric", "identifier", "Row ID"],
                ["building_id", "group", "numeric", "identifier", "Building group"],
                ["period_code", "time", "numeric", "period code", "Ordered period code"],
                ["target", "target", "numeric", "units", "Target"],
            ])
            (root / "task.md").write_text("Regression with RMSE.", encoding="utf-8")

            profile = DataInspector().inspect_directory(root)
            context = build_llm_context(profile, max_chars=8_000)
            report = build_llm_markdown(profile, max_chars=8_000)

            self.assertEqual(profile.task.time_column, "period_code")
            self.assertEqual(profile.task.group_columns, ["building_id"])
            self.assertLessEqual(len(json.dumps(context, ensure_ascii=False, separators=(",", ":"))), 8_000)
            self.assertLessEqual(len(report), 8_000)
            self.assertIn("| Range or class |", report)
            self.assertIn("## Required result", report)
            self.assertIn("## Column overview", report)
            self.assertNotIn("Missing test", report)
            self.assertNotIn("train/test", report.lower())

    def test_duration_feature_is_not_mistaken_for_time_axis(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pd.DataFrame({
                "id": range(60),
                "synack": [value / 1000 for value in range(60)],
                "attack_cat": ["normal", "attack"] * 30,
            }).to_csv(root / "train.csv", index=False)
            write_dictionary(root, [
                ["id", "id", "numeric", "identifier", "Row ID"],
                ["synack", "feature", "numeric", "ms", "Время SYN → SYN-ACK"],
                ["attack_cat", "target", "category", "label", "Attack class"],
            ])
            (root / "task.md").write_text(
                "Binary classification. Metric: macro F1.", encoding="utf-8"
            )

            profile = DataInspector().inspect_directory(root)

            self.assertIsNone(profile.task.time_column)
            self.assertEqual(profile.task.task_type, TaskType.BINARY_CLASSIFICATION)


if __name__ == "__main__":
    unittest.main()
