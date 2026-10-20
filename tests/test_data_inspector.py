from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from mlmom.contracts import FileRole, SemanticType, TaskType
from mlmom.data_inspector import DataInspector, build_llm_context
from mlmom.llm_report import build_llm_markdown


class DataInspectorTests(unittest.TestCase):
    def test_manifest_supports_arbitrary_names_and_auxiliary_tables(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "tables").mkdir()
            (root / "reference").mkdir()
            train = pd.DataFrame({
                "record_id": range(30),
                "country_code": ["RU", "KZ", "BY"] * 10,
                "amount": range(30),
                "outcome": [0, 1, 2] * 10,
            })
            test = train.drop(columns="outcome").iloc[:8].copy()
            test["record_id"] = range(100, 108)
            train.to_csv(root / "tables" / "learning_data.csv", index=False)
            test.to_csv(root / "tables" / "scoring_data.csv", index=False)
            pd.DataFrame({"record_id": test["record_id"], "outcome": 0}).to_csv(
                root / "submission_format.csv", index=False
            )
            pd.DataFrame([
                ["record_id", "id", "numeric", "identifier", "Row key"],
                ["country_code", "feature", "category", "code", "Country"],
                ["amount", "feature", "numeric", "currency", "Amount"],
                ["outcome", "target", "category", "class", "Outcome class"],
            ], columns=["column_name", "role", "type", "unit", "description"]).to_csv(
                root / "fields.csv", index=False
            )
            pd.DataFrame({"country_code": ["RU", "KZ", "BY"], "region": ["A", "B", "C"]}).to_csv(
                root / "reference" / "countries.csv", index=False
            )
            (root / "task.md").write_text("Multiclass classification. Metric: macro F1.", encoding="utf-8")
            manifest = {
                "dataset_name": "arbitrary_names",
                "train": "tables/learning_data.csv",
                "test": "tables/scoring_data.csv",
                "sample_submission": "submission_format.csv",
                "dictionaries": ["fields.csv"],
                "task_description": "task.md",
                "auxiliary_tables": ["reference/countries.csv"],
                "target_columns": ["outcome"],
                "id_columns": ["record_id"],
                "task_type": "multiclass_classification",
                "primary_metric": "macro_f1",
            }
            (root / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            profile = DataInspector().inspect_directory(root)

            self.assertEqual(profile.task.task_type, TaskType.MULTICLASS_CLASSIFICATION)
            self.assertEqual(profile.task.target_columns, ["outcome"])
            self.assertEqual(profile.task.id_columns, ["record_id"])
            self.assertTrue(profile.checks.train_test_schema_match)
            self.assertIn(FileRole.AUXILIARY, {file.role for file in profile.files})
            self.assertTrue(any("country_code" in note for note in profile.report_notes))

    def test_regression_is_inferred_from_train_test_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = pd.DataFrame({"id": range(100), "x": range(100), "price": [i * 1.5 for i in range(100)]})
            test = pd.DataFrame({"id": range(100, 120), "x": range(20)})
            train.to_csv(root / "train.csv", index=False)
            test.to_csv(root / "test.csv", index=False)
            pd.DataFrame({"id": test["id"], "price": 0.0}).to_csv(root / "sample_submission.csv", index=False)
            (root / "task.md").write_text("Predict price. Evaluation metric is RMSE.", encoding="utf-8")

            profile = DataInspector().inspect_directory(root)

            self.assertEqual(profile.task.task_type, TaskType.REGRESSION)
            self.assertEqual(profile.task.target_columns, ["price"])
            self.assertEqual(profile.task.primary_metric, "rmse")
            target = profile.target_summaries["price"]
            self.assertEqual(sum(bucket["count"] for bucket in target["distribution"]), 100)

    def test_low_cardinality_numeric_target_is_marked_ambiguous(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = pd.DataFrame({"feature": range(120), "target": [0, 1, 2] * 40})
            test = pd.DataFrame({"feature": range(20)})
            train.to_csv(root / "train.csv", index=False)
            test.to_csv(root / "test.csv", index=False)

            profile = DataInspector().inspect_directory(root)

            self.assertEqual(profile.task.task_type, TaskType.MULTICLASS_CLASSIFICATION)
            self.assertEqual(profile.task.inference_confidence, "low")
            self.assertIn("ambiguous_task_definition", {issue.code for issue in profile.issues})

    def test_datetime_text_and_unknown_categories_are_profiled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            descriptions = [f"This is a sufficiently long unique text record number {i}" for i in range(40)]
            train = pd.DataFrame({
                "id": range(40),
                "event_date": pd.date_range("2024-01-01", periods=40).astype(str),
                "description": descriptions,
                "category": ["a", "b"] * 20,
                "target": ["yes", "no"] * 20,
            })
            test = train.drop(columns="target").iloc[:5].copy()
            test["id"] = range(100, 105)
            test.loc[0, "category"] = "new"
            train.to_csv(root / "train.csv", index=False)
            test.to_csv(root / "test.csv", index=False)

            profile = DataInspector().inspect_directory(root)
            by_name = {column.name: column for column in profile.columns}

            self.assertEqual(by_name["event_date"].semantic_type, SemanticType.DATETIME)
            self.assertEqual(by_name["description"].semantic_type, SemanticType.TEXT)
            self.assertEqual(by_name["category"].unseen_test_examples, ["new"])

    def test_unseen_period_code_and_prompt_reports_are_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = pd.DataFrame({
                "id": range(200),
                "period_code": [1, 2, 3, 4, 5] * 40,
                "target": [float(i) for i in range(200)],
            })
            test = pd.DataFrame({"id": range(200, 220), "period_code": [6] * 20})
            dictionary = pd.DataFrame([
                ["id", "id", "numeric", "identifier", "Row ID"],
                ["period_code", "feature", "numeric", "period code", "Ordered period code"],
                ["target", "target", "numeric", "units", "Target"],
            ], columns=["feature", "role", "dtype", "unit", "description"])
            train.to_csv(root / "train.csv", index=False)
            test.to_csv(root / "test.csv", index=False)
            dictionary.to_csv(root / "data_dictionary.csv", index=False)

            profile = DataInspector().inspect_directory(root)
            context = build_llm_context(profile, max_chars=8_000)
            report = build_llm_markdown(profile, max_chars=8_000)
            period = next(column for column in profile.columns if column.name == "period_code")

            self.assertEqual(period.unseen_test_examples, ["6"])
            self.assertLessEqual(len(json.dumps(context, ensure_ascii=False, separators=(",", ":"))), 8_000)
            self.assertLessEqual(len(report), 8_000)
            self.assertIn("| Range or class |", report)
            self.assertIn("## Column overview", report)


if __name__ == "__main__":
    unittest.main()
