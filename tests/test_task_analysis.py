from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from mlmom.contracts import (
    MetricDirection,
    ResultFormat,
    TaskDefinition,
    TaskType,
    ValidationStrategy,
)
from mlmom.data_inspector import DataInspector
from mlmom.task_graph import analyze_task
from mlmom.task_validator import TaskValidator


class FakeChatClient:
    model = "fake/test-model"

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.messages: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]]) -> str:
        self.messages.append(messages)
        if not self.responses:
            raise AssertionError("FakeChatClient has no response left")
        return self.responses.pop(0)


class TaskAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        pd.DataFrame({
            "id": range(60),
            "feature": [value / 10 for value in range(60)],
            "price": [value * 1.5 for value in range(60)],
        }).to_csv(root / "train.csv", index=False)
        pd.DataFrame([
            ["id", "id", "numeric", "identifier", "Row identifier"],
            ["feature", "feature", "numeric", "units", "Input feature"],
            ["price", "target", "numeric", "currency", "Target price"],
        ], columns=["column_name", "role", "type", "unit", "description"]).to_csv(
            root / "data_dictionary.csv", index=False
        )
        (root / "task.md").write_text(
            "Predict `price` with RMSE. Smaller is better.\n\n"
            "File `submission.csv`: CSV in UTF-8, comma delimiter, without index.\n"
            "```csv\nid,price\n```\n"
            "For each ID exactly one finite prediction is required. Missing predictions and "
            "duplicate IDs are not allowed. Row order must match the inference input.",
            encoding="utf-8",
        )
        self.profile = DataInspector().inspect_directory(root)

    def valid_task(self) -> TaskDefinition:
        return TaskDefinition(
            objective="Predict the numeric price target for every record.",
            task_type=TaskType.REGRESSION,
            target_columns=["price"],
            id_columns=["id"],
            excluded_feature_columns=["id", "price"],
            primary_metric="rmse",
            metric_direction=MetricDirection.MINIMIZE,
            validation_strategy=ValidationStrategy.KFOLD,
            result_format=self.profile.task.result_format,
            requirements=["One prediction per ID"],
            evidence=["task.md explicitly names price and RMSE"],
        )

    def test_validator_accepts_consistent_regression_task(self) -> None:
        report = TaskValidator().validate(self.profile, self.valid_task())

        self.assertTrue(report.valid)
        self.assertEqual(report.findings, [])

    def test_missing_mandatory_exclusions_are_added_without_llm_repair(self):
        task=self.valid_task().model_copy(update={'excluded_feature_columns':['feature']})
        client=FakeChatClient([task.model_dump_json()])
        result=analyze_task(self.profile,client)
        self.assertEqual(result.status,'validated')
        self.assertEqual(len(client.messages),1)
        self.assertEqual(result.task.excluded_feature_columns,['feature','price','id'])

    def test_validator_rejects_hallucinated_columns_metric_and_output(self) -> None:
        task = self.valid_task().model_copy(update={
            "target_columns": ["invented_target"],
            "primary_metric": "accuracy",
            "metric_direction": MetricDirection.MAXIMIZE,
            "excluded_feature_columns": ["id", "invented_target"],
            "result_format": ResultFormat(
                file_name="wrong.csv",
                columns=["id", "invented_target"],
                id_columns=["id"],
                prediction_columns=["invented_target"],
            ),
        })

        report = TaskValidator().validate(self.profile, task)
        codes = {finding.code for finding in report.findings}

        self.assertFalse(report.valid)
        self.assertIn("unknown_columns", codes)
        self.assertIn("target_mismatch", codes)
        self.assertIn("metric_mismatch", codes)
        self.assertIn("metric_not_applicable", codes)
        self.assertIn("result_format_mismatch", codes)

    def test_graph_repairs_invalid_first_answer(self) -> None:
        invalid = self.valid_task().model_copy(update={
            "primary_metric": "accuracy",
            "metric_direction": MetricDirection.MAXIMIZE,
        })
        client = FakeChatClient([
            json.dumps(invalid.model_dump(mode="json")),
            "```json\n" + json.dumps(self.valid_task().model_dump(mode="json")) + "\n```",
        ])

        result = analyze_task(self.profile, client, max_attempts=3)

        self.assertEqual(result.status, "validated")
        self.assertEqual(result.attempts, 2)
        self.assertEqual(result.task.primary_metric, "rmse")  # type: ignore[union-attr]
        self.assertEqual(len(client.messages), 2)
        repair_payload = json.loads(client.messages[1][1]["content"])
        self.assertIn("validation_feedback", repair_payload)
        self.assertIn("previous_task", repair_payload)

    def test_graph_stops_after_invalid_json_limit(self) -> None:
        client = FakeChatClient(["not json", "still not json"])

        result = analyze_task(self.profile, client, max_attempts=2)

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.attempts, 2)
        self.assertIsNone(result.task)
        self.assertEqual(result.validation.findings[0].code, "invalid_llm_output")


if __name__ == "__main__":
    unittest.main()
