"""Run the offline feature suite and emit a machine-readable JSON summary."""

import json
import sys
import time
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


class StructuredTestResult(unittest.TestResult):
    def __init__(self):
        super().__init__()
        self.cases = []
        self._started = {}

    def startTest(self, test):
        super().startTest(test)
        self._started[test] = time.perf_counter()

    def _record(self, test, outcome, details=None):
        case = {
            "test": test.id(),
            "status": outcome,
            "duration_seconds": round(time.perf_counter() - self._started.pop(test), 4),
            "details": details,
        }
        method = getattr(test, test._testMethodName, None)
        feature_data = getattr(method, "feature_case", None)
        if feature_data:
            case["inputs"] = feature_data["inputs"]
            case["expected_output"] = feature_data["expected_output"]
            case["actual_output"] = getattr(test, "feature_output", None)
        self.cases.append(case)

    def addSuccess(self, test):
        super().addSuccess(test)
        self._record(test, "passed")

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self._record(test, "failed", self._exc_info_to_string(err, test))

    def addError(self, test, err):
        super().addError(test, err)
        self._record(test, "error", self._exc_info_to_string(err, test))

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self._record(test, "skipped", reason)

    def addSubTest(self, test, subtest, err):
        super().addSubTest(test, subtest, err)
        if err is not None:
            self._record(test, "failed", self._exc_info_to_string(err, test))


def main() -> int:
    suite = unittest.defaultTestLoader.discover(
        str(PROJECT_ROOT / "tests"),
        pattern="test_*.py",
        top_level_dir=str(PROJECT_ROOT),
    )
    result = StructuredTestResult()
    suite.run(result)
    summary = {
        "suite": "ResearchMate AI offline feature checks",
        "mode": "offline; Gemini and Tavily requests are mocked",
        "total": result.testsRun,
        "passed": len(result.successes) if hasattr(result, "successes") else sum(
            case["status"] == "passed" for case in result.cases
        ),
        "failed": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "cases": result.cases,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
