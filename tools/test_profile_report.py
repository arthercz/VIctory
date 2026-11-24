#!/usr/bin/env python3

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import profile_report


class ProfileReportTests(unittest.TestCase):
    def test_percentile_interpolates_and_handles_empty_input(self):
        self.assertEqual(profile_report.percentile([], 0.95), 0.0)
        self.assertEqual(profile_report.percentile([7.0], 0.95), 7.0)
        self.assertEqual(profile_report.percentile([0.0, 10.0], 0.5), 5.0)
        self.assertEqual(profile_report.percentile([0.0, 10.0, 20.0], 0.95), 19.0)

    def test_load_capture_keeps_valid_events_after_bad_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps({"k": "list", "t": 1.0, "ms": 10.0}),
                        "{incomplete",
                        json.dumps({"k": "newer-event", "t": 2.0}),
                        json.dumps({"k": "cmd", "t": 3.0, "ms": 4.0}),
                    ]
                ),
                encoding="utf-8",
            )
            capture = profile_report.load_capture(path)

        self.assertEqual(len(capture.events["list"]), 1)
        self.assertEqual(len(capture.events["cmd"]), 1)
        self.assertEqual(capture.malformed, 1)
        self.assertEqual(capture.unknown, 1)
        self.assertEqual(capture.span_seconds(), 2.0)

    def test_pipeline_exposure_is_nonexclusive_and_deduplicated_per_list(self):
        capture = profile_report.Capture(Path("capture.jsonl"))
        capture.events["list"] = [
            {"ms": 20.0, "pipelines": ["compute:a", "mesh:b", "compute:a"]},
            {"ms": 5.0, "pipelines": ["compute:a"]},
        ]
        exposure = profile_report.pipeline_exposure(capture)
        self.assertEqual(exposure["compute:a"], (25.0, 20.0, 2))
        self.assertEqual(exposure["mesh:b"], (20.0, 20.0, 1))

    def test_command_pipeline_time_prefers_direct_timing(self):
        capture = profile_report.Capture(Path("capture.jsonl"))
        capture.events["cmd"] = [
            {"ms": 3.0, "op": "dispatch [compute:0123] | details"},
            {"ms": 7.0, "op": "dispatch [compute:0123] [mesh:abcd]"},
        ]
        timing = profile_report.command_pipeline_time(capture)
        self.assertEqual(timing["compute:0123"], (10.0, 7.0, 2))
        self.assertEqual(timing["mesh:abcd"], (7.0, 7.0, 1))

    def test_main_reports_and_compares_two_captures(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline = Path(directory) / "baseline.jsonl"
            candidate = Path(directory) / "candidate.jsonl"
            baseline.write_text(
                "\n".join(
                    json.dumps({"k": "list", "t": i, "ms": 100.0})
                    for i in range(20)
                ),
                encoding="utf-8",
            )
            candidate.write_text(
                "\n".join(
                    json.dumps({"k": "list", "t": i, "ms": 50.0})
                    for i in range(20)
                ),
                encoding="utf-8",
            )
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                result = profile_report.main(
                    [str(baseline), "--compare", str(candidate), "--top", "1"]
                )

        self.assertEqual(result, 0)
        self.assertIn("A/B comparison", output.getvalue())
        self.assertIn("materially improves", output.getvalue())


if __name__ == "__main__":
    unittest.main()
