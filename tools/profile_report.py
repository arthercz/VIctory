#!/usr/bin/env python3
"""Analyse VIctory GPU performance captures.

Create a capture with a diagnostics build and VICTORY_PROFILE:

  $env:VICTORY_PROFILE = "F:\\VIctory-logs\\default.jsonl"
  .\\target\\diag\\victory.exe run <title.exe> --execute

Report one run:

  python tools/profile_report.py F:\\VIctory-logs\\default.jsonl --timeline

Compare an experiment with a baseline:

  python tools/profile_report.py F:\\VIctory-logs\\default.jsonl \
      --compare F:\\VIctory-logs\\loop-1000.jsonl

The input is JSONL written by crates/victory-gpu/src/profile.rs. The parser
ignores incomplete final lines, so captures remain useful after a crash.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

KNOWN_KINDS = ("sub", "list", "cmd")
PIPELINE_IN_COMMAND = re.compile(
    r"\[((?:compute|mesh|vertex|pixel)(?:[^\]\\]|\\.)*)\]",
    re.IGNORECASE,
)


@dataclass
class Capture:
    path: Path
    events: dict[str, list[dict]] = field(
        default_factory=lambda: {kind: [] for kind in KNOWN_KINDS}
    )
    malformed: int = 0
    unknown: int = 0
    lines: int = 0

    def all_events(self) -> Iterable[dict]:
        for kind in KNOWN_KINDS:
            yield from self.events[kind]

    def span_seconds(self) -> float:
        times = [number(event.get("t")) for event in self.all_events()]
        times = [value for value in times if value is not None]
        return max(times, default=0.0) - min(times, default=0.0)


def number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    return None


def milliseconds(events: Iterable[dict]) -> list[float]:
    result = []
    for event in events:
        value = number(event.get("ms"))
        if value is not None and value >= 0:
            result.append(value)
    return result


def load_capture(path: Path) -> Capture:
    capture = Capture(path=path)
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            capture.lines += 1
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                capture.malformed += 1
                continue
            if not isinstance(event, dict):
                capture.unknown += 1
                continue
            kind = event.get("k")
            if kind in capture.events:
                capture.events[kind].append(event)
            else:
                capture.unknown += 1
    return capture


def percentile(values: Iterable[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * min(1.0, max(0.0, fraction))
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def latency_stats(events: Iterable[dict]) -> dict[str, float]:
    values = milliseconds(events)
    total = sum(values)
    return {
        "count": float(len(values)),
        "total": total,
        "mean": total / len(values) if values else 0.0,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values, default=0.0),
    }


def pipeline_exposure(capture: Capture) -> dict[str, tuple[float, float, int]]:
    """Return pipeline -> (list-ms exposure, peak list-ms, list count).

    A list may bind several pipelines, so this is exposure, not exclusive GPU
    attribution. It is still useful for narrowing down pipelines present in
    slow lists.
    """

    total: defaultdict[str, float] = defaultdict(float)
    peak: defaultdict[str, float] = defaultdict(float)
    count: defaultdict[str, int] = defaultdict(int)
    for event in capture.events["list"]:
        ms = number(event.get("ms")) or 0.0
        pipelines = event.get("pipelines", [])
        if not isinstance(pipelines, list):
            continue
        for pipeline in set(p for p in pipelines if isinstance(p, str)):
            total[pipeline] += ms
            peak[pipeline] = max(peak[pipeline], ms)
            count[pipeline] += 1
    return {name: (total[name], peak[name], count[name]) for name in total}


def command_pipeline_time(capture: Capture) -> dict[str, tuple[float, float, int]]:
    total: defaultdict[str, float] = defaultdict(float)
    peak: defaultdict[str, float] = defaultdict(float)
    count: defaultdict[str, int] = defaultdict(int)
    for event in capture.events["cmd"]:
        ms = number(event.get("ms")) or 0.0
        op = event.get("op", "")
        if not isinstance(op, str):
            continue
        for pipeline in set(PIPELINE_IN_COMMAND.findall(op)):
            total[pipeline] += ms
            peak[pipeline] = max(peak[pipeline], ms)
            count[pipeline] += 1
    return {name: (total[name], peak[name], count[name]) for name in total}


def print_latency_table(capture: Capture) -> None:
    print("latency                 count      total       mean        p50        p95        p99        max")
    for kind, label in (("list", "command lists"), ("sub", "submissions"), ("cmd", "timed commands")):
        stats = latency_stats(capture.events[kind])
        print(
            f"  {label:18} {int(stats['count']):7d}"
            f" {stats['total']:10.1f} {stats['mean']:10.2f}"
            f" {stats['p50']:10.2f} {stats['p95']:10.2f}"
            f" {stats['p99']:10.2f} {stats['max']:10.2f} ms"
        )


def short_operation(event: dict, width: int = 100) -> str:
    operation = event.get("op", "")
    if not isinstance(operation, str):
        return ""
    return operation.split(" | ", 1)[0].replace("\n", " ")[:width]


def print_top_events(capture: Capture, kind: str, top: int, title: str) -> None:
    events = sorted(
        capture.events[kind],
        key=lambda event: number(event.get("ms")) or 0.0,
        reverse=True,
    )[:top]
    if not events:
        return
    print(f"\n== {title} ==")
    for event in events:
        ms = number(event.get("ms")) or 0.0
        at = number(event.get("t")) or 0.0
        identity = f"id {event.get('id', '?')} list {event.get('i', '?')}"
        if kind == "cmd":
            detail = short_operation(event)
        else:
            pipelines = event.get("pipelines", [])
            if not isinstance(pipelines, list):
                pipelines = []
            shown = ", ".join(str(item) for item in pipelines[:3])
            more = f" (+{len(pipelines) - 3})" if len(pipelines) > 3 else ""
            detail = shown + more
        print(f"  {ms:10.2f} ms  t={at:9.3f}  {identity:22}  {detail}")


def print_pipelines(capture: Capture, top: int) -> None:
    direct = command_pipeline_time(capture)
    exposure = pipeline_exposure(capture)
    values = direct if direct else exposure
    if not values:
        return
    meaning = "timed-command GPU time" if direct else "slow-list exposure (non-exclusive)"
    print(f"\n== top {top} pipelines by {meaning} ==")
    for name, (total, peak, count) in sorted(
        values.items(), key=lambda item: item[1][0], reverse=True
    )[:top]:
        print(
            f"  {total:10.1f} ms total  {peak:9.2f} ms peak"
            f"  {count:7d} samples  {name}"
        )


def print_timeline(capture: Capture, bucket_seconds: int) -> None:
    if not capture.events["list"]:
        return
    buckets: defaultdict[int, list[float]] = defaultdict(list)
    for event in capture.events["list"]:
        at = number(event.get("t")) or 0.0
        ms = number(event.get("ms")) or 0.0
        bucket = int(at // bucket_seconds) * bucket_seconds
        buckets[bucket].append(ms)
    print(f"\n== command-list timeline ({bucket_seconds} s buckets) ==")
    largest = max((sum(values) for values in buckets.values()), default=1.0)
    for bucket in sorted(buckets):
        values = buckets[bucket]
        total = sum(values)
        bar = "#" * max(1, round(50 * total / largest))
        print(
            f"  t={bucket:6d}s  {len(values):6d} lists"
            f"  {total:10.1f} ms  p95 {percentile(values, 0.95):8.2f} ms"
            f"  max {max(values):8.2f} ms  {bar}"
        )


def print_report(capture: Capture, top: int, timeline: bool, bucket_seconds: int) -> None:
    counts = {kind: len(capture.events[kind]) for kind in KNOWN_KINDS}
    print(f"== {capture.path} ==")
    print(
        f"events: {counts['sub']} submissions, {counts['list']} lists,"
        f" {counts['cmd']} timed commands; span {capture.span_seconds():.1f} s"
    )
    if capture.malformed or capture.unknown:
        print(
            f"ignored: {capture.malformed} malformed/incomplete lines,"
            f" {capture.unknown} unknown events"
        )
    print_latency_table(capture)

    list_ms = sum(milliseconds(capture.events["list"]))
    span = capture.span_seconds()
    if span > 0:
        print(
            f"profiled list time / capture span: {100.0 * list_ms / (span * 1000.0):.1f}%"
            " (wall-time indicator, not GPU utilisation)"
        )

    print_pipelines(capture, top)
    print_top_events(capture, "cmd", top, f"top {top} timed commands")
    print_top_events(capture, "list", top, f"top {top} command lists")
    if timeline:
        print_timeline(capture, bucket_seconds)


def change_percent(baseline: float, candidate: float) -> float | None:
    if baseline == 0:
        return None
    return 100.0 * (candidate - baseline) / baseline


def print_comparison(baseline: Capture, candidate: Capture) -> None:
    print(f"\n== A/B comparison: {baseline.path.name} -> {candidate.path.name} ==")
    print("negative change means the candidate is faster")
    print("metric                       baseline    candidate       change")

    rows: list[tuple[str, float, float]] = []
    for kind, label in (("list", "list"), ("sub", "submission"), ("cmd", "command")):
        left = latency_stats(baseline.events[kind])
        right = latency_stats(candidate.events[kind])
        if left["count"] and right["count"]:
            rows.extend(
                (
                    (f"{label} p50 ms", left["p50"], right["p50"]),
                    (f"{label} p95 ms", left["p95"], right["p95"]),
                    (f"{label} p99 ms", left["p99"], right["p99"]),
                    (f"{label} max ms", left["max"], right["max"]),
                )
            )

    for label, left, right in rows:
        change = change_percent(left, right)
        shown = "       n/a" if change is None else f"{change:+10.1f}%"
        print(f"  {label:25} {left:10.2f} {right:12.2f} {shown}")

    left_lists = milliseconds(baseline.events["list"])
    right_lists = milliseconds(candidate.events["list"])
    if left_lists and right_lists:
        left_p95 = percentile(left_lists, 0.95)
        right_p95 = percentile(right_lists, 0.95)
        ratio = right_p95 / left_p95 if left_p95 else 1.0
        if min(len(left_lists), len(right_lists)) < 20:
            verdict = "too few command-list samples for a reliable verdict"
        elif ratio <= 0.80:
            verdict = "candidate materially improves command-list tail latency"
        elif ratio >= 1.20:
            verdict = "candidate materially regresses command-list tail latency"
        else:
            verdict = "no material command-list p95 change (within +/-20%)"
        print(f"\nverdict: {verdict}")
        print("This measures performance only; inspect the rendered output for correctness.")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("capture", type=Path, help="baseline or single JSONL capture")
    result.add_argument("--compare", type=Path, help="candidate JSONL capture")
    result.add_argument("--top", type=int, default=20, help="number of hotspots to show")
    result.add_argument("--timeline", action="store_true", help="show list time over the run")
    result.add_argument(
        "--bucket",
        type=int,
        default=10,
        help="timeline bucket size in seconds (default: 10)",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.top < 1:
        parser().error("--top must be at least 1")
    if args.bucket < 1:
        parser().error("--bucket must be at least 1")
    try:
        baseline = load_capture(args.capture)
        candidate = load_capture(args.compare) if args.compare else None
    except OSError as error:
        print(f"profile_report: {error}", file=sys.stderr)
        return 2

    if not any(baseline.events.values()):
        print(f"profile_report: no recognised events in {baseline.path}", file=sys.stderr)
        return 1
    if candidate is not None and not any(candidate.events.values()):
        print(f"profile_report: no recognised events in {candidate.path}", file=sys.stderr)
        return 1

    print_report(baseline, args.top, args.timeline, args.bucket)
    if candidate is not None:
        print_report(candidate, args.top, args.timeline, args.bucket)
        print_comparison(baseline, candidate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
