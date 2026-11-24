#!/usr/bin/env python3
"""Report on a VIctory performance capture (VICTORY_PROFILE=<file.jsonl>).

Usage: python tools/profile_report.py <file.jsonl> [--top N] [--timeline]

Aggregates the JSONL events the host writes during a run:
  sub  - one submission (id, t, ms, lists, pipelines)
  list - one command list of a submission (id, i, t, ms, cmds, pipelines)
  cmd  - one GPU-timed command (id, i, t, ms, op)

Prints: the slowest commands, the pipelines that cost the most GPU time
in total and the most at once, and (with --timeline) the submissions over
time so load phases are visible.
"""

import json
import re
import sys
from collections import defaultdict


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    top = 20
    if "--top" in sys.argv:
        i = sys.argv.index("--top")
        top = int(sys.argv[i + 1])
    if not args:
        print(__doc__)
        return 1

    subs = []
    lists = []
    cmds = []
    with open(args[0], encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            k = e.get("k")
            if k == "sub":
                subs.append(e)
            elif k == "list":
                lists.append(e)
            elif k == "cmd":
                cmds.append(e)

    if not subs and not cmds:
        print("no events found")
        return 1

    t0 = min((e.get("t", 0) for e in subs + lists + cmds), default=0)
    t1 = max((e.get("t", 0) for e in subs + lists + cmds), default=0)
    print(f"events: {len(subs)} submissions, {len(lists)} lists, {len(cmds)} timed commands")
    print(f"span: {t1 - t0:.1f} s")
    print()

    # Pipelines by total and peak GPU time (from timed commands).
    total = defaultdict(float)
    peak = defaultdict(float)
    count = defaultdict(int)
    for c in cmds:
        ms = c.get("ms", 0)
        for p in re.findall(r"\[((?:compute|mesh|vertex|pixel)[^\]]*)\]", c.get("op", "")):
            total[p] += ms
            peak[p] = max(peak[p], ms)
            count[p] += 1
    print(f"== top {top} pipelines by total GPU time (timed commands) ==")
    for p, ms in sorted(total.items(), key=lambda kv: -kv[1])[:top]:
        print(f"  {ms:10.1f} ms total  {peak[p]:9.1f} ms peak  {count[p]:6d} cmds  {p}")
    print()

    print(f"== top {top} slowest commands ==")
    for c in sorted(cmds, key=lambda c: -c.get("ms", 0))[:top]:
        op = c.get("op", "")
        op = op.split(" | ")[0][:90]
        print(f"  {c.get('ms', 0):10.1f} ms  t={c.get('t', 0):8.1f}  sub {c.get('id')} list {c.get('i')}  {op}")
    print()

    # Submission timeline: bucket submissions by 10 s, total ms per bucket.
    if "--timeline" in sys.argv and subs:
        buckets = defaultdict(lambda: [0, 0.0, 0])
        for s in subs:
            b = int(s.get("t", 0) // 10) * 10
            buckets[b][0] += 1
            buckets[b][1] += s.get("ms", 0)
            buckets[b][2] = max(buckets[b][2], s.get("ms", 0))
        print("== submissions over time (10 s buckets) ==")
        for b in sorted(buckets):
            n, ms, mx = buckets[b]
            bar = "#" * min(60, int(ms / 1000))
            print(f"  t={b:5d}s  {n:5d} subs  {ms:9.0f} ms  max {mx:8.0f} ms  {bar}")

    # The heaviest submissions.
    print(f"\n== top {top} submissions ==")
    for s in sorted(subs, key=lambda s: -s.get("ms", 0))[:top]:
        pipes = s.get("pipelines", [])
        shown = pipes[:3]
        more = f" (+{len(pipes) - 3})" if len(pipes) > 3 else ""
        print(
            f"  {s.get('ms', 0):10.1f} ms  t={s.get('t', 0):8.1f}  id {s.get('id')}  "
            f"lists {s.get('lists')}  {', '.join(shown)}{more}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
