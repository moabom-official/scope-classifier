"""Phase 3 — 라벨 분포·spot check.

분포 + 길이 + 무작위 표본 + sanity check (제목 키워드 vs label 불일치 flag).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

from scope_dataset.writer import iter_records

# 비교 신호 후보 키워드 (recall 아니라 sanity check 용)
_COMPARISON_SIGNALS = [
    re.compile(r"\bvs\.?\b", re.IGNORECASE),
    re.compile(r"\bversus\b", re.IGNORECASE),
    re.compile(r"비교"),
    re.compile(r"대결"),
    re.compile(r"차이점"),
]


def has_comparison_signal(text: str) -> bool:
    """제목에 명백한 비교 키워드가 있나 (sanity check 용)."""
    if not text:
        return False
    return any(p.search(text) for p in _COMPARISON_SIGNALS)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="라벨 분포·spot check")
    p.add_argument("--in", dest="input_path", default="data/labels/v1.jsonl")
    p.add_argument("--sample", type=int, default=30, help="무작위 표본 수")
    p.add_argument("--seed", type=int, default=42, help="표본 추출 seed")
    args = p.parse_args(argv)

    in_path = Path(args.input_path)
    if not in_path.exists():
        print(f"[ERROR] 입력 없음: {in_path}", file=sys.stderr)
        return 1

    rows = list(iter_records(in_path))
    total = len(rows)
    if total == 0:
        print("[ERROR] 빈 데이터셋", file=sys.stderr)
        return 1

    # 분포
    label_count = Counter(r.get("label") for r in rows)
    pos_ratio = label_count.get(1, 0) / total
    title_lengths = [len(r.get("title", "")) for r in rows]
    desc_lengths = [len(r.get("description", "")) for r in rows]

    print(f"\n=== 라벨 분포 (n={total}) ===")
    print(f"  is_comparison=1: {label_count.get(1, 0)} ({pos_ratio:.1%})")
    print(f"  is_comparison=0: {label_count.get(0, 0)} ({1 - pos_ratio:.1%})")
    if label_count.get(None) or any(k not in (0, 1) for k in label_count):
        unknown = sum(v for k, v in label_count.items() if k not in (0, 1))
        print(f"  invalid: {unknown}")

    print(f"\n=== 길이 통계 ===")
    print(f"  title  avg={sum(title_lengths) / total:.0f}, max={max(title_lengths)}")
    print(f"  desc   avg={sum(desc_lengths) / total:.0f}, max={max(desc_lengths)}")

    # Sanity check: 제목에 비교 키워드 있는데 label=0
    suspicious_neg = [
        r for r in rows
        if r.get("label") == 0 and has_comparison_signal(r.get("title", ""))
    ]
    print(f"\n=== Sanity Check ===")
    print(f"  제목에 비교 키워드 있는데 label=0: {len(suspicious_neg)}")
    for r in suspicious_neg[:5]:
        print(f"    - {r.get('video_id')}: {r.get('title', '')[:60]}")
        print(f"      rationale: {r.get('rationale', '')}")

    # 무작위 표본
    rnd = random.Random(args.seed)
    sample = rnd.sample(rows, min(args.sample, total))
    print(f"\n=== 무작위 표본 ({len(sample)}개) — 사용자 spot check 용 ===")
    for i, r in enumerate(sample, 1):
        label = r.get("label")
        title = r.get("title", "")[:70]
        rationale = r.get("rationale", "")
        print(f"  [{i:2d}] label={label} | {title}")
        print(f"        → {rationale}")

    print(f"\n결과: {in_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
