"""JSONL writer — idempotent + 재시작 안전.

라벨링 결과는 append. video_id 중복 시 skip. crash 후 재실행해도 같은 결과.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator


def load_existing_video_ids(path: Path) -> set[str]:
    """이미 적재된 jsonl 의 video_id 집합. 없으면 빈 set."""
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # 손상된 행은 skip — 다음 적재 시 video_id 가 다시 처리됨
                continue
            vid = row.get("video_id")
            if isinstance(vid, str):
                ids.add(vid)
    return ids


def append_records(path: Path, records: Iterable[dict]) -> int:
    """jsonl 에 record append. 디렉터리 없으면 생성. 적재 건수 반환."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    return count


def iter_records(path: Path) -> Iterator[dict]:
    """jsonl 라인 단위 dict 생성기. 손상된 행 skip."""
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def now_iso() -> str:
    """`labeled_at` 등 timestamp 표준 — UTC ISO 8601."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
