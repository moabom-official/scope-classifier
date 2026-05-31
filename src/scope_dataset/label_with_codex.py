"""Phase 2 — Codex CLI 로 candidates 에 weak label 부여.

`codex exec --model gpt-5.5 --effort medium "<prompt>"` 를 subprocess 로 호출.
ssh 비인터랙티브 환경 대비 `bash -lc` 로 login shell PATH 보장.

재시작 안전: 출력 jsonl 에 이미 라벨링된 video_id 는 skip.
실패 → `errors.jsonl` 분리.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from scope_dataset.prompt import (
    LabeledVideo,
    ParseError,
    VideoForLabeling,
    build_full_prompt,
    filter_already_labeled,
    parse_codex_response,
)
from scope_dataset.writer import append_records, iter_records, load_existing_video_ids, now_iso


@dataclass(frozen=True)
class CodexConfig:
    model: str = "gpt-5.5"
    effort: str = "medium"
    timeout_sec: int = 600


class CodexCallError(RuntimeError):
    """Codex 호출 실패 (CLI 에러·timeout·응답 비어 있음)."""


def call_codex(prompt: str, config: CodexConfig) -> str:
    """Codex exec 1회 호출. stdout 반환.

    `bash -lc` 로 nvm/PATH 안전 + non-interactive 셸에서도 동작.

    `--effort` 는 wrapper-level 옵션이고 `codex exec` 가 직접 지원하지 않으므로
    `-c model_reasoning_effort='medium'` 형태의 config override 로 통일.
    `--model` 은 모든 버전에서 지원.
    """
    # codex exec 는 PROMPT 를 인자로 받음. shell 메타 회피 위해 stdin 으로 넘김.
    cmd = [
        "bash",
        "-lc",
        (
            f"codex exec --model '{config.model}' "
            f"-c model_reasoning_effort='{config.effort}' -"
        ),
    ]

    try:
        proc = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=config.timeout_sec,
        )
    except subprocess.TimeoutExpired as e:
        raise CodexCallError(f"codex timeout after {config.timeout_sec}s") from e
    except FileNotFoundError as e:
        raise CodexCallError(f"codex CLI 를 찾을 수 없음: {e}") from e

    if proc.returncode != 0:
        raise CodexCallError(
            f"codex exit code {proc.returncode}\n"
            f"stderr: {proc.stderr[:500]}"
        )

    out = (proc.stdout or "").strip()
    if not out:
        raise CodexCallError("codex stdout 가 비어 있음")
    return out


def label_batch(
    batch: Sequence[VideoForLabeling],
    config: CodexConfig,
) -> tuple[list[LabeledVideo], str]:
    """배치 1개 → 라벨링 결과 + 원본 stdout (디버그용).

    실패 시 CodexCallError 또는 ParseError 그대로 raise.
    """
    prompt = build_full_prompt(batch)
    raw = call_codex(prompt, config)
    return parse_codex_response(raw), raw


def _chunk(items: list[VideoForLabeling], size: int) -> Iterable[list[VideoForLabeling]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def label_dataset(
    candidates: list[VideoForLabeling],
    *,
    out_path: Path,
    errors_path: Path,
    config: CodexConfig,
    batch_size: int = 30,
    limit: int | None = None,
) -> dict:
    """전체 라벨링 — idempotent, append, errors 분리.

    Returns: {"labeled": N, "errors": N, "skipped_already": N}
    """
    already = load_existing_video_ids(out_path)
    todo = filter_already_labeled(candidates, already)
    if limit is not None:
        todo = todo[:limit]

    skipped = len(candidates) - len(todo) - (len(candidates) - limit if limit else 0) * 0
    skipped_already = len(candidates) - len(filter_already_labeled(candidates, already))

    labeled_total = 0
    error_total = 0

    source_by_id = {v.video_id: v for v in todo}

    for batch_idx, batch in enumerate(_chunk(todo, batch_size), start=1):
        print(
            f"[label] batch {batch_idx}: {len(batch)} 영상 → codex "
            f"(model={config.model}, effort={config.effort})",
            flush=True,
        )
        try:
            results, _raw = label_batch(batch, config)
        except (CodexCallError, ParseError) as e:
            print(f"[label] batch {batch_idx} 실패 ({type(e).__name__}): {e}", file=sys.stderr)
            errors = [
                {"video_id": v.video_id, "title": v.title, "error": str(e)[:300], "batch": batch_idx}
                for v in batch
            ]
            append_records(errors_path, errors)
            error_total += len(batch)
            continue

        labeled_at = now_iso()
        records = []
        result_ids = set()
        for r in results:
            src = source_by_id.get(r.video_id)
            if src is None:
                # Codex 가 모르는 video_id 반환 — skip
                continue
            result_ids.add(r.video_id)
            records.append(
                r.to_record(
                    source_video=src,
                    model=config.model,
                    effort=config.effort,
                    labeled_at=labeled_at,
                )
            )

        if records:
            append_records(out_path, records)
            labeled_total += len(records)

        # 응답에 없는 batch 영상 → errors
        missing = [v for v in batch if v.video_id not in result_ids]
        if missing:
            errs = [
                {"video_id": v.video_id, "title": v.title, "error": "missing_in_response", "batch": batch_idx}
                for v in missing
            ]
            append_records(errors_path, errs)
            error_total += len(missing)

    return {
        "labeled": labeled_total,
        "errors": error_total,
        "skipped_already": skipped_already,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Codex 로 candidates 라벨링")
    p.add_argument("--in", dest="input_path", default="data/candidates.jsonl", help="후보 jsonl")
    p.add_argument("--out", default="data/labels/v1.jsonl", help="라벨 결과 jsonl")
    p.add_argument("--errors", default="data/labels/errors.jsonl", help="실패 영상 jsonl")
    p.add_argument("--model", default="gpt-5.5", help="codex --model")
    p.add_argument(
        "--effort", default="medium",
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        help="codex --effort",
    )
    p.add_argument("--batch", type=int, default=30, help="배치 크기")
    p.add_argument("--limit", type=int, default=None, help="최대 라벨링 영상 수")
    p.add_argument("--timeout", type=int, default=600, help="codex 호출 timeout (초)")
    args = p.parse_args(argv)

    in_path = Path(args.input_path)
    if not in_path.exists():
        print(f"[ERROR] 후보 jsonl 없음: {in_path}", file=sys.stderr)
        return 1

    cands = [
        VideoForLabeling(
            video_id=r["video_id"],
            title=r.get("title", ""),
            description=r.get("description", ""),
        )
        for r in iter_records(in_path)
        if r.get("video_id")
    ]
    print(f"[label] 후보 {len(cands)} 건 로드")

    stats = label_dataset(
        cands,
        out_path=Path(args.out),
        errors_path=Path(args.errors),
        config=CodexConfig(model=args.model, effort=args.effort, timeout_sec=args.timeout),
        batch_size=args.batch,
        limit=args.limit,
    )
    print(f"[label] 완료: {stats}")
    return 0 if stats["labeled"] > 0 or stats["errors"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
