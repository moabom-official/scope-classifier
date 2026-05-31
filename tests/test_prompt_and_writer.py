"""prompt + writer 단위 테스트. 네트워크·DB·LLM 미사용."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scope_dataset.prompt import (
    LabeledVideo,
    ParseError,
    VideoForLabeling,
    build_full_prompt,
    build_user_prompt,
    filter_already_labeled,
    parse_codex_response,
)
from scope_dataset.writer import (
    append_records,
    iter_records,
    load_existing_video_ids,
    now_iso,
)


# ---------- prompt ----------

def test_build_user_prompt_deterministic() -> None:
    videos = [
        VideoForLabeling("abc", "iPhone 12 vs 13 비교"),
        VideoForLabeling("def", "갤럭시 S24 한 달 리뷰"),
    ]
    p1 = build_user_prompt(videos)
    p2 = build_user_prompt(videos)
    assert p1 == p2
    assert "abc" in p1 and "def" in p1


def test_build_user_prompt_empty_raises() -> None:
    with pytest.raises(ValueError):
        build_user_prompt([])


def test_description_truncated() -> None:
    long_desc = "가" * 2000
    videos = [VideoForLabeling("x", "title", description=long_desc)]
    prompt = build_user_prompt(videos)
    # 800자 cap — 2000 짜리가 그대로 박히지는 않음
    assert prompt.count("가") <= 850  # 약간의 여유


def test_full_prompt_includes_system_and_user() -> None:
    videos = [VideoForLabeling("a", "t")]
    full = build_full_prompt(videos)
    assert "is_comparison" in full  # system 정의
    assert "영상:" in full           # user header


# ---------- parser ----------

def test_parse_valid_response() -> None:
    raw = json.dumps([
        {"video_id": "a", "is_comparison": 1, "rationale": "비교"},
        {"video_id": "b", "is_comparison": 0, "rationale": "단일"},
    ])
    out = parse_codex_response(raw)
    assert len(out) == 2
    assert out[0] == LabeledVideo("a", 1, "비교")
    assert out[1] == LabeledVideo("b", 0, "단일")


def test_parse_codex_with_code_fence() -> None:
    raw = '```json\n[{"video_id": "a", "is_comparison": 1, "rationale": "v"}]\n```'
    out = parse_codex_response(raw)
    assert out == [LabeledVideo("a", 1, "v")]


def test_parse_with_surrounding_text() -> None:
    raw = '결과는 다음과 같습니다:\n[{"video_id": "a", "is_comparison": 0, "rationale": "단"}]\n끝.'
    out = parse_codex_response(raw)
    assert out == [LabeledVideo("a", 0, "단")]


def test_parse_empty_response_raises() -> None:
    with pytest.raises(ParseError):
        parse_codex_response("")


def test_parse_invalid_json_raises() -> None:
    with pytest.raises(ParseError):
        parse_codex_response("[{not valid json}]")


def test_parse_non_array_top_raises() -> None:
    with pytest.raises(ParseError):
        parse_codex_response('{"video_id": "a", "is_comparison": 1}')


def test_parse_missing_video_id_raises() -> None:
    with pytest.raises(ParseError):
        parse_codex_response('[{"is_comparison": 1, "rationale": "x"}]')


def test_parse_out_of_range_label_raises() -> None:
    with pytest.raises(ParseError):
        parse_codex_response('[{"video_id": "a", "is_comparison": 2, "rationale": "x"}]')


def test_parse_string_label_coerced() -> None:
    raw = '[{"video_id": "a", "is_comparison": "1", "rationale": "x"}]'
    out = parse_codex_response(raw)
    assert out[0].label == 1


def test_rationale_capped_at_40() -> None:
    long_rat = "가" * 100
    raw = json.dumps([{"video_id": "a", "is_comparison": 1, "rationale": long_rat}])
    out = parse_codex_response(raw)
    assert len(out[0].rationale) <= 40


# ---------- filter (idempotent) ----------

def test_filter_already_labeled() -> None:
    cands = [
        VideoForLabeling("a", "t1"),
        VideoForLabeling("b", "t2"),
        VideoForLabeling("c", "t3"),
    ]
    out = filter_already_labeled(cands, {"a", "c"})
    assert [v.video_id for v in out] == ["b"]


# ---------- writer ----------

def test_writer_append_and_load(tmp_path: Path) -> None:
    target = tmp_path / "labels.jsonl"
    n = append_records(target, [
        {"video_id": "a", "label": 1, "rationale": "x"},
        {"video_id": "b", "label": 0, "rationale": "y"},
    ])
    assert n == 2
    ids = load_existing_video_ids(target)
    assert ids == {"a", "b"}

    # 두 번째 append (append-only)
    append_records(target, [{"video_id": "c", "label": 0, "rationale": "z"}])
    ids = load_existing_video_ids(target)
    assert ids == {"a", "b", "c"}


def test_writer_creates_dirs(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "dir" / "out.jsonl"
    n = append_records(target, [{"video_id": "a", "label": 1}])
    assert n == 1
    assert target.exists()


def test_load_existing_handles_missing(tmp_path: Path) -> None:
    target = tmp_path / "doesnt_exist.jsonl"
    assert load_existing_video_ids(target) == set()


def test_load_existing_skips_corrupt_line(tmp_path: Path) -> None:
    target = tmp_path / "corrupt.jsonl"
    target.write_text(
        '{"video_id": "a", "label": 1}\n'
        'NOT JSON\n'
        '{"video_id": "b", "label": 0}\n',
        encoding="utf-8",
    )
    assert load_existing_video_ids(target) == {"a", "b"}


def test_iter_records(tmp_path: Path) -> None:
    target = tmp_path / "labels.jsonl"
    append_records(target, [{"video_id": "a"}, {"video_id": "b"}])
    rows = list(iter_records(target))
    assert [r["video_id"] for r in rows] == ["a", "b"]


def test_now_iso_format() -> None:
    s = now_iso()
    # ISO 8601 UTC — `+00:00` 또는 `Z` 포함, 'T' 구분자
    assert "T" in s
    assert s.endswith("+00:00") or s.endswith("Z")
