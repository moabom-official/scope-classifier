"""preprocessing 단위 테스트. 오프라인, 결정적."""
from __future__ import annotations

import pytest

from scope_dataset.preprocessing import (
    build_input_text,
    detect_language,
    filter_dataset,
    normalize_text,
    stratified_split,
    stratified_split_dict,
)


# ============ normalize_text ============

def test_normalize_text_url_removal() -> None:
    assert "https" not in normalize_text("아이폰 리뷰 https://youtu.be/abc 추천")
    assert "www" not in normalize_text("Check www.example.com here")


def test_normalize_text_html_entity_decode() -> None:
    assert normalize_text("AT&amp;T &lt;3") == "AT&T <3"


def test_normalize_text_whitespace_normalize() -> None:
    assert normalize_text("a   b\n\nc\t\td") == "a b c d"


def test_normalize_text_strips() -> None:
    assert normalize_text("  hello  ") == "hello"


def test_normalize_text_idempotent() -> None:
    """**inference 안정성 핵심**: 두 번 적용해도 결과 동일."""
    cases = [
        "iPhone 15 vs 14 https://youtu.be/abc 비교",
        "한 달 사용기   여러   공백",
        "AT&amp;T &amp; SK",
        "  trim me  ",
    ]
    for text in cases:
        once = normalize_text(text)
        twice = normalize_text(once)
        assert once == twice, f"idempotency 깨짐: {text!r} → {once!r} → {twice!r}"


def test_normalize_text_handles_non_string() -> None:
    assert normalize_text(None) == ""  # type: ignore[arg-type]
    assert normalize_text(123) == ""   # type: ignore[arg-type]


def test_normalize_text_preserves_emoji() -> None:
    """이모지는 보존 (KLUE tokenizer 가 UNK 처리해도 학습엔 큰 영향 X)."""
    assert "🆚" in normalize_text("iPhone 🆚 Galaxy")


# ============ build_input_text ============

def test_build_input_text_combines() -> None:
    out = build_input_text("Title", "Description")
    assert out == "Title [SEP] Description"


def test_build_input_text_no_description() -> None:
    out = build_input_text("Title only", "")
    assert out == "Title only"


def test_build_input_text_applies_normalize() -> None:
    out = build_input_text("Title https://x.com", "Desc  with  spaces")
    assert "https" not in out
    assert "  " not in out


# ============ detect_language ============

@pytest.mark.parametrize("text,expected", [
    ("아이폰 리뷰", "ko"),
    ("iPhone Review", "en"),
    ("iPhone 15 한 달 사용", "ko"),  # 혼합 — 한국어 우선
    ("Привет мир", "ru"),
    ("こんにちは世界", "ja"),
    ("你好世界", "zh"),
    ("", "other"),
])
def test_detect_language(text: str, expected: str) -> None:
    assert detect_language(text) == expected


# ============ filter_dataset ============

def _row(video_id: str, title: str, label: int, description: str = "") -> dict:
    return {"video_id": video_id, "title": title, "label": label, "description": description}


def test_filter_keeps_korean_and_english() -> None:
    rows = [
        _row("a", "아이폰 리뷰", 0),
        _row("b", "iPhone Review", 1),
        _row("c", "你好", 0),
        _row("d", "Привет", 1),
        _row("e", "こんにちは", 0),
    ]
    out = filter_dataset(rows, keep_languages=("ko", "en"), balance_english_label1=False)
    ids = {r["video_id"] for r in out}
    assert ids == {"a", "b"}


def test_filter_balances_english_label1() -> None:
    """영어 label=1 이 영어 label=0 수까지 truncate."""
    rows = (
        [_row(f"en1_{i}", f"English review {i}", 1) for i in range(10)]
        + [_row(f"en0_{i}", f"English use {i}", 0) for i in range(3)]
        + [_row(f"ko1_{i}", f"한국어 리뷰 {i}", 1) for i in range(5)]
        + [_row(f"ko0_{i}", f"한국어 사용 {i}", 0) for i in range(5)]
    )
    out = filter_dataset(rows, keep_languages=("ko", "en"), balance_english_label1=True, seed=0)
    en_1 = [r for r in out if "English review" in r["title"]]
    en_0 = [r for r in out if "English use" in r["title"]]
    ko_1 = [r for r in out if "한국어 리뷰" in r["title"]]
    ko_0 = [r for r in out if "한국어 사용" in r["title"]]
    assert len(en_1) == len(en_0) == 3  # 영어 1:1
    assert len(ko_1) == 5 and len(ko_0) == 5  # 한국어 그대로


def test_filter_balance_no_op_when_already_balanced() -> None:
    """영어 label=1 ≤ label=0 이면 truncate 안 함."""
    rows = (
        [_row(f"en1_{i}", f"English r {i}", 1) for i in range(3)]
        + [_row(f"en0_{i}", f"English u {i}", 0) for i in range(5)]
    )
    out = filter_dataset(rows, keep_languages=("ko", "en"), balance_english_label1=True)
    assert len([r for r in out if r["label"] == 1]) == 3
    assert len([r for r in out if r["label"] == 0]) == 5


def test_filter_drops_invalid_label() -> None:
    rows = [
        _row("a", "아이폰", 0),
        _row("b", "갤럭시", 99),  # invalid
        {"video_id": "c", "title": "삼성", "label": None},
    ]
    out = filter_dataset(rows, keep_languages=("ko",))
    assert {r["video_id"] for r in out} == {"a"}


def test_filter_seed_reproducible() -> None:
    rows = (
        [_row(f"en1_{i}", f"English r {i}", 1) for i in range(20)]
        + [_row(f"en0_{i}", f"English u {i}", 0) for i in range(5)]
    )
    a = filter_dataset(rows, balance_english_label1=True, seed=42)
    b = filter_dataset(rows, balance_english_label1=True, seed=42)
    assert [r["video_id"] for r in a] == [r["video_id"] for r in b]


# ============ stratified_split ============

def _balanced_rows(n_per_label: int = 100) -> list[dict]:
    rows = [_row(f"a{i}", f"t{i}", 1) for i in range(n_per_label)]
    rows += [_row(f"b{i}", f"t{i}", 0) for i in range(n_per_label)]
    return rows


def test_stratified_split_ratios_8_1_1() -> None:
    rows = _balanced_rows(100)
    train, val, test = stratified_split(rows, ratios=(0.8, 0.1, 0.1), seed=42)
    assert len(train) == 160  # 80% × 200
    assert len(val) == 20
    assert len(test) == 20


def test_stratified_split_preserves_label_balance() -> None:
    rows = _balanced_rows(100)
    train, val, test = stratified_split(rows, seed=42)
    for split in (train, val, test):
        labels = [r["label"] for r in split]
        ratio_1 = sum(labels) / len(labels)
        assert 0.45 <= ratio_1 <= 0.55  # 균형 유지


def test_stratified_split_no_overlap() -> None:
    rows = _balanced_rows(100)
    train, val, test = stratified_split(rows, seed=42)
    ids = lambda L: {r["video_id"] for r in L}
    assert not (ids(train) & ids(val))
    assert not (ids(train) & ids(test))
    assert not (ids(val) & ids(test))
    assert ids(train) | ids(val) | ids(test) == ids(rows)


def test_stratified_split_seed_reproducible() -> None:
    rows = _balanced_rows(50)
    a = stratified_split(rows, seed=7)
    b = stratified_split(rows, seed=7)
    for sa, sb in zip(a, b):
        assert [r["video_id"] for r in sa] == [r["video_id"] for r in sb]


def test_stratified_split_dict_keys() -> None:
    rows = _balanced_rows(10)
    out = stratified_split_dict(rows, seed=0)
    assert set(out) == {"train", "val", "test"}


def test_stratified_split_bad_ratios() -> None:
    with pytest.raises(ValueError):
        stratified_split(_balanced_rows(10), ratios=(0.6, 0.2, 0.1))
