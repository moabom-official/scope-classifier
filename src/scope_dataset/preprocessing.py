"""학습·추론 공통 전처리.

핵심 원칙: **`normalize_text` 함수는 train/inference 양쪽에서 동일하게 호출**.
운영 통합 시 모아봄 main repo 의 추론 노드도 이 모듈을 import 한다.

흐름:
1. `normalize_text(title, description)` — URL/HTML/공백 정규화 (train·inference 공통)
2. `detect_language(text)` — ko/en/ja/zh/ru/other
3. `filter_dataset(rows, ...)` — 언어 필터 + (옵션) 영어 label=1 truncate 로 sub-population 균형
4. `stratified_split(rows, ratios, seed)` — label 비율 유지 split

참조: [[project-scope-classifier]] / [[reference-azure-pg-backup]]
"""
from __future__ import annotations

import html
import random
import re
from collections import defaultdict
from typing import Iterable, Literal

# === 정규식 — 모듈 import 시 1회 컴파일 ===
_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
_WHITESPACE_PATTERN = re.compile(r"\s+")
_KOREAN_PATTERN = re.compile(r"[가-힣]")
_JAPANESE_PATTERN = re.compile(r"[぀-ゟ゠-ヿ]")
_CHINESE_PATTERN = re.compile(r"[一-鿿]")
_RUSSIAN_PATTERN = re.compile(r"[а-яА-Я]")

Language = Literal["ko", "en", "ja", "zh", "ru", "other"]


# === train/inference 공통 ===

def normalize_text(text: str) -> str:
    """URL 제거 + HTML entity decode + 공백 정규화.

    **이 함수는 학습 데이터와 운영 추론 입력 양쪽에 동일하게 적용된다.**
    train-serving skew 방지 위해 다른 곳에서 별도 전처리 추가 X.

    Idempotent: 두 번 적용해도 결과 동일.
    """
    if not isinstance(text, str):
        return ""
    out = _URL_PATTERN.sub(" ", text)
    out = html.unescape(out)
    out = _WHITESPACE_PATTERN.sub(" ", out).strip()
    return out


def build_input_text(title: str, description: str = "", *, sep: str = " [SEP] ") -> str:
    """모델 input 문자열 구성. title + sep + description (전처리 적용).

    description 은 없으면 title 만. 둘 다 같은 normalize_text 거침.
    `sep` 는 KLUE-BERT WordPiece 토크나이저가 그대로 보존하는 단순 separator.
    """
    t = normalize_text(title)
    d = normalize_text(description)
    if d:
        return f"{t}{sep}{d}"
    return t


def detect_language(text: str) -> Language:
    """언어 추정 — 한국어/일본어/중국어/러시아어/영어 등 (가장 우세한 신호 1개).

    우선순위: ko > ja > zh > ru > en/other.
    한자만 있는 일본어는 zh 로 오탐 가능하지만 우리 task 에선 무관 (둘 다 제거 예정).
    """
    if not isinstance(text, str) or not text:
        return "other"
    if _KOREAN_PATTERN.search(text):
        return "ko"
    if _JAPANESE_PATTERN.search(text):
        return "ja"
    if _CHINESE_PATTERN.search(text):
        return "zh"
    if _RUSSIAN_PATTERN.search(text):
        return "ru"
    return "en"


# === 데이터셋 가공 ===

def filter_dataset(
    rows: list[dict],
    *,
    keep_languages: Iterable[Language] = ("ko", "en"),
    balance_english_label1: bool = True,
    seed: int = 42,
) -> list[dict]:
    """언어 필터 + (옵션) 영어 label=1 sub-population 균형.

    - `keep_languages`: 통과할 언어 (기본 ko + en)
    - `balance_english_label1`: True 면 영어 label=1 을 영어 label=0 수에 맞춰 random truncate
      → 영어 sub-population 도 1:1, shortcut learning(언어→label) 방지
    - 한국어는 자연 분포 그대로 (이미 약 1:1)
    """
    keep_set = set(keep_languages)
    rnd = random.Random(seed)

    by_bucket: dict[tuple[Language, int], list[dict]] = defaultdict(list)
    dropped = 0
    for r in rows:
        title = r.get("title", "") or ""
        desc = r.get("description", "") or ""
        lang = detect_language(title + " " + desc)
        if lang not in keep_set:
            dropped += 1
            continue
        label = r.get("label")
        if label not in (0, 1):
            dropped += 1
            continue
        by_bucket[(lang, label)].append(r)

    if balance_english_label1 and "en" in keep_set:
        en1 = by_bucket.get(("en", 1), [])
        en0_count = len(by_bucket.get(("en", 0), []))
        if len(en1) > en0_count:
            rnd.shuffle(en1)
            by_bucket[("en", 1)] = en1[:en0_count]

    out: list[dict] = []
    for bucket in by_bucket.values():
        out.extend(bucket)
    return out


def stratified_split(
    rows: list[dict],
    *,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """label 비율 유지 train/val/test split.

    각 label 그룹을 독립적으로 shuffle 후 ratio 만큼 분배 → label 분포 보존.
    no overlap (`video_id` 중복 X — 단 입력에 중복 있으면 그건 그대로).
    """
    train_r, val_r, test_r = ratios
    if abs(train_r + val_r + test_r - 1.0) > 1e-6:
        raise ValueError(f"ratios 합이 1 이어야 합니다: {ratios}")

    rnd = random.Random(seed)
    by_label: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_label[r.get("label")].append(r)

    train_out: list[dict] = []
    val_out: list[dict] = []
    test_out: list[dict] = []
    for label, group in by_label.items():
        rnd.shuffle(group)
        n = len(group)
        n_train = int(n * train_r)
        n_val = int(n * val_r)
        train_out.extend(group[:n_train])
        val_out.extend(group[n_train : n_train + n_val])
        test_out.extend(group[n_train + n_val :])

    # 각 split 내에서도 shuffle (label 묶임 방지)
    rnd.shuffle(train_out)
    rnd.shuffle(val_out)
    rnd.shuffle(test_out)
    return train_out, val_out, test_out


def stratified_split_dict(
    rows: list[dict],
    *,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> dict[str, list[dict]]:
    """편의: stratified_split 의 dict 형태."""
    train, val, test = stratified_split(rows, ratios=ratios, seed=seed)
    return {"train": train, "val": val, "test": test}
