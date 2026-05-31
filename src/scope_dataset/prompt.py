"""Codex 호출용 분류 prompt + 응답 파서.

본 모듈은 결정적·순수 함수만 (네트워크/LLM 호출 X). 테스트 가능.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, Sequence

VALID_LABELS = (0, 1)


@dataclass(frozen=True)
class VideoForLabeling:
    """라벨링 대상 영상 한 건의 입력 형태."""
    video_id: str
    title: str
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "video_id": self.video_id,
            "title": self.title,
            "description": (self.description or "")[:800],  # 설명은 800자까지만
        }


@dataclass(frozen=True)
class LabeledVideo:
    """Codex 응답 1건 — 파싱·검증 완료한 형태."""
    video_id: str
    label: int                  # 0 or 1
    rationale: str              # <= 20자

    def to_record(self, *, source_video: VideoForLabeling, model: str, effort: str, labeled_at: str) -> dict:
        return {
            "video_id": self.video_id,
            "title": source_video.title,
            "description": source_video.description,
            "label": self.label,
            "rationale": self.rationale,
            "model": model,
            "effort": effort,
            "labeled_at": labeled_at,
        }


SYSTEM_INSTRUCTION = """\
당신은 유튜브 테크 리뷰 영상을 분류하는 분석가다.
분류 기준 (binary):
- is_comparison=1: 영상에서 2개 이상의 서로 다른 제품을 동시에 다루며 비교/대결/대비/순위를 매기는 영상
- is_comparison=0: 그 외 — 단일 제품 심층 리뷰, 사용기, 언박싱, 출시 뉴스, 단일 제품 사용 팁 등

규칙:
- 제목과 설명에 명시된 정보만 사용한다. 추측 금지.
- 단일 제품의 '구버전 vs 신버전' 같은 셀프 비교는 비교 영상으로 본다 (제품 A vs 제품 B 와 동일).
- 한 제품의 여러 모델(예: Pro/Air)을 비교하는 경우도 비교 영상으로 본다.
- 단일 제품 + 잠깐 다른 제품 언급(언어적 비유)만 있는 경우는 0.
- 출력은 반드시 JSON 배열 하나. 그 외 문장 금지.
"""

USER_TEMPLATE_HEADER = """\
다음 영상들을 분류해줘. 각 영상마다 {{video_id, is_comparison, rationale}} 형태로 응답.
rationale 은 20자 이내 한국어. 다음 JSON 배열만 출력:

[{"video_id": "...", "is_comparison": 0|1, "rationale": "..."}]

영상:
"""


def build_user_prompt(videos: Sequence[VideoForLabeling]) -> str:
    """배치 영상 list → Codex 에 줄 user prompt 본문."""
    if not videos:
        raise ValueError("videos 가 비어 있습니다")
    items = [v.to_dict() for v in videos]
    return USER_TEMPLATE_HEADER + json.dumps(items, ensure_ascii=False, indent=2)


def build_full_prompt(videos: Sequence[VideoForLabeling]) -> str:
    """Codex exec 가 stdin/argv 로 받을 단일 prompt 문자열 (system + user 합본).

    codex exec 는 system/user 분리 옵션이 없어 합쳐서 한 번에 줌.
    """
    return SYSTEM_INSTRUCTION + "\n\n" + build_user_prompt(videos)


class ParseError(ValueError):
    """LLM 응답 파싱 실패."""


def parse_codex_response(text: str) -> list[LabeledVideo]:
    """Codex stdout (또는 추출된 JSON) → LabeledVideo 리스트.

    Codex 가 markdown code fence 로 감쌌어도 가운데의 JSON 만 뽑는다.
    """
    if not isinstance(text, str) or not text.strip():
        raise ParseError("응답이 비어 있음")

    json_text = _extract_json_array(text)
    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError as e:
        raise ParseError(f"JSON 파싱 실패: {e}") from e

    if not isinstance(payload, list):
        raise ParseError(f"JSON 최상위가 array 가 아님: {type(payload).__name__}")

    results: list[LabeledVideo] = []
    for i, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ParseError(f"item[{i}] 가 object 가 아님")

        vid = item.get("video_id")
        if not isinstance(vid, str) or not vid:
            raise ParseError(f"item[{i}] video_id 누락/형식")

        raw_label = item.get("is_comparison")
        try:
            label = int(raw_label)
        except (TypeError, ValueError) as e:
            raise ParseError(f"item[{i}] is_comparison 정수화 실패: {raw_label!r}") from e
        if label not in VALID_LABELS:
            raise ParseError(f"item[{i}] is_comparison 값 범위 밖: {label}")

        rationale = (item.get("rationale") or "")[:40]  # 안전상 40자 cap

        results.append(LabeledVideo(video_id=vid, label=label, rationale=rationale))

    return results


def _extract_json_array(text: str) -> str:
    """텍스트에서 첫 JSON array `[...]` 를 추출.

    Codex 응답이 종종 ```json ... ``` fence 로 감싸지거나 앞뒤 설명이 붙어 옴.
    """
    # markdown code fence 우선
    fence_start = text.find("```")
    if fence_start != -1:
        after_fence = text[fence_start + 3 :]
        # 첫 줄 뛰어넘기 (json 같은 언어 힌트일 수도)
        nl = after_fence.find("\n")
        if nl != -1:
            after_fence = after_fence[nl + 1 :]
        fence_end = after_fence.find("```")
        if fence_end != -1:
            inner = after_fence[:fence_end].strip()
            # inner 가 array 로 시작하면 그대로
            if inner.lstrip().startswith("["):
                return inner

    # 일반 텍스트에서 첫 '[' 부터 마지막 ']' 까지
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ParseError("JSON array 마커 [ ] 를 찾지 못함")
    return text[start : end + 1]


def filter_already_labeled(
    candidates: Iterable[VideoForLabeling],
    already_labeled_ids: set[str],
) -> list[VideoForLabeling]:
    """idempotent — 이미 라벨링된 video_id 는 제외."""
    return [v for v in candidates if v.video_id not in already_labeled_ids]
