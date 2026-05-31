"""scope-classifier: 비교 영상 분류 데이터셋 수집 + 분류기.

모듈:
- `collect_videos`     라벨링 대상 영상 풀 수집 (운영 PG / YouTube API)
- `label_with_codex`   Codex CLI 로 weak label 부여 + idempotent jsonl writer
- `inspect_labels`     라벨 분포·spot check
- `upload_to_hf`       HuggingFace Hub dataset 업로드 (옵션)
- `prompt`             Codex 호출용 prompt 템플릿
"""

__version__ = "0.1.0"
