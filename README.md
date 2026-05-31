# scope-classifier

유튜브 영상 제목·설명 기반 **비교 영상 분류 데이터셋 + 분류기**.

[모아봄](https://github.com/moabom-official/Moabom_Prototype) 프로젝트의 영상 선정 단계에서 "여러 제품을 동시에 다루는 비교 영상"이 선정되는 문제를 해결하기 위해 만들어졌다. KLUE-BERT distillation 패턴으로, LLM 으로 weak label 데이터셋을 만든 뒤 소형 분류 모델을 학습시켜 운영에 통합한다.

## 무엇이 들어 있나

- **데이터셋 수집 파이프라인** (Codex CLI 활용 weak label)
- **데이터셋 `data/labels/v1.jsonl`** (Binary: `is_comparison ∈ {0, 1}`)
- (후속) KLUE-BERT fine-tune + ONNX export + 운영 통합 가이드

## 분류 정의

| 라벨 | 의미 | 예시 제목 |
|---|---|---|
| `is_comparison=1` | 2개 이상 제품을 동시에 다루며 비교/대결하는 영상 | "iPhone 12 vs 13 비교", "AirPods Pro vs Galaxy Buds" |
| `is_comparison=0` | 그 외 (단일 제품 리뷰·언박싱·뉴스·랭킹·기타) | "iPhone 15 Pro 한 달 사용기", "갤럭시 S24 언박싱" |

> v1 은 binary. roundup·news·unboxing 등 다른 노이즈 카테고리는 v2 dataset 에서 multi-class 확장 후보.

## 사용 흐름 (홈 데스크탑)

```bash
ssh desktop
cd ~/projects/scope-classifier
source .venv/bin/activate

# 1. 라벨링 대상 풀 (운영 PG + YouTube raw fetch)
python -m scope_dataset.collect_videos \
  --source operations_db,youtube_search \
  --target 3000 \
  --out data/candidates.jsonl

# 2. mini sample 100건 라벨링 (codex effort 검증)
python -m scope_dataset.label_with_codex \
  --in data/candidates.jsonl --limit 100 \
  --out data/labels/sample.jsonl \
  --model gpt-5.5 --effort medium --batch 30

# 3. spot check (30개 무작위 표본 + sanity check)
python -m scope_dataset.inspect_labels --in data/labels/sample.jsonl

# 4. 정확도 ≥90% 면 본격 3000건 라벨링
python -m scope_dataset.label_with_codex \
  --in data/candidates.jsonl \
  --out data/labels/v1.jsonl \
  --model gpt-5.5 --effort medium --batch 30

# 5. (옵션) HuggingFace Hub 업로드
python -m scope_dataset.upload_to_hf \
  --in data/labels/v1.jsonl \
  --repo moabom-official/scope-classifier-v1
```

## Codex effort 가이드

| effort | 권장 시점 |
|---|---|
| `medium` (시작) | 첫 100건 sample 라벨링. binary 분류엔 충분 |
| `high` | medium spot check 정확도 < 90% 일 때 escalate |
| `xhigh` | 권장 X — 단순 분류엔 overkill |

## 환경 변수 (`.env`)

`.env.example` 복사 후 운영 PG 자격 + (옵션) YouTube/HF 토큰 입력:

```bash
cp .env.example .env
# .env 직접 편집
```

## 의존성

```bash
uv pip install -e ".[dev,hf]"
```

또는:
```bash
pip install -e ".[dev,hf]"
```

## 테스트

```bash
pytest tests/ -v
```

오프라인 (네트워크·DB·LLM 무사용) 으로 동작.

## License

- **Code**: Apache 2.0
- **Dataset (`data/labels/v1.jsonl`)**: Apache 2.0 with attribution
- 라벨은 GPT-5.5 weak label. 원본 영상 메타데이터 (`title`, `description`) 는 YouTube public 자료.

## 후속 작업 (Roadmap)

1. KLUE-BERT fine-tune + 평가 (`train.py`, `eval.py`)
2. ONNX 변환 + 운영 통합 (모아봄 main repo)
3. roundup·news·unboxing 분류 확장 (v2 dataset)
