# scope-classifier

유튜브 영상 제목·설명 기반 **비교 영상 분류 데이터셋 + 분류기**. [모아봄](https://github.com/moabom-official/Moabom_Prototype) 의 영상 선정 단계에서 "여러 제품을 동시에 다루는 비교 영상"이 선정되는 문제를 해결하기 위해 만들어졌다.

**LLM (Codex GPT-5.5) 으로 weak label 데이터셋 만들기 → 소형 모델 fine-tune (KLUE-RoBERTa) → 운영 통합** 패턴.

## 현 상태

| Phase | 상태 | 산출물 |
|---|---|---|
| 1. 데이터셋 수집 (Codex weak label) | ✅ | `data/labels/v1.jsonl` (1936건 라벨링) |
| 2. 전처리 + split (8:1:1) | ✅ | `data/splits/v1/{train,val,test}.jsonl` (1776건) |
| 3. 3 모델 fine-tune 비교 | ✅ | [runs/README.md](./runs/README.md) — **`klue/roberta-large` 채택** (test acc **89.94%**) |
| 4. ONNX export + 모아봄 운영 통합 | ⏳ | 후속 PR |

## 분류 정의 (v1)

| 라벨 | 의미 | 예시 제목 |
|---|---|---|
| `is_comparison=1` | 2개 이상 제품을 동시에 비교/대결 | "iPhone 12 vs 13 비교", "AirPods Pro vs Galaxy Buds" |
| `is_comparison=0` | 그 외 (단일 제품 리뷰·언박싱·뉴스·랭킹) | "iPhone 15 Pro 한 달 사용기", "갤럭시 S24 언박싱" |

> v1 binary. roundup·news·unboxing 등 다른 노이즈 분리는 v2 dataset 이슈.

## 학습 결과 요약

| 모델 | params | test acc | test F1 macro | 학습 환경 |
|---|---|---|---|---|
| klue/bert-base | 110M | 87.15% | 87.09% | RTX 4060 Ti |
| **klue/roberta-large** | 340M | **89.94%** ⭐ | **89.94%** ⭐ | RTX 4060 Ti |
| deberta-v3-xlarge-korean | 800M | 88.27% | 88.22% | RTX A6000 (Runpod) |

상세 — [**runs/README.md**](./runs/README.md): 학습 명령 / 하이퍼파라미터 / per-epoch 로그 / 학습 곡선 / 알려진 이슈.

## 디렉터리 구조

```
scope-classifier/
├── README.md                          ← 본 문서
├── LICENSE                            ← Apache 2.0
├── pyproject.toml
├── .env.example
├── src/scope_dataset/
│   ├── prompt.py                      ← Codex 분류 prompt + JSON 파서
│   ├── writer.py                      ← idempotent jsonl writer
│   ├── collect_videos.py              ← Phase 1: 라벨링 대상 풀
│   ├── label_with_codex.py            ← Phase 2: Codex CLI 호출 라벨링
│   ├── inspect_labels.py              ← Phase 3: 분포·spot check + CSV export
│   ├── upload_to_hf.py                ← (옵션) HuggingFace Hub 업로드
│   ├── preprocessing.py               ← 전처리 (train·inference 공통)
│   ├── preprocess.py                  ← 전처리 CLI
│   └── train.py                       ← HF Trainer 기반 fine-tune
├── tests/
│   ├── test_prompt_and_writer.py      ← 21 케이스
│   └── test_preprocessing.py          ← 28 케이스
├── data/
│   ├── labels/v1.jsonl                ← v1 라벨 (1936건)
│   └── splits/v1/{train,val,test}.jsonl  ← 전처리·split 결과 (1776건)
└── runs/
    ├── README.md                      ← 학습 결과 보고서
    ├── v1_learning_curves.png         ← 학습 곡선
    └── <model>/                       ← metrics.json / train_summary.json / epoch_metrics.json
```

## Quick Start

### 1. 환경 셋업
```bash
git clone https://github.com/moabom-official/scope-classifier.git
cd scope-classifier
uv venv --python python3.10  # 또는 3.11
source .venv/bin/activate
uv pip install -e ".[dev,hf,train]"
cp .env.example .env  # 필요 시 PG/YouTube/HF 자격 입력
```

### 2. 데이터셋 수집 (Codex weak label)
```bash
# 라벨링 대상 풀
python -m scope_dataset.collect_videos \
  --source operations_db,youtube_search \
  --target 3000 \
  --out data/candidates.jsonl

# Codex 라벨링 (idempotent, 백그라운드 가능)
python -m scope_dataset.label_with_codex \
  --in data/candidates.jsonl \
  --out data/labels/v1.jsonl \
  --model gpt-5.5 --effort medium --batch 30

# 분포·spot check + CSV export (Excel 검수)
python -m scope_dataset.inspect_labels --in data/labels/v1.jsonl
python -m scope_dataset.inspect_labels \
  --in data/labels/v1.jsonl --export-csv ~/Desktop/labels.csv
```

### 3. 전처리 + split
```bash
python -m scope_dataset.preprocess \
  --in data/labels/v1.jsonl \
  --out-dir data/splits/v1 \
  --languages ko,en --balance-english
```

### 4. 학습
```bash
# 예: roberta-large
python -m scope_dataset.train \
  --model klue/roberta-large \
  --data-dir data/splits/v1 \
  --output-dir runs/klue-roberta-large \
  --epochs 5 --batch 16 --eval-batch 32 --lr 1e-5 \
  --max-length 256 --bf16 --seed 42

cat runs/klue-roberta-large/metrics.json
```

다른 모델 / 외부 GPU 흐름 등 상세는 [runs/README.md](./runs/README.md).

### 5. HuggingFace Hub 업로드 (옵션)
```bash
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

## 테스트
```bash
pytest tests/ -v
```
오프라인 (네트워크·DB·LLM 무사용) 통과.

## License

- **Code**: Apache 2.0
- **Dataset** (`data/labels/v1.jsonl`): Apache 2.0 with attribution
- 라벨은 Codex GPT-5.5 weak label. 원본 영상 메타데이터 (`title`, `description`) 는 YouTube public 자료.

## Roadmap

1. **ONNX export + 모아봄 운영 통합 PR** (next)
2. roundup·news·unboxing 분류 확장 (v2 dataset, multi-class)
3. 자막 후검증 노드 (borderline 케이스만)
4. HuggingFace Hub dataset card 정식 등록
