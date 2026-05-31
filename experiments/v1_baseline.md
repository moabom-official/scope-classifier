# v1 Baseline — 비교 영상 분류기 학습 결과

3개 모델 fine-tune 비교 (binary `is_comparison: 0/1`, dataset v1 1776건). **결론: `klue/roberta-large` 채택**.

## 환경

- **데이터셋**: `data/splits/v1/{train,val,test}.jsonl` (전처리 + 8:1:1 stratified split, seed=42)
  - filter: ko + en 만 유지, 영어 sub-population 도 label 1:1 (shortcut learning 방지)
  - 1936 → 1776 (-13 ja/zh/ru, -147 영어 label=1 random truncate)
  - 분포: label=1=906, label=0=870 (1:0 = 1.04:1)
  - splits: train 1420 / val 177 / test 179
- **공통 전처리**: `scope_dataset/preprocessing.py:normalize_text` (URL 제거, HTML decode, 공백 정규화) — **train·inference 양쪽에서 동일 함수**
- **mixed precision**: bf16 (Ada Lovelace, Ampere native)
- **공통 hyperparams**: `--max-length 256 --seed 42 --warmup-ratio 0.1 --weight-decay 0.01`
- **best model 선택**: `load_best_model_at_end=True, metric_for_best_model="f1_macro"`

## 데이터셋 메타

`data/splits/v1/meta.json`:
```json
{
  "source_rows": 1936,
  "filtered_rows": 1776,
  "languages_kept": ["ko", "en"],
  "balance_english": true,
  "ratios": [0.8, 0.1, 0.1],
  "seed": 42,
  "splits": {"train": 1420, "val": 177, "test": 179},
  "label_distribution": {
    "filtered": {"1": 906, "0": 870},
    "per_split": {
      "train": {"1": 724, "0": 696},
      "val":   {"1": 90,  "0": 87},
      "test":  {"1": 92,  "0": 87}
    }
  }
}
```

## 학습 명령 (재현)

### 1. klue/bert-base (110M)

```bash
python -m scope_dataset.train \
  --model klue/bert-base \
  --data-dir data/splits/v1 \
  --output-dir runs/klue-bert-base \
  --epochs 5 --batch 16 --eval-batch 64 --lr 2e-5 \
  --max-length 256 --bf16 --seed 42
```

학습 환경: 데스크탑 RTX 4060 Ti 16GB

### 2. klue/roberta-large (340M)

```bash
python -m scope_dataset.train \
  --model klue/roberta-large \
  --data-dir data/splits/v1 \
  --output-dir runs/klue-roberta-large \
  --epochs 5 --batch 16 --eval-batch 32 --lr 1e-5 \
  --max-length 256 --bf16 --seed 42
```

학습 환경: 데스크탑 RTX 4060 Ti 16GB

### 3. team-lucid/deberta-v3-xlarge-korean (800M)

```bash
python -m scope_dataset.train \
  --model team-lucid/deberta-v3-xlarge-korean \
  --data-dir data/splits/v1 \
  --output-dir runs/deberta-v3-xlarge-korean \
  --epochs 3 --batch 8 --eval-batch 16 --lr 1e-5 \
  --max-length 256 --bf16 --gradient-checkpointing --seed 42
```

학습 환경: Runpod RTX A6000 48GB

## 하이퍼파라미터 비교

| 항목 | bert-base | roberta-large | deberta-v3-xlarge |
|---|---|---|---|
| params | 110M | 340M | 800M |
| epochs | 5 | 5 | 3 |
| batch (train) | 16 | 16 | 8 |
| batch (eval) | 64 | 32 | 16 |
| lr | 2e-5 | 1e-5 | 1e-5 |
| max_length | 256 | 256 | 256 |
| bf16 | ✓ | ✓ | ✓ |
| gradient_checkpointing | ✗ | ✗ | ✓ |
| warmup_ratio | 0.1 | 0.1 | 0.1 |
| weight_decay | 0.01 | 0.01 | 0.01 |
| seed | 42 | 42 | 42 |

## 최종 결과 (test set 평가)

| 모델 | params | test acc | test F1 macro | test AUROC | best epoch | 학습 시간 |
|---|---|---|---|---|---|---|
| klue/bert-base | 110M | 87.15% | 87.09% | 95.55% | 4 | 54.5s (4060 Ti) |
| **klue/roberta-large** | 340M | **89.94%** ⭐ | **89.94%** ⭐ | 95.83% | 1 | 168.8s (4060 Ti) |
| deberta-v3-xlarge-korean | 800M | 88.27% | 88.22% | 95.28% | (best_load) | 283.8s (RTX A6000) |

### Per-class (test)

| 모델 | label=0 P | label=0 R | label=0 F1 | label=1 P | label=1 R | label=1 F1 |
|---|---|---|---|---|---|---|
| klue/bert-base | 0.900 | 0.828 | 0.862 | 0.848 | 0.913 | 0.880 |
| **klue/roberta-large** | **0.888** | **0.908** | 0.898 | **0.911** | **0.891** | 0.901 |
| deberta-v3-xlarge-korean | 0.913 | 0.839 | 0.874 | 0.859 | 0.924 | 0.890 |

## Per-epoch eval (overfit 분석)

상세 — `runs/<model>/epoch_metrics.json`.

### bert-base (epoch 4 가 best — 약한 overfit)

| epoch | eval_loss | eval_acc | eval_f1 | train_loss (approx) |
|---|---|---|---|---|
| 1 | 0.400 | 0.836 | 0.835 | 0.39 |
| 2 | 0.338 | 0.864 | 0.864 | 0.22 |
| 3 | 0.380 ↑ | 0.887 | 0.887 | 0.12 |
| **4** | 0.456 ↑ | **0.893** | **0.893** | 0.04 |
| 5 | 0.489 ↑ | 0.893 | 0.893 | 0.01 |

→ train_loss → 0 이지만 eval_acc 계속 ↑. eval_loss 증가는 약한 overfit 신호. best epoch 4.

### roberta-large (epoch 1 이 best — 명확한 overfit ⚠️)

| epoch | eval_loss | eval_acc | eval_f1 | train_loss |
|---|---|---|---|---|
| **1** | 0.319 | **0.881** | **0.881** | 0.37 |
| 2 | 0.342 ↑ | 0.876 ↓ | 0.876 ↓ | 0.25 |
| 3 | 0.472 ↑↑ | 0.864 ↓ | 0.864 ↓ | 0.07 |
| 4 | 0.581 ↑↑ | 0.870 | 0.870 | 0.06 |
| 5 | 0.648 ↑↑ | 0.876 | 0.876 | 0.02 |

→ **epoch 1 이 best**. 사실상 1 epoch 만 학습 충분. 학습 시간 단축 가능.

### deberta-v3-xlarge-korean

| epoch | eval_loss | eval_acc | eval_f1 | eval_auroc |
|---|---|---|---|---|
| 3 (final) | 0.679 | 0.876 | 0.876 | 0.950 |
| best (load_best) | 0.541 | 0.881 | 0.881 | 0.952 |

→ Runpod 세션 종료로 raw 로그 부분 회수 불가. final/best metrics 만 보존.

## 분석 — roberta-large 가 best 인 이유

1. **데이터 1776건 vs xlarge 800M = overfit 위험 ↑**. 큰 모델일수록 작은 데이터셋에 불리.
2. **AUROC 거의 동등** (95.3~95.8) — 분류 능력 자체는 비슷하지만 calibration 차이.
3. **xlarge 는 label=0 recall 0.84** — roberta-large 0.91 보다 약함. small-data 영역에서 일반적 패턴.
4. **roberta-large 의 epoch 1 best** 는 빠른 수렴을 시사 — pre-train 품질이 우리 task 와 맞음.

→ `klue/roberta-large` 채택, ONNX export + 모아봄 main repo 운영 통합 PR 후보.

## 운영 통합용 weights

weights 는 repo 에 commit X (xlarge 3GB / roberta-large 1.3GB / bert-base 440MB) — GitHub 100MB limit 초과 + LFS 불요.

배포 경로 (다음 PR):
1. roberta-large 의 `runs/klue-roberta-large/best/` checkpoint → ONNX export
2. HuggingFace Hub `moabom-official/scope-classifier-roberta-large-v1` 에 push
3. 모아봄 main repo Dockerfile 에서 download

## 재현 방법

```bash
# 1. clone + venv + 의존성
git clone https://github.com/moabom-official/scope-classifier.git
cd scope-classifier
uv venv --python python3.10  # 또는 python3.11
source .venv/bin/activate
uv pip install -e ".[train]"  # PyTorch + Transformers + accelerate + sklearn

# 2. (필요 시) Transformers downgrade — DeBERTa-v2 호환
uv pip install "transformers>=4.45,<5" "tokenizers<0.21"

# 3. 학습 (위 명령 그대로)

# 4. 결과 확인
cat runs/<model>/metrics.json
cat runs/<model>/epoch_metrics.json
```

## 알려진 이슈 (재발 방지 메모)

1. **HF transformers 5.x + DeBERTa-v2**: tokenizer Unigram vocab dict→Sequence TypeError. → `transformers>=4.45,<5` + `tokenizers<0.21` 사용.
2. **HF transformers 4.45+ Trainer**: `tokenizer=` 인자 제거됨 → `processing_class=` 로 변경.
3. **PyTorch wheel CUDA 호환**: Runpod host driver `12.4` 가 PyPI default `torch>=2.5+cu128` 과 mismatch 가능 → `pip install --index-url https://download.pytorch.org/whl/cu121 torch` 로 명시.
4. **Runpod `/workspace` volume storage**: large checkpoint write 시 inline_container 에러 → `--output-dir /root/runs/...` 로 container disk 사용.
5. **`--save-total-limit`**: 큰 모델 (xlarge) 학습 시 디스크 부담. CLI arg 추가 검토 후속.
