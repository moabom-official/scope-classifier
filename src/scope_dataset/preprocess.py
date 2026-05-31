"""전처리 CLI — v1.jsonl → filter → balance → split → splits/{train,val,test}.jsonl.

사용:
    python -m scope_dataset.preprocess \
      --in data/labels/v1.jsonl \
      --out-dir data/splits/v1 \
      --languages ko,en --balance-english \
      --ratios 0.8,0.1,0.1 --seed 42
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from scope_dataset.preprocessing import (
    Language,
    filter_dataset,
    normalize_text,
    stratified_split_dict,
)
from scope_dataset.writer import append_records, iter_records


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="라벨 데이터 전처리 + split")
    p.add_argument("--in", dest="input_path", default="data/labels/v1.jsonl")
    p.add_argument("--out-dir", default="data/splits/v1")
    p.add_argument("--languages", default="ko,en", help="유지할 언어 콤마 구분")
    p.add_argument(
        "--balance-english",
        action="store_true",
        default=True,
        help="영어 sub-population 도 label 1:1 로 truncate (기본 활성)",
    )
    p.add_argument(
        "--no-balance-english",
        dest="balance_english",
        action="store_false",
    )
    p.add_argument("--ratios", default="0.8,0.1,0.1", help="train,val,test 비율")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--normalize-now",
        action="store_true",
        help="split jsonl 에 normalize_text 미리 적용한 'input_text' 필드 추가 "
             "(학습 시 매번 호출 안 해도 됨). 단 원본 title/description 도 유지.",
    )
    args = p.parse_args(argv)

    in_path = Path(args.input_path)
    if not in_path.exists():
        print(f"[ERROR] 입력 없음: {in_path}", file=sys.stderr)
        return 1

    rows = list(iter_records(in_path))
    print(f"[load] {len(rows)} rows from {in_path}")

    keep_langs: list[Language] = [s.strip() for s in args.languages.split(",") if s.strip()]  # type: ignore[misc]
    filtered = filter_dataset(
        rows,
        keep_languages=keep_langs,
        balance_english_label1=args.balance_english,
        seed=args.seed,
    )
    print(f"[filter] languages={keep_langs}, balance_english={args.balance_english}: "
          f"{len(rows)} → {len(filtered)} (-{len(rows)-len(filtered)})")

    label_dist = Counter(r["label"] for r in filtered)
    print(f"[filter] label 분포: 1={label_dist.get(1, 0)}, 0={label_dist.get(0, 0)} "
          f"(1:0 = {label_dist.get(1, 0)/max(label_dist.get(0, 1), 1):.3f}:1)")

    ratios_t = tuple(float(x) for x in args.ratios.split(","))
    if len(ratios_t) != 3:
        print(f"[ERROR] --ratios 는 train,val,test 3개", file=sys.stderr)
        return 1

    splits = stratified_split_dict(filtered, ratios=ratios_t, seed=args.seed)  # type: ignore[arg-type]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_label_dist: dict[str, dict[str, int]] = {}
    for name, split_rows in splits.items():
        path = out_dir / f"{name}.jsonl"
        # 기존 split 파일 덮어쓰기 (idempotent reproducible)
        path.unlink(missing_ok=True)

        records = []
        for r in split_rows:
            rec = dict(r)
            if args.normalize_now:
                title = r.get("title", "") or ""
                desc = r.get("description", "") or ""
                rec["input_text"] = (normalize_text(title) + " [SEP] " + normalize_text(desc)).strip()
                rec["input_text"] = rec["input_text"].rstrip(" [SEP]")  # desc 빈 경우
            records.append(rec)
        append_records(path, records)

        ld = Counter(r["label"] for r in split_rows)
        split_label_dist[name] = {"1": ld.get(1, 0), "0": ld.get(0, 0)}
        print(f"[split] {name}: {len(split_rows)} → {path}  "
              f"(1={ld.get(1, 0)}, 0={ld.get(0, 0)})")

    # 메타 정보
    full_label = Counter(r["label"] for r in filtered)
    meta = {
        "source_path": str(in_path),
        "source_rows": len(rows),
        "filtered_rows": len(filtered),
        "languages_kept": keep_langs,
        "balance_english": args.balance_english,
        "ratios": list(ratios_t),
        "seed": args.seed,
        "splits": {name: len(s) for name, s in splits.items()},
        "label_distribution": {
            "filtered": {"1": full_label.get(1, 0), "0": full_label.get(0, 0)},
            "per_split": split_label_dist,
        },
    }
    meta_path = out_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[meta] {meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
