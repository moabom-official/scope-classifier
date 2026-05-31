"""(옵션) HuggingFace Hub 에 dataset 업로드.

전제: `pip install -e ".[hf]"` + `.env` 의 `HF_TOKEN` 또는 `huggingface-cli login`.

사용자 manual 실행. 자동 호출 X.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="HuggingFace Hub 에 dataset 업로드")
    p.add_argument("--in", dest="input_path", default="data/labels/v1.jsonl")
    p.add_argument(
        "--repo",
        default=os.getenv("HF_REPO_ID", "moabom-official/scope-classifier-v1"),
        help="HuggingFace repo id (user_or_org/dataset_name)",
    )
    p.add_argument("--private", action="store_true", help="비공개 dataset 으로 업로드")
    p.add_argument("--env-file", default=".env")
    args = p.parse_args(argv)

    if Path(args.env_file).exists():
        load_dotenv(args.env_file)

    in_path = Path(args.input_path)
    if not in_path.exists():
        print(f"[ERROR] 입력 없음: {in_path}", file=sys.stderr)
        return 1

    try:
        from datasets import Dataset
    except ImportError:
        print(
            "[ERROR] datasets 미설치. `pip install -e \".[hf]\"` 또는 "
            "`pip install datasets huggingface-hub` 후 재실행.",
            file=sys.stderr,
        )
        return 1

    token = os.getenv("HF_TOKEN")
    if not token:
        print(
            "[WARN] HF_TOKEN 환경변수 없음. `huggingface-cli login` 으로 사전 인증 필요.",
            file=sys.stderr,
        )

    print(f"[hf] {in_path} → {args.repo} 업로드 중 ...")
    ds = Dataset.from_json(str(in_path))
    print(f"[hf] {len(ds)} rows. columns: {ds.column_names}")

    ds.push_to_hub(
        args.repo,
        token=token,
        private=args.private,
        commit_message="dataset v1 upload (Codex weak label)",
    )
    print(f"[hf] 업로드 완료: https://huggingface.co/datasets/{args.repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
