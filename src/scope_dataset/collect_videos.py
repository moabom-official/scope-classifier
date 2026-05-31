"""Phase 1 — 라벨링 대상 영상 풀 수집.

Source 1: 모아봄 운영 Azure PG `videos` (실제 운영 traffic 의 영상)
Source 2: YouTube Data API v3 raw search (옵션 — 비교 영상 비율 보강)

출력: `data/candidates.jsonl` ({video_id, title, description, source})
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

from scope_dataset.writer import append_records, iter_records, load_existing_video_ids

YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
YOUTUBE_VIDEO_URL = "https://www.googleapis.com/youtube/v3/videos"


def collect_from_pg(limit: int | None = None) -> list[dict]:
    """모아봄 운영 PG `videos` 테이블에서 라벨링 대상 fetch.

    join 안 함 — `videos` 만으로 충분 (모든 영상은 제목·설명 보유).
    `video_selection_scores` join 옵션은 PR 단순화 위해 일단 제외.
    """
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
    except ImportError as e:
        raise RuntimeError("psycopg2 가 설치돼야 합니다: pip install psycopg2-binary") from e

    host = os.getenv("PG_HOST")
    user = os.getenv("PG_USER")
    pwd = os.getenv("PG_PWD")
    db = os.getenv("PG_DB", "techdb")
    port = int(os.getenv("PG_PORT", "5432"))
    if not all([host, user, pwd]):
        raise RuntimeError(".env 의 PG_HOST/USER/PWD 가 설정돼야 합니다")

    conn = psycopg2.connect(
        host=host, user=user, password=pwd, dbname=db, port=port, sslmode="require"
    )
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        sql = (
            "SELECT video_id, title, COALESCE(description, '') AS description "
            "FROM videos WHERE title IS NOT NULL AND title <> '' "
        )
        if limit:
            sql += f"LIMIT {int(limit)}"
        cur.execute(sql)
        rows = cur.fetchall()
        return [
            {
                "video_id": r["video_id"],
                "title": r["title"],
                "description": r["description"] or "",
                "source": "operations_db",
            }
            for r in rows
        ]
    finally:
        conn.close()


def collect_from_youtube_search(
    queries: list[str],
    *,
    per_query: int = 20,
    api_key: str | None = None,
) -> list[dict]:
    """YouTube Data API v3 검색 — 검색어별 per_query 개 fetch.

    검색어 예: "iPhone 12 vs 13", "갤럭시 S24 비교"
    """
    api_key = api_key or os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise RuntimeError("YOUTUBE_API_KEY 가 .env 에 설정돼야 합니다")

    collected: list[dict] = []
    for q in queries:
        try:
            search_resp = requests.get(
                YOUTUBE_SEARCH_URL,
                params={
                    "part": "snippet",
                    "type": "video",
                    "maxResults": min(per_query, 50),
                    "q": q,
                    "key": api_key,
                },
                timeout=15,
            )
            search_resp.raise_for_status()
            items = search_resp.json().get("items", [])
            for it in items:
                vid = (it.get("id") or {}).get("videoId")
                snip = it.get("snippet") or {}
                if not vid:
                    continue
                collected.append(
                    {
                        "video_id": vid,
                        "title": snip.get("title", ""),
                        "description": snip.get("description", ""),
                        "source": f"youtube_search:{q}",
                    }
                )
            time.sleep(0.5)  # rate limit 완화
        except requests.RequestException as e:
            print(f"[WARN] '{q}' 검색 실패: {e}", file=sys.stderr)
            continue

    return collected


def dedupe_by_video_id(records: list[dict]) -> list[dict]:
    """video_id 중복 제거 (첫 발견을 유지)."""
    seen: set[str] = set()
    out: list[dict] = []
    for r in records:
        vid = r.get("video_id")
        if not vid or vid in seen:
            continue
        seen.add(vid)
        out.append(r)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="라벨링 대상 영상 풀 수집")
    p.add_argument(
        "--source",
        default="operations_db",
        help="콤마 구분: operations_db, youtube_search. 예: 'operations_db,youtube_search'",
    )
    p.add_argument("--target", type=int, default=3000, help="목표 영상 수")
    p.add_argument("--out", default="data/candidates.jsonl", help="출력 jsonl 경로")
    p.add_argument(
        "--queries",
        default=(
            # === 비교 영상 후보 (label=1 추정) ===
            "iPhone 15 vs iPhone 14 비교,"
            "iPhone 16 vs 15 비교,"
            "갤럭시 S24 vs S23 비교,"
            "갤럭시 S25 vs S24 비교,"
            "Pixel 9 vs iPhone 15,"
            "AirPods Pro 2 vs Galaxy Buds,"
            "MacBook Pro vs Air 비교,"
            "M3 vs M2 비교,"
            "iPad Pro vs Air 비교,"
            "갤럭시 Z Flip vs Fold,"
            "맥북 윈도우 노트북 비교,"
            # === 추천/랭킹 (label=1 추정) ===
            "2024 노트북 추천 TOP 5,"
            "2024 스마트폰 추천,"
            "무선이어폰 추천 베스트,"
            "태블릿 추천 순위,"
            # === 단일 제품 리뷰 (label=0 추정) ===
            "iPhone 15 Pro 한 달 사용기,"
            "iPhone 15 Pro Max 리뷰,"
            "갤럭시 S24 Ultra 솔직 리뷰,"
            "갤럭시 S24 한 달 사용 후기,"
            "MacBook Air M3 리뷰,"
            "MacBook Pro M3 사용기,"
            "AirPods Pro 2 장기 사용 리뷰,"
            "iPad Pro 사용 후기,"
            "Galaxy Tab 리뷰,"
            "Pixel 8 Pro 솔직 리뷰,"
            # === 언박싱 (label=0 추정) ===
            "iPhone 15 Pro 언박싱,"
            "갤럭시 S24 개봉기,"
            # === 뉴스 (label=0) ===
            "iPhone 16 출시 예정,"
            "갤럭시 S25 루머,"
            # === v1.1 label=0 보강 (다양한 단일 제품 리뷰) ===
            "갤럭시 워치 6 리뷰,"
            "LG 그램 17 리뷰,"
            "닌텐도 스위치2 한 달 사용,"
            "PS5 슬림 솔직 리뷰,"
            "아이패드 미니 7 리뷰,"
            "갤럭시 Z 플립6 솔직 후기,"
            "갤럭시 Z 폴드6 솔직 후기,"
            "갤럭시 버즈 3 Pro 리뷰,"
            "AirPods 4 리뷰,"
            "샤오미 14 솔직 리뷰,"
            "Steam Deck 한 달 사용,"
            "Meta Quest 3 사용기,"
            "ASUS ROG 노트북 리뷰,"
            "삼성 갤럭시북 리뷰,"
            "델 XPS 솔직 후기,"
            "워치6 클래식 한 달 후기,"
            "Apple Watch 9 리뷰,"
            "키보드 리뷰 솔직 후기,"
            "게이밍 모니터 솔직 리뷰,"
            "기계식 키보드 한 달 후기"
        ),
        help="youtube_search source 검색어 (콤마 구분, 기본값 50개)",
    )
    p.add_argument("--per-query", type=int, default=50, help="검색어당 영상 수 (max 50)")
    p.add_argument(
        "--env-file", default=".env", help=".env 위치 (기본 cwd)"
    )
    args = p.parse_args(argv)

    if Path(args.env_file).exists():
        load_dotenv(args.env_file)

    out_path = Path(args.out)
    already_ids = {r["video_id"] for r in iter_records(out_path) if r.get("video_id")}

    sources = [s.strip() for s in args.source.split(",") if s.strip()]
    aggregated: list[dict] = []

    if "operations_db" in sources:
        print(f"[collect] operations_db 에서 fetch 중 ...")
        try:
            pg_rows = collect_from_pg()
            print(f"[collect] operations_db: {len(pg_rows)} 행")
            aggregated.extend(pg_rows)
        except Exception as e:
            print(f"[ERROR] PG fetch 실패: {e}", file=sys.stderr)

    if "youtube_search" in sources:
        queries = [q.strip() for q in args.queries.split(",") if q.strip()]
        print(f"[collect] YouTube search ({len(queries)} 검색어) ...")
        try:
            yt_rows = collect_from_youtube_search(queries, per_query=args.per_query)
            print(f"[collect] youtube_search: {len(yt_rows)} 행")
            aggregated.extend(yt_rows)
        except Exception as e:
            print(f"[ERROR] YouTube fetch 실패: {e}", file=sys.stderr)

    # dedupe + 기존 후보 풀에 없는 것만
    deduped = dedupe_by_video_id(aggregated)
    new_only = [r for r in deduped if r["video_id"] not in already_ids]

    if args.target and len(new_only) > args.target:
        new_only = new_only[: args.target]

    n = append_records(out_path, new_only)
    total_after = len(already_ids) + n
    print(f"[collect] new={n}, total={total_after}, out={out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
