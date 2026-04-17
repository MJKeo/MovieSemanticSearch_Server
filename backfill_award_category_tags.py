"""
Temporary backfill script for movie_awards.category_tag_ids.

Run once after deploying the new column + GIN index. Safe to delete the
file after a successful run.

What it does:
  1. ALTER TABLE ... ADD COLUMN IF NOT EXISTS category_tag_ids INT[] NOT NULL DEFAULT '{}'
  2. CREATE INDEX IF NOT EXISTS idx_awards_category_tag_ids ... USING GIN (...)
  3. SELECT DISTINCT category FROM movie_awards
  4. For each distinct category, compute tag_ids via tags_for_category()
     and run one UPDATE statement to set category_tag_ids on every row
     with that category.

Why per-distinct-category UPDATE rather than per-row: there are only ~766
distinct category strings across ~36k rows, so the work collapses into
~766 statements instead of 36k. Each UPDATE hits the existing primary-
key-driven scan + b-tree on category, finishes in milliseconds.

Idempotent: re-running just re-applies the same tag arrays. The ALTER
TABLE / CREATE INDEX statements use IF NOT EXISTS, so re-running on a
table that already has the column is a no-op.
"""
from __future__ import annotations

import os

import psycopg
from dotenv import load_dotenv

from schemas.award_category_tags import tags_for_category

load_dotenv()


def _connect() -> psycopg.Connection:
    return psycopg.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        dbname=os.getenv("POSTGRES_DB"),
    )


def main() -> None:
    with _connect() as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            print("Step 1/3: ensure schema is up to date...")
            cur.execute(
                """
                ALTER TABLE public.movie_awards
                  ADD COLUMN IF NOT EXISTS category_tag_ids INT[] NOT NULL DEFAULT '{}'
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_awards_category_tag_ids
                  ON public.movie_awards USING GIN (category_tag_ids gin__int_ops)
                """
            )

            print("Step 2/3: enumerate distinct categories...")
            cur.execute(
                "SELECT category, COUNT(*) FROM public.movie_awards GROUP BY category ORDER BY COUNT(*) DESC"
            )
            rows = cur.fetchall()
            total_rows = sum(c for _, c in rows)
            print(f"  {len(rows):,} distinct category values across {total_rows:,} rows.")

            print("Step 3/3: backfill tag arrays per distinct category...")
            updated_categories = 0
            updated_rows = 0
            empty_tag_categories = 0
            empty_tag_rows = 0
            for idx, (category, row_count) in enumerate(rows, start=1):
                tag_ids = tags_for_category(category)
                # Empty list is a legitimate result for empty/unknown
                # category strings — the column already defaults to '{}'
                # so nothing to do, but we still report it for visibility.
                if not tag_ids:
                    empty_tag_categories += 1
                    empty_tag_rows += row_count
                    if idx % 100 == 0 or idx == len(rows):
                        print(f"  [{idx:>4}/{len(rows)}] processed (empty)")
                    continue

                cur.execute(
                    "UPDATE public.movie_awards SET category_tag_ids = %s WHERE category = %s",
                    (tag_ids, category),
                )
                updated_categories += 1
                updated_rows += row_count
                if idx % 100 == 0 or idx == len(rows):
                    print(f"  [{idx:>4}/{len(rows)}] {updated_rows:>6,} rows backfilled so far")

            print()
            print("Done.")
            print(f"  Categories with tags written:    {updated_categories:,}")
            print(f"  Categories with empty tag list:  {empty_tag_categories:,} (rows: {empty_tag_rows:,})")
            print(f"  Total rows updated:              {updated_rows:,}")


if __name__ == "__main__":
    main()
