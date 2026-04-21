"""
One-shot backfill: stamp LIVE_ACTION onto every movie_card row whose
keyword_ids lacks ANIMATION.

Run once after deploying the new OverallKeyword.LIVE_ACTION enum member
and the updated Movie.keyword_ids() logic. Safe to delete the file after
a successful run.

Pipeline:
  1. Count eligible rows (for progress reporting).
  2. Issue one batched UPDATE per chunk of eligible rows, chunked via
     the GIN-indexed `@>` filter on `keyword_ids` combined with a LIMIT
     + CTE so each statement locks at most BATCH_SIZE rows.

Idempotent by construction: the NOT (keyword_ids @> ARRAY[live_action])
guard filters out already-stamped rows, so re-runs report zero updates.
"""

from __future__ import annotations

import os

import psycopg
from dotenv import load_dotenv

from implementation.classes.overall_keywords import OverallKeyword

load_dotenv()


# Batch size controls how many rows each UPDATE locks. Keeping per-statement
# locking modest avoids long idle_in_transaction windows; the GIN index on
# keyword_ids makes each batch's eligibility filter cheap regardless of total
# movie count.
BATCH_SIZE = 5000


def _connect() -> psycopg.Connection:
    return psycopg.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        dbname=os.getenv("POSTGRES_DB"),
    )


def main() -> None:
    live_action_id = OverallKeyword.LIVE_ACTION.keyword_id
    animation_id = OverallKeyword.ANIMATION.keyword_id

    with _connect() as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            print("Step 1/2: counting eligible rows (no ANIMATION, no LIVE_ACTION)...")
            cur.execute(
                """
                SELECT COUNT(*)
                FROM public.movie_card
                WHERE NOT (keyword_ids @> ARRAY[%s]::int[])
                  AND NOT (keyword_ids @> ARRAY[%s]::int[])
                """,
                (animation_id, live_action_id),
            )
            eligible_total = cur.fetchone()[0]
            print(f"  {eligible_total:,} rows need LIVE_ACTION stamped.")

            if eligible_total == 0:
                print("Nothing to do.")
                return

            print(f"Step 2/2: appending LIVE_ACTION (id={live_action_id}) in batches of {BATCH_SIZE:,}...")
            total_updated = 0
            batch_count = 0
            while True:
                # CTE + LIMIT caps each UPDATE to BATCH_SIZE eligible rows.
                # Since the WHERE clause excludes already-stamped rows, every
                # iteration strictly shrinks the eligible set; the loop
                # terminates when rowcount drops to 0.
                cur.execute(
                    """
                    WITH targets AS (
                        SELECT movie_id
                        FROM public.movie_card
                        WHERE NOT (keyword_ids @> ARRAY[%s]::int[])
                          AND NOT (keyword_ids @> ARRAY[%s]::int[])
                        LIMIT %s
                    )
                    UPDATE public.movie_card mc
                    SET keyword_ids = array_append(mc.keyword_ids, %s)
                    FROM targets
                    WHERE mc.movie_id = targets.movie_id
                    """,
                    (animation_id, live_action_id, BATCH_SIZE, live_action_id),
                )
                updated = cur.rowcount
                if updated == 0:
                    break
                total_updated += updated
                batch_count += 1
                print(
                    f"  [batch {batch_count}] {updated:>5,} updated "
                    f"(running total: {total_updated:,} / {eligible_total:,})"
                )

            print()
            print("Done.")
            print(f"  Batches processed:  {batch_count:,}")
            print(f"  Rows updated:       {total_updated:,}")


if __name__ == "__main__":
    main()
