"""
One-off fix: set lineage='joker' on public.movie_franchise_metadata for
Joker (2019), tmdb_id=475557. The rest of the franchise path (entry/token
index, movie_card.lineage_entry_ids) is being rebuilt shortly, so only the
source-of-truth lineage column is written here.
"""

import asyncio

from dotenv import load_dotenv

from db.postgres import pool

load_dotenv()

JOKER_MOVIE_ID = 475557


async def main() -> None:
    await pool.open()
    try:
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT 1 FROM public.movie_card WHERE movie_id = %s",
                    (JOKER_MOVIE_ID,),
                )
                if await cur.fetchone() is None:
                    raise RuntimeError(
                        f"movie_card row missing for movie_id={JOKER_MOVIE_ID}; "
                        "cannot insert franchise metadata (FK constraint)."
                    )

                await cur.execute(
                    """
                    INSERT INTO public.movie_franchise_metadata (movie_id, lineage)
                    VALUES (%s, %s)
                    ON CONFLICT (movie_id) DO UPDATE SET lineage = EXCLUDED.lineage
                    RETURNING lineage
                    """,
                    (JOKER_MOVIE_ID, "joker"),
                )
                row = await cur.fetchone()
            await conn.commit()
        print(f"OK: movie_id={JOKER_MOVIE_ID} lineage={row[0]!r}")
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
