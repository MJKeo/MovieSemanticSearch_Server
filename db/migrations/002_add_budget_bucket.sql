-- Migration 002: Add budget_bucket column to public.movie_card
--
-- budget_bucket stores the era-adjusted budget classification for a movie.
-- Possible values: 'small', 'large', or NULL (mid-range or no budget data).
-- Re-ingestion is required to populate existing rows; NULL is a safe default.

ALTER TABLE public.movie_card
    ADD COLUMN IF NOT EXISTS budget_bucket TEXT;
