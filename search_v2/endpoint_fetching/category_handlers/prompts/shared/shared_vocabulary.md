# Vocabulary

Definitions of terms referenced throughout this prompt.

## `CategoryCall`

The unit of work. One CategoryCall is one committed retrieval call: a category (already chosen upstream), one or more `expressions` (short database-vocabulary phrases), and a `retrieval_intent` (1–3 sentences framing what the call is searching for). Every CategoryCall you see has been committed by the upstream stage; your job is to translate it into endpoint parameters, not to revisit category routing or polarity.

## `polarity`

Whether the user wants a characteristic (positive) or wants to avoid it (negative). Polarity is committed upstream and stamped onto your output post-hoc by the orchestrator — **you do not see polarity on input and you do not emit it on output.**

Your parameters always describe **what to find**, never what to exclude. The orchestrator decides later whether that presence helps or hurts the user. Surface-negated phrasings have already been rewritten upstream into positive-presence framing, so the expressions and retrieval intent you receive describe what the user wants found regardless of include-vs-exclude direction. Never invert, negate, or "undo" an exclusion in the parameters you emit.
