## 1 — Global Invariants (must hold on EVERY test case)

These assertions should be baked into a shared helper and run after every single test, regardless of what scenario it's targeting.

1. Every movie_id present in the input `candidates` dict has a corresponding entry in `final_scores`. No keys are missing and no extra keys appear.
2. Every value in `final_scores` is >= 0.0 and <= 1.0.
3. `space_contexts` contains exactly 8 entries, one per vector space, in a consistent order.
4. The `normalized_weight` values across all 8 `space_contexts` entries sum to exactly 1.0 (within floating point tolerance, e.g. 1e-9).
5. Every space where `is_active` is False has a `normalized_weight` of exactly 0.0.
6. Every space where `is_active` is True has a `normalized_weight` strictly greater than 0.0.
7. The anchor space's `did_run_original` is always True and `did_run_subquery` is always False, regardless of inputs.
8. The anchor space's `effective_relevance` is always None.
9. `per_space_normalized` contains keys only for active spaces (spaces where `is_active` is True). No inactive space appears as a key.
10. Every value inside every dict in `per_space_normalized` is > 0.0 and <= 1.0. (The sparse representation means 0.0 entries are simply absent.)
11. Within each space's normalized dict in `per_space_normalized`, at least one candidate has a normalized score of exactly 1.0 (the best candidate in that pool). Exception: the dict is empty (no candidates had positive blended scores).

---

## 2 — Empty and Minimal Input Boundaries

12. Empty candidates dict: `final_scores` is an empty dict. `per_space_normalized` is an empty dict. `space_contexts` still contains 8 entries with valid weights.
13. Single candidate with a positive anchor score only: `final_scores` has one entry with value 1.0 (it's the best and only candidate in the only active space, normalized to 1.0, weight is 1.0).
14. Single candidate with positive scores across multiple active spaces: `final_scores` value is exactly 1.0 (best in every space it appears in, each normalized to 1.0, weights sum to 1.0).
15. Single candidate with all zero scores (a lexical-only candidate): `final_scores` value is 0.0. `per_space_normalized` has entries for active spaces but none of them contain this candidate.
16. Two candidates where one has all zeros (lexical-only) and the other has positive scores: the all-zero candidate scores 0.0, the other scores > 0.0.

---

## 3 — Space Execution Flags (Stage 1 correctness)

These tests verify that the `SpaceExecutionContext` fields are set correctly based on relevance and subquery combinations. All assertions target `space_contexts` in the result.

17. Non-anchor space with relevance `NOT_RELEVANT` and subquery `None`: `did_run_original` is False, `did_run_subquery` is False, `effective_relevance` is `NOT_RELEVANT`, `is_active` is False.
18. Non-anchor space with relevance `NOT_RELEVANT` and subquery text present: `did_run_original` is False (search was NOT dispatched because relevance was NOT_RELEVANT at dispatch time), `did_run_subquery` is True, `effective_relevance` is `SMALL` (promoted), `is_active` is True.
19. Non-anchor space with relevance `SMALL` and subquery `None`: `did_run_original` is True, `did_run_subquery` is False, `effective_relevance` is `SMALL` (unchanged), `is_active` is True.
20. Non-anchor space with relevance `SMALL` and subquery text present: `did_run_original` is True, `did_run_subquery` is True, `effective_relevance` is `SMALL`, `is_active` is True.
21. Same as 19 and 20 but with `MEDIUM` relevance. Verify `effective_relevance` is `MEDIUM` in both subquery-present and subquery-absent variants.
22. Same as 19 and 20 but with `LARGE` relevance. Verify `effective_relevance` is `LARGE` in both variants.
23. All seven non-anchor spaces set to `NOT_RELEVANT` with no subqueries: only anchor is active. Anchor `normalized_weight` is 1.0.
24. All seven non-anchor spaces have subquery text but all have relevance `NOT_RELEVANT`: all seven are promoted to `SMALL`, all are active with `did_run_original` False and `did_run_subquery` True.
25. Verify each of the 7 non-anchor spaces independently: for each space, set only that space to `LARGE` with a subquery while all others are `NOT_RELEVANT` with no subquery. Confirm only that space plus anchor are active. (7 sub-tests.)

---

## 4 — Blend Logic (Stage 2 correctness observed through final scores)

Each test below manipulates scores in a specific space while holding everything else constant to isolate blend behavior.

26. **Both-ran blend, candidate in both searches.** Set a space to `MEDIUM` with subquery text. Give a candidate original=0.50 and subquery=0.90 in that space. Give a second candidate original=0.50 and subquery=0.50. Verify the first candidate scores higher overall (blended 0.82 vs 0.50, so first is better after normalization).
27. **Both-ran blend, candidate appears only in subquery (original=0.0).** Two candidates in a blended space: A has subquery=0.80, original=0.0 (blended=0.64). B has subquery=0.80, original=0.80 (blended=0.80). Verify B scores higher than A in the final result. This confirms the 20% penalty for missing the original search.
28. **Both-ran blend, candidate appears only in original (subquery=0.0).** Two candidates in a blended space: A has original=0.80, subquery=0.0 (blended=0.16). B has original=0.40, subquery=0.40 (blended=0.40). Verify B scores higher than A despite A having a better original score. This confirms the heavy 80% penalty for missing the subquery.
29. **Original-only blend (subquery text is None, relevance > NOT_RELEVANT).** Two candidates with different original scores. Verify their final scores reflect 100% of the original score ordering (no dilution from a missing subquery dimension). Also verify the space's context shows `did_run_subquery` is False.
30. **Subquery-only blend (promoted SMALL).** Set relevance `NOT_RELEVANT` with subquery text present. Two candidates with different subquery scores. Verify ordering follows subquery scores. Verify the space's context shows `did_run_original` is False and `effective_relevance` is `SMALL`.
31. **Both-ran, both candidates have identical blended scores.** Two candidates where `0.8 * sub + 0.2 * orig` produces the same value for both (e.g., A: sub=0.5, orig=1.0 → 0.6; B: sub=0.75, orig=0.0 → 0.6). Verify they get identical final scores for that space's contribution.
32. **Blend precision: verify the 80/20 ratio numerically.** Single active non-anchor space, two candidates. Calculate expected blended values by hand, then verify the final scores match the expected ratio after normalization. This catches hardcoded wrong constants.

---

## 5 — Normalization Behavior (Stage 3 correctness)

33. **All candidates in a space have identical blended scores.** All should receive normalized score of 1.0 in that space. Verify via `per_space_normalized`.
34. **Two candidates with very different blended scores.** The better one should get 1.0 and the worse one should get approximately `exp(-3.0)` ≈ 0.0498. Verify numerically.
35. **Tight cluster: 5 candidates with blended scores 0.80, 0.79, 0.78, 0.77, 0.76.** All five should have normalized scores relatively close together (the worst should be above 0.2). The spread should be much smaller than in a case with a wide gap. Verify the specific exponential decay values.
36. **Wide spread: 5 candidates with blended scores 0.90, 0.70, 0.50, 0.30, 0.10.** The worst candidate should be near 0.05. The dropoff should be steep. Verify specific values.
37. **Best candidate always gets exactly 1.0, not approximately 1.0.** With varying pool sizes (2, 10, 100 candidates), check that the max value in each space's normalized dict is exactly `1.0` (not 0.9999... due to floating point).
38. **Pool of size 1 (only one candidate with blended > 0 in a space).** That candidate should get exactly 1.0. Other candidates should be absent from that space's normalized dict.
39. **Two candidates with blended scores that differ by an extremely small epsilon (e.g. 0.000001).** Verify both still receive high normalized scores (close to 1.0), since the range is tiny and the gap fraction is nearly 0 for both.
40. **Candidates with very small blended scores (e.g., 0.001 and 0.002).** Normalization should still work correctly — it operates on relative distance, not absolute magnitude. The candidate with 0.002 should get 1.0 and the one with 0.001 should get `exp(-3.0)` ≈ 0.05.

---

## 6 — Weight Computation (Stage 4 correctness)

41. **All non-anchor spaces `LARGE`.** Active non-anchor mean = 3.0. Anchor raw = 2.4. All 7 non-anchor get 3.0. Total = 23.4. Verify each normalized weight: anchor ≈ 0.1026, each non-anchor ≈ 0.1282.
42. **All non-anchor spaces `SMALL`.** Active mean = 1.0. Anchor raw = 0.8. Verify weights.
43. **Mixed: one `LARGE`, one `MEDIUM`, one `SMALL`, four `NOT_RELEVANT`.** Active non-anchor mean = 2.0. Anchor raw = 1.6. Verify each weight numerically.
44. **Single non-anchor space active (`LARGE`).** Anchor raw = 3.0 × 0.8 = 2.4. That space raw = 3.0. Total = 5.4. Anchor ≈ 0.444, space ≈ 0.556. Verify.
45. **Single non-anchor space active (`SMALL`).** Anchor raw = 0.8. Space raw = 1.0. Verify anchor ≈ 0.444, space ≈ 0.556.
46. **Verify anchor is always strictly less than the mean of active non-anchor weights** (when at least one non-anchor is active). Anchor raw = 0.8 × mean, so anchor normalized < mean of active non-anchor normalized weights. Test across several configurations.
47. **Promoted spaces participate in weight calculation.** Set three spaces to `NOT_RELEVANT` with subqueries (all promoted to `SMALL`), others `NOT_RELEVANT` with no subquery. Verify the three promoted spaces have `normalized_weight` > 0 and the non-promoted inactive ones have weight 0.
48. **Verify weight ordering reflects relevance ordering.** In a configuration with one `SMALL`, one `MEDIUM`, one `LARGE` space, verify `weight(LARGE) > weight(MEDIUM) > weight(SMALL) > weight(anchor)` where anchor is expected to be below the mean.

---

## 7 — Final Score Composition (Stage 5 correctness)

49. **Candidate present in all active spaces as the top scorer in each.** Should receive a final score very close to 1.0 (exactly 1.0 if it's the sole top scorer in every space with score exactly 1.0 × weight for all spaces).
50. **Candidate present in only one active space.** Final score = that space's weight × 1.0 (if it's the best in that space). Verify the score equals the weight numerically.
51. **Candidate absent from all active spaces (all scores 0.0).** Final score must be exactly 0.0.
52. **Two candidates with complementary coverage.** A scores high in anchor + viewer_experience, B scores high in anchor + production. With viewer_experience weighted higher than production, A should score higher than B. With weights swapped (production weighted higher), B should score higher than A.
53. **Score is additive across spaces.** Take a candidate that scores X in space A (weight wA) and Y in space B (weight wB) and 0 everywhere else. Verify final score ≈ wA × normA(X) + wB × normB(Y) by computing the expected values from the normalized dicts.
54. **No re-normalization: candidate missing from a heavily-weighted space is heavily penalized.** Set viewer_experience to `LARGE` (high weight). Candidate A appears in viewer_experience with a high score. Candidate B does not appear in viewer_experience at all but has excellent scores everywhere else. Verify A scores significantly higher than B.

---

## 8 — Ordering and Ranking Correctness

55. **Strict ordering: candidate with uniformly higher cosine similarities across all spaces ranks first.** Three candidates: A > B > C in every space. A's final score > B's > C's.
56. **Ordering respects weights: a candidate that dominates in the highest-weighted space wins.** Set viewer_experience to `LARGE`, production to `SMALL`. Candidate A has mediocre scores everywhere but high in viewer_experience. Candidate B has high scores everywhere but mediocre in viewer_experience. A should beat B.
57. **Ordering is stable across repeated calls.** Call `calculate_vector_scores` with the same inputs 100 times. All results must be bitwise identical (determinism check).
58. **Ranking is sensible for the "cozy 90s movie" scenario from the design doc.** Use the full example from the plan (3 candidates: strong match, decent match, weak match). Verify the ordering matches expectations and the score gap between the strong and weak match is large.

---

## 9 — Promotion Rule Edge Cases

59. **Promotion doesn't change execution flags.** Space has relevance `NOT_RELEVANT` with subquery. Give a candidate a non-zero `_score_original` value for that space (stale data that shouldn't be used). Verify this original score does NOT affect the final score — only the subquery score matters because `did_run_original` is False for promoted spaces.
60. **Promotion doesn't double-promote.** Space has relevance `SMALL` with subquery present. Verify `effective_relevance` is `SMALL` (not promoted further to `MEDIUM`).
61. **Multiple simultaneous promotions.** Three spaces are `NOT_RELEVANT` with subquery text. Verify all three get `effective_relevance` of `SMALL`, all three contribute to the weight calculation, and anchor's weight is computed from the mean of all active non-anchor weights including the promoted ones.
62. **Promotion affects final scores.** Two otherwise-identical requests: one where reception has relevance `NOT_RELEVANT` with subquery (promoted), one where reception has relevance `NOT_RELEVANT` without subquery (not promoted). A candidate with a high reception subquery score should have a higher final score in the first request than in the second.

---

## 10 — Candidates Not Returned in Searches (0.0 score handling)

63. **Candidate with blended=0.0 in an active space is excluded from that space's normalization pool.** Two candidates in a space: A with blended=0.80, B with blended=0.0. The normalization pool should contain only A. B should have a normalized score of 0.0 (absent from `per_space_normalized`). A should have normalized=1.0 (best and only in pool).
64. **A candidate with blended=0.0 does NOT drag down the pool statistics.** Three candidates: A=0.80, B=0.79, C=0.0. The normalization should use range = 0.80 - 0.79 = 0.01, NOT 0.80 - 0.0 = 0.80. Both A and B should get high normalized scores (near 1.0). Verify C gets 0.0.
65. **Candidate that appears in only one of many active spaces.** Final score is pulled down heavily because it gets 0.0 in all other spaces (not re-normalized). Verify it scores much lower than a candidate that appears in all active spaces, even if its single-space score is excellent.
66. **All candidates have blended=0.0 in a particular active space.** This space's `per_space_normalized` should be an empty dict. No candidate gets any contribution from this space. Final scores should only reflect other active spaces.

---

## 11 — Numerical Precision and Stability

67. **Very large candidate pool (e.g., 5000 candidates).** Run with a realistically-sized pool and verify: all invariants hold, performance is reasonable (should complete in well under 1 second), no floating point overflow or NaN.
68. **Cosine similarities at the boundary: 0.0 exactly.** Candidate has a score of exactly 0.0 in a search that ran (it wasn't returned in top-N). Verify it's treated as blended=0.0 and excluded from normalization.
69. **Cosine similarities that are extremely small (e.g., 0.0001).** Verify this IS included in the normalization pool (it's > 0.0) and gets a valid normalized score.
70. **Blended score that's barely above zero due to the 80/20 split.** Candidate has subquery=0.0 and original=0.001 in a both-ran space. Blended = 0.0002. Verify this still participates in normalization and receives a positive normalized score.
71. **All 8 spaces active, all with maximum relevance, every candidate appearing in every search.** Stress test the full pipeline end-to-end. Verify weights, normalization, and final scores are all well-formed.
72. **Two candidates whose final scores differ by less than 1e-10.** Verify the scoring function does not introduce spurious ordering due to floating-point accumulation order. Both scores should be the same to within machine epsilon.

---

## 12 — Each Space Functions Independently

73. **Scores in one space do not affect normalization in another space.** Three candidates. In space A, candidate 1 is the best. In space B, candidate 3 is the best. Verify each space's `per_space_normalized` independently reflects its own pool's ranking without contamination. Candidate 1 gets 1.0 in space A's normalized dict; candidate 3 gets 1.0 in space B's normalized dict.
74. **Adding a new candidate to one space's pool doesn't change another space's normalization.** Run once with candidates {1, 2} where both have scores in spaces A and B. Run again with candidates {1, 2, 3} where candidate 3 has scores only in space A. Space B's normalized scores for candidates 1 and 2 should be identical in both runs.

---

## 13 — Stale Data and Defensive Behavior

75. **Candidate has non-zero `_score_original` in a space that was NOT searched (relevance=NOT_RELEVANT, no subquery).** The stale original score should be completely ignored. Verify the candidate's final score is unaffected by this stale value — it should be identical to a run where that stale value is 0.0.
76. **Candidate has non-zero `_score_subquery` in a space where no subquery text was generated.** The stale subquery score should be ignored. Verify identical result to a run where that value is 0.0.
77. **Candidate has non-zero `_score_original` in a promoted-SMALL space (where `did_run_original` is False).** The original score should NOT be used. Only the subquery score should contribute. Verify by comparing to a run where the original score is 0.0 — results must be identical.

---

## 14 — Per-Space Isolation Tests (one space at a time)

78. **For each of the 7 non-anchor spaces individually:** Set up a configuration where only that space (plus anchor) is active. Provide 3 candidates with varying scores. Verify the final score ordering is correct and that only anchor and that space appear in `per_space_normalized`. (7 sub-tests: plot_events, plot_analysis, viewer_experience, watch_context, narrative_techniques, production, reception.)

---

## 15 — Regression Scenarios from the Design Document

79. **Plan example: "You've Got Mail" vs "The Shawshank Redemption" scenario.** Reproduce the exact inputs from the design plan. Verify "You've Got Mail" scores dramatically higher than "Shawshank." Verify the weight distribution matches the plan's computed values (anchor ≈ 0.138, viewer_experience ≈ 0.259, etc.).
80. **Plan Appendix D Scenario 2: candidate appears only in anchor.** Verify it receives a low final score equal to `anchor_weight × 1.0` (if it's the best in anchor) or `anchor_weight × normalized_score` otherwise.
81. **Plan Appendix D Scenario 3: candidate in subquery but not original in a blended space.** Verify the 20% penalty is reflected in the final score vs an identical candidate that appeared in both.
82. **Plan Appendix D Scenario 4: candidate in original but not subquery in a blended space.** Verify the 80% penalty is reflected. This candidate should score much worse than one that appeared in the subquery.

---

## 16 — Output Structure Completeness

83. **`per_space_normalized` keys exactly match the set of active space names.** No extra keys, no missing keys.
84. **`per_space_normalized` for an active space with zero qualifying candidates is an empty dict** (not absent from the top-level dict — it should still be present as a key mapping to `{}`).
85. **`space_contexts` entries have all fields populated.** No None values on `did_run_original`, `did_run_subquery`, or `normalized_weight`. `effective_relevance` is None only for anchor and non-None for all others.
86. **`final_scores` keys are exactly the same set as input `candidates` keys.** Test with non-sequential movie IDs (e.g., {7, 42, 999, 100000}) to ensure no assumptions about ID contiguity.