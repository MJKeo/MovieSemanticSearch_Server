# Conventions Draft

Observed patterns staged for review. Remove entries you disagree
with, then run /solidify-draft-conventions to merge the rest into
docs/conventions.md.

Entries are added automatically during /safe-clear based on
patterns observed in the session.

## Type-specific behavior belongs on schema classes, not inline in processors
**Observed:** When the batch result processor needed post-validation fixups for concept_tags only, an inline `if metadata_type == CONCEPT_TAGS` check with a lazy import was proposed. The user rejected it as "hacky" and asked for a classmethod on the schema class instead (`validate_and_fix()`), so each metadata type owns its own post-processing logic and the processor just calls the uniform interface.
**Proposed convention:** When a generic pipeline function needs type-specific behavior, put that behavior on the schema/model class as a method with a default implementation, not as inline conditionals in the pipeline. The pipeline calls the uniform interface; the class decides what to do.
**Sessions observed:** 1

## Don't introduce base classes that only add one method — extend existing ones
**Observed:** A `MetadataOutput` base class was created solely to hold `validate_and_fix()`, with `EmbeddableOutput` inheriting from it. The user immediately said "just combine them, I don't see a reason to keep them separate." The extra layer added no value since `ConceptTagsOutput` could have its own standalone override.
**Proposed convention:** Don't create new base classes for a single method. Add the method to the nearest existing base class. If a non-subclass needs the same interface, implement it independently (duck typing) rather than creating a shared base just for one method.
**Sessions observed:** 1

## Validate expected tags against model input evidence, not parametric knowledge
**Observed:** When reviewing concept tag test results, the user required that expected tags be justified by what the model actually sees in its input (user prompt), not by what Claude knows about the movie from parametric knowledge. Light of My Life's `sad_ending` was removed because the input data lacked ending information — even though the movie does end sadly. Oldboy's `kidnapping` was rejected because the input describes imprisonment, not kidnapping.
**Proposed convention:** Test set expected tags must be derivable from the model's actual input data (user prompt contents). If input evidence doesn't support a tag, remove it from expected even if the tag is factually correct — the test measures what the model *should* produce given its inputs, not what's true about the movie.
**Sessions observed:** 1

## Defer LLM output validators to measurement, not hard schema errors
**Observed:** When adding a model_validator to FranchiseOutput that enforced null-pairing constraints (franchise_role must be null when franchise_name is null), the user said to remove it and instead measure how often the LLM violates the constraint during evaluation. If the rate is negligible, keep removed. If significant, re-add as a validate_and_fix() fixup — not a hard validation error that rejects the entire output.
**Proposed convention:** For LLM structured output schemas, don't add model_validators that enforce cross-field consistency constraints before empirical data exists. Remove them pre-evaluation, measure violation rates during testing, and if needed re-add as deterministic fixups in validate_and_fix() rather than hard errors. Hard validators reject useful outputs over fixable issues.
**Sessions observed:** 1

## Comparative evaluation for mutually exclusive LLM classification categories
**Observed:** The concept tags ending category (happy/sad/bittersweet) used sequential threshold-matching with tiebreakers, which caused bittersweet to reach 0% recall (always absorbed by sad). Replacing with comparative evaluation (build the case for each option independently, then select the strongest fit) fixed the structural problem.
**Proposed convention:** When an LLM classification category has mutually exclusive options (at most one can be selected), use comparative evaluation — require the model to evaluate evidence for each option before selecting, rather than sequential threshold-matching where the first strong match wins.
**Sessions observed:** 1

## Plot twist requires audience recontextualization, not just surprise events
**Observed:** The user agreed that early betrayals (Machete's assassination setup) are "plot setups, not plot twists" and asked for the definition to be refined. The key distinction: a twist must change the audience's understanding of events *already shown*. An event that happens before the audience has formed contrary expectations is dramatic irony or setup, not a twist.
**Proposed convention:** When classifying `plot_twist`, require that the revelation recontextualizes earlier events the audience has already processed under a different assumption. Betrayals, deceptions, or surprises that occur before the audience has formed a contrary understanding are plot events, not twists.
**Sessions observed:** 1
