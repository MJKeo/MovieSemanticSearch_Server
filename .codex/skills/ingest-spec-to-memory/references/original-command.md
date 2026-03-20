# .claude/commands/ingest-spec-to-memory.md

I am providing a product spec. Your job is to read it thoroughly,
cross-reference it against the existing documentation system, and
update permanent storage to reflect the spec's content — surfacing
conflicts for my resolution before writing anything.

Before analyzing the spec, read:
- docs/PROJECT.md
- docs/conventions.md
- All files in docs/decisions/
- All files in docs/modules/

## Process

### Phase 1: Extract Knowledge

Read the spec and identify every piece of information that falls
into one of these categories:

- **Product context:** Target audience, problem statement, goals,
  success criteria, priorities, constraints
- **Decisions:** Choices the spec makes with reasoning — technology
  picks, architectural approaches, tradeoff resolutions
- **Module impacts:** How existing modules should behave, new
  modules to create, changes to module boundaries or interactions
- **Conventions:** New patterns, naming schemes, data formats, or
  rules the spec introduces

### Phase 2: Conflict Detection

Compare every extracted item against the existing documentation.
Organize findings into:

**No conflict — new information:**
Items that add to our docs without contradicting anything. List
each with where it would go (PROJECT.md, decisions/, modules/).

**Conflict — needs resolution:**
Items where the spec says one thing and our docs say another.
For each conflict:
- What the spec says (quote or paraphrase the relevant section)
- What our docs currently say (cite the file and section)
- Why this matters (what breaks or changes if we go one way)
- Your assessment of which seems more current or correct, framed
  as input, not a decision

STOP HERE. Present all conflicts and wait for me to resolve each
one. Do not proceed to Phase 3 until I have addressed every
conflict.

### Phase 3: Update Documentation (only after all conflicts resolved)

Based on the spec content and my conflict resolutions:

**docs/PROJECT.md** — If the spec updates product context,
priorities, or constraints, draft the specific changes and show
them to me for approval. Do NOT write to PROJECT.md without my
explicit approval.

**docs/decisions/** — For each significant decision in the spec,
create a numbered decision record. Include the spec as the context
source. If the decision supersedes an existing record, update the
old record's Status to "Superseded by [NNN]".

**docs/modules/** — Update any module docs affected by the spec.
If the spec introduces a new module, create its doc.

**docs/conventions.md** — If the spec introduces new conventions,
present them to me for approval before adding. Do NOT write to
conventions.md without my explicit approval.

### Phase 4: Summary

Report what was updated:
- Decisions created (filenames + one-line summaries)
- Module docs created or updated (filenames + what changed)
- PROJECT.md changes made (if approved)
- Conventions added (if approved)
- Any spec content that didn't fit into the documentation
  system (flag for manual handling)

## Rules

- Never silently overwrite existing documentation
- Every conflict must be resolved by me before you write anything
- PROJECT.md and conventions.md changes require my explicit approval
- Decision records and module docs can be written after conflicts
  are resolved without additional approval
- If the spec is ambiguous about something, add it to the conflict
  list as "Spec unclear — needs clarification" rather than guessing
- Reference the spec by section when citing what it says

Focus on: $ARGUMENTS