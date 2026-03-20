# .claude/commands/initiate-spec-understanding-conversation.md

I am providing a product spec. Your job is to deeply understand it,
cross-reference it against the actual codebase, and surface every
question, conflict, and decision that needs to be resolved before
implementation begins.

Before analyzing the spec, read:
- docs/PROJECT.md for priorities, constraints, and system overview
- DIFF_CONTEXT.md for any recent changes that may be relevant
- The module docs in docs/modules/ for any modules the spec touches

## Analysis Process

1. **Understand intent:** What is this spec trying to accomplish and
   why? State it back in one paragraph to confirm understanding.

2. **Map to codebase:** Identify which files, modules, and systems
   would need to change. Read them. Understand their current state.

3. **Surface conflicts:** Where does the spec assume something that
   doesn't match the codebase? Where does it contradict an existing
   decision in docs/decisions/? Where does it violate a convention
   in docs/conventions.md or a cross-codebase invariant?

4. **Identify gaps:** What does the spec leave unspecified that must
   be decided before implementation? Think about error handling,
   edge cases, data migration, backward compatibility, and
   performance implications given PROJECT.md priorities.

5. **Challenge assumptions:** Where does the spec make a choice that
   could be improved? Are there simpler approaches? Does it
   over-engineer or under-engineer given the stated priorities?

## Output

Present your findings as discussion points organized by urgency:

**Must resolve before implementation** — Ambiguities or conflicts
that would lead to wrong implementation if assumed.

**Should discuss** — Design choices where alternatives exist and
the tradeoff is worth exploring.

**Worth noting** — Observations that don't block implementation but
may matter later.

For each point:
- State the issue concretely (reference spec sections and code paths)
- Explain why it matters
- Offer your perspective where you have one, but frame it as input
  for discussion, not a decision

Do not create a plan. Do not write code. Do not make implementation
decisions. Only facilitate discussion.

After we've resolved all discussion points, I will ask you to
proceed to planning via /extract-finalized-decisions and then implementation.

Focus on: $ARGUMENTS