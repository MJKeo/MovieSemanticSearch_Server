# Search V2 — Step 2 → Step 3 → Query Generation runner.
#
# Thin CLI wrapper that takes a raw query, runs Step 2 to extract
# traits, fans out Step 3 over those traits, and then for each
# emitted CategoryCall runs the per-handler query-generation stage.
# Prints the full payload from each stage — including the per-handler
# LLM's reasoning fields (per-endpoint `<route>_walk` blocks,
# coverage_exploration, coverage_assignments, etc.) which the
# production `run_query_generation` strips before returning.
#
# To get those reasoning fields we call the handler LLM directly
# using the same prompt builder + schema factory the production
# handler uses, then run the same extractor on its raw output. This
# keeps the on-the-wire behavior identical while letting us inspect
# the intermediate reasoning the production callsite discards.
#
# Deterministic categories (NO_LLM_PURE_CODE, e.g. TRENDING /
# MEDIA_TYPE) and EXPLICIT_NO_OP categories never reach the LLM, so
# we fall back to the production `run_query_generation` for those
# and just print its output.
#
# Usage:
#   python -m search_v2.run_query_generation
#   python -m search_v2.run_query_generation "your query here"

from __future__ import annotations

import argparse
import asyncio
import json

from pydantic import BaseModel

from implementation.llms.generic_methods import generate_llm_response_async
from schemas.enums import HandlerBucket
from schemas.step_2 import Trait
from schemas.step_3 import CategoryCall, TraitDecomposition
from schemas.trait_category import CategoryName
from search_v2.endpoint_fetching.category_handlers.generated_endpoint_spec import (
    GeneratedEndpointSpec,
)
from search_v2.endpoint_fetching.category_handlers.handler import (
    _HANDLER_LLM_KWARGS,
    _HANDLER_LLM_MODEL,
    _HANDLER_LLM_PROVIDER,
    determine_operation_type,
    run_query_generation,
)
from search_v2.endpoint_fetching.category_handlers.output_extractor import (
    extract_fired_endpoints,
)
from search_v2.endpoint_fetching.category_handlers.prompt_builder import (
    build_system_prompt,
    build_user_message,
)
from search_v2.endpoint_fetching.category_handlers.schema_factories import (
    get_output_schema,
)
from search_v2.step_2 import run_step_2
from search_v2.step_3 import run_step_3


# Default query — picked to exercise the routing path the user is
# investigating (element/motif → keyword endpoint). Override via the
# CLI positional arg.
_DEFAULT_QUERY = "clown horror movies"


# Buckets that bypass the handler LLM entirely.
_NON_LLM_BUCKETS: frozenset[HandlerBucket] = frozenset({
    HandlerBucket.NO_LLM_PURE_CODE,
    HandlerBucket.EXPLICIT_NO_OP,
})


def _dump_pydantic(model: BaseModel) -> str:
    """Serialize a pydantic model to indented JSON for printing.

    `mode="json"` coerces enums / dates / etc. to JSON-native types
    so the output matches what gets serialized over the wire.
    """
    return json.dumps(
        model.model_dump(mode="json"), indent=2, ensure_ascii=False
    )


def _dump_spec(spec: GeneratedEndpointSpec) -> str:
    """Serialize a GeneratedEndpointSpec (dataclass with a pydantic
    `params` field) to indented JSON. Handled separately from
    `_dump_pydantic` because GeneratedEndpointSpec is a dataclass,
    not a BaseModel."""
    payload = {
        "route": spec.route.value,
        "operation_type": spec.operation_type.value,
        "was_promoted": spec.was_promoted,
        "params_wrapper": (
            type(spec.params).__name__ if spec.params is not None else None
        ),
        "params": (
            spec.params.model_dump(mode="json")
            if spec.params is not None
            else None
        ),
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


async def _run_handler_with_full_output(
    *,
    category_call: CategoryCall,
    trait: Trait,
) -> tuple[BaseModel | None, list[GeneratedEndpointSpec]]:
    """Run the handler LLM and return BOTH the raw structured output
    (with all reasoning fields intact) and the extracted endpoint
    specs.

    Mirrors `handler.run_query_generation`'s LLM path inline so we can
    surface the intermediate reasoning fields the production callsite
    discards. For non-LLM categories (deterministic / no-op) we fall
    back to the production function — there's no reasoning to surface,
    just the deterministic output.
    """
    category = category_call.category

    # Deterministic / no-op categories never invoke the LLM, so there
    # is no raw output to surface — defer to the production function.
    if category.bucket in _NON_LLM_BUCKETS:
        specs = await run_query_generation(
            category_call=category_call, trait=trait
        )
        return None, specs

    # Same prompt + schema the production handler builds.
    system_prompt = build_system_prompt(category)
    user_message = build_user_message(category_call)
    response_format = get_output_schema(category)

    # Single attempt here — production retries once on failure, but the
    # debug runner is best-effort and surfacing the raw exception is
    # more useful than masking it.
    raw_output, _input_tokens, _output_tokens = await generate_llm_response_async(
        provider=_HANDLER_LLM_PROVIDER,
        user_prompt=user_message,
        system_prompt=system_prompt,
        response_format=response_format,
        model=_HANDLER_LLM_MODEL,
        **_HANDLER_LLM_KWARGS,
    )

    # Same extractor + operation_type assignment the production handler
    # applies after the LLM call. Keeps the spec list identical to what
    # the production codepath would emit for this CategoryCall.
    fired = extract_fired_endpoints(category, raw_output)
    specs = [
        GeneratedEndpointSpec(
            route=route,
            params=params,
            operation_type=determine_operation_type(
                category, route, trait.polarity
            ),
        )
        for route, params in fired
    ]
    return raw_output, specs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run search_v2 Step 2 → Step 3 → query-generation on a "
            "raw query and print the full output of each stage, "
            "including the handler LLM's reasoning fields."
        )
    )
    parser.add_argument(
        "query",
        type=str,
        nargs="?",
        default=_DEFAULT_QUERY,
        help=(
            "The raw user query to process. Defaults to a built-in "
            "sample query if omitted."
        ),
    )
    return parser.parse_args()


async def _main_async() -> None:
    args = _parse_args()
    print(f"[query] {args.query}")

    # Step 2 — sequential by necessity (Step 3 needs traits).
    print("\n========== STEP 2 ==========")
    analysis, _, _, _ = await run_step_2(args.query)
    print(_dump_pydantic(analysis))

    if not analysis.traits:
        print("\n(no traits emitted by Step 2 — nothing to decompose)")
        return

    # Step 3 — fan out across traits in parallel; result order matches
    # analysis.traits via gather's ordering guarantee.
    step_3_results = await asyncio.gather(
        *(run_step_3(trait) for trait in analysis.traits)
    )

    print("\n========== STEP 3 ==========")
    for trait, (decomposition, _, _, _) in zip(
        analysis.traits, step_3_results
    ):
        print(f'\n--- Trait: "{trait.surface_text}" ---')
        print(_dump_pydantic(decomposition))

    # Query generation — per CategoryCall we want BOTH the raw handler
    # output (with reasoning fields) and the extracted specs. Per-trait
    # we fan out across CategoryCalls to keep wall-clock down; per-trait
    # blocks print sequentially for readable output.
    print("\n========== QUERY GENERATION ==========")
    for trait, (decomposition, _, _, _) in zip(
        analysis.traits, step_3_results
    ):
        print(f'\n--- Trait: "{trait.surface_text}" ---')

        if not decomposition.category_calls:
            print("(no category calls to generate from)")
            continue

        per_call_results = await asyncio.gather(
            *(
                _run_handler_with_full_output(
                    category_call=call, trait=trait
                )
                for call in decomposition.category_calls
            )
        )

        for call, (raw_output, specs) in zip(
            decomposition.category_calls, per_call_results
        ):
            print(f"\n  CategoryCall: {call.category.value}")
            print(f"    bucket: {call.category.bucket.value}")
            print(f"    expressions: {call.expressions}")
            print(f'    retrieval_intent: "{call.retrieval_intent}"')

            # Raw handler-LLM output — every reasoning field the LLM
            # produced. None for deterministic / no-op categories.
            print("    [raw handler output]")
            if raw_output is None:
                print(
                    "      (deterministic / no-op category — no LLM "
                    "call, no reasoning fields)"
                )
            else:
                rendered = _dump_pydantic(raw_output)
                print(
                    "\n".join("      " + ln for ln in rendered.splitlines())
                )

            # Extracted specs — what stage 4 would actually fire.
            if not specs:
                print("    [extracted specs] (none — handler abstained)")
                continue

            print("    [extracted specs]")
            for i, spec in enumerate(specs):
                print(f"      [spec {i}]")
                rendered = _dump_spec(spec)
                print(
                    "\n".join("        " + ln for ln in rendered.splitlines())
                )


def main() -> None:
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
