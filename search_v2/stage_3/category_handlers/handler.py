# Runs a single category handler on one Step-2 coverage_evidence
# entry: builds the system prompt (via prompt_builder), calls the
# handler LLM with the per-category output schema from
# schema_factories.get_output_schema, then executes or defers each
# emitted endpoint_parameters into a HandlerResult.
#
# Scoped to a single category per invocation — fan-out across
# coverage_evidence entries lives one layer up in the orchestrator.
#
# See search_improvement_planning/category_handler_planning.md
# §"Handler return contract" and §"From LLM output to return
# buckets" for the routing rules this module implements.
