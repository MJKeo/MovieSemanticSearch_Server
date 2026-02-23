#!/usr/bin/env sh
set -eu

QDRANT_URL="${QDRANT_URL:-http://qdrant:6333}"
ALIAS="${QDRANT_COLLECTION_ALIAS:-movies}"
PHYSICAL="${QDRANT_PHYSICAL_COLLECTION:-movies_v1}"

echo "[qdrant-init] Waiting for Qdrant at $QDRANT_URL ..."
until curl -fsS "$QDRANT_URL/healthz" >/dev/null; do
  sleep 0.5
done
echo "[qdrant-init] Qdrant is up."

# --- Create collection (PUT is idempotent; 409 means it already exists) ---
echo "[qdrant-init] Creating collection $PHYSICAL (if missing) ..."

create_code="$(
  curl -sS -o /dev/null -w "%{http_code}" \
    -X PUT "$QDRANT_URL/collections/$PHYSICAL" \
    -H "Content-Type: application/json" \
    --data-binary @- <<'JSON'
{
  "vectors": {
    "anchor": { "size": 1536, "distance": "Cosine", "on_disk": true },
    "plot_events": { "size": 1536, "distance": "Cosine", "on_disk": true },
    "plot_analysis": { "size": 1536, "distance": "Cosine", "on_disk": true },
    "viewer_experience": { "size": 1536, "distance": "Cosine", "on_disk": true },
    "watch_context": { "size": 1536, "distance": "Cosine", "on_disk": true },
    "narrative_techniques": { "size": 1536, "distance": "Cosine", "on_disk": true },
    "production": { "size": 1536, "distance": "Cosine", "on_disk": true },
    "reception": { "size": 1536, "distance": "Cosine", "on_disk": true }
  },
  "quantization_config": {
    "scalar": { "type": "int8", "quantile": 0.99, "always_ram": true }
  }
}
JSON
)"

case "$create_code" in
  200|201|409) echo "[qdrant-init] create_collection: ok ($create_code)" ;;
  *) echo "[qdrant-init] create_collection: failed ($create_code)"; exit 1 ;;
esac

# --- Helper to create payload indexes (409 means it already exists) ---
create_index () {
  field="$1"
  echo "[qdrant-init] Creating payload index: $field"

  code="$(
    curl -sS -o /dev/null -w "%{http_code}" \
      -X PUT "$QDRANT_URL/collections/$PHYSICAL/index" \
      -H "Content-Type: application/json" \
      --data-binary @- <<JSON
{ "field_name": "$field", "field_schema": "integer" }
JSON
  )"

  case "$code" in
    200|201|409) echo "[qdrant-init] index $field: ok ($code)" ;;
    *) echo "[qdrant-init] index $field: failed ($code)"; exit 1 ;;
  esac
}

create_index "release_ts"
create_index "runtime_minutes"
create_index "maturity_rank"
create_index "genre_ids"
create_index "watch_offer_keys"

# --- Create alias (409 means it already exists) ---
echo "[qdrant-init] Ensuring alias $ALIAS -> $PHYSICAL"

alias_code="$(
  curl -sS -o /dev/null -w "%{http_code}" \
    -X POST "$QDRANT_URL/collections/aliases" \
    -H "Content-Type: application/json" \
    --data-binary @- <<JSON
{
  "actions": [
    {
      "create_alias": {
        "collection_name": "$PHYSICAL",
        "alias_name": "$ALIAS"
      }
    }
  ]
}
JSON
)"

case "$alias_code" in
  200|201|409) echo "[qdrant-init] alias: ok ($alias_code)" ;;
  *) echo "[qdrant-init] alias: failed ($alias_code)"; exit 1 ;;
esac

echo "[qdrant-init] Done."