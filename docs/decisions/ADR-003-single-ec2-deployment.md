# ADR-003: Single EC2 Deployment

**Status:** Active

## Context

The system needs to host Postgres, Qdrant, Redis, the API server,
and Nginx. QPS is low and latency is dominated by LLM calls, not
infrastructure.

## Decision

Run all services as Docker containers on a single EC2 t3.large
(2 vCPU, 8 GB RAM), orchestrated with Docker Compose. Nginx runs
on the same instance as a reverse proxy with HTTPS termination
via Let's Encrypt.

### RAM Budget

| Service | Est. RAM |
|---------|----------|
| Qdrant (quantized + memmap) | 2.5–3.0 GB |
| Postgres | 200–400 MB |
| Redis | 200–500 MB |
| API server | 300–500 MB |
| OS + Docker overhead | 500–800 MB |
| **Total** | **~4–5 GB** |

### Monthly Cost

| Resource | Cost |
|----------|------|
| EC2 t3.large (on-demand) | ~$60/mo |
| EC2 t3.large (1yr reserved) | ~$37/mo |
| EBS GP3 250 GB | ~$20/mo |
| S3 backups | ~$1–2/mo |
| OpenAI (LLM + embeddings) | ~$2–10/mo |
| **Total (reserved)** | **~$60–69/mo** |

## Alternatives Considered

1. **Managed services** (RDS, ElastiCache, ECS): Adds $150–200/mo
   with no meaningful benefit at this scale.
2. **Serverless** (Lambda + managed DBs): Higher latency for cold
   starts, complex connection management, higher cost.
3. **Multi-instance**: Unnecessary complexity for low QPS.

## Consequences

- All inter-service communication over Docker internal network.
  Postgres, Redis, and Qdrant ports NOT exposed to the internet.
- Daily backups: `pg_dump` → S3, Qdrant snapshot → S3.
- Elastic IP for stable DNS. Free when attached to running instance.
- EBS GP3 250 GB for Postgres data and Qdrant memmap files.

## References

- guides/server_architecture_guide.md (sections 1, 8)
