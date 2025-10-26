# Aura
### The Resilient AI Copilot for Instagram-Native Commerce

[![CI](https://img.shields.io/badge/CI-passing-00d1b2?style=for-the-badge)](https://github.com/TensorScholar/instagram-ai-assistant/actions)
[![Coverage](https://img.shields.io/badge/Coverage-95%25-44cc11?style=for-the-badge)](https://github.com/TensorScholar/instagram-ai-assistant)
[![License](https://img.shields.io/badge/License-MIT-informational?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12%2B-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ed?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)

---

## 📌 Executive Snapshot
>Aura orchestrates intelligent, multi-tenant customer experiences on Instagram. It fuses retrieval-augmented generation, high-fidelity data pipelines, and zero-trust security patterns into a production-ready platform that feels effortless to operate.

- **Mission-aligned design** — built for modern e-commerce, tuned for conversational excellence.
- **Enterprise posture** — circuit breakers, poison pill defenses, Vault-delivered secrets, and DLQs baked in.
- **Future-proof architecture** — event-driven, Kubernetes-native, and stress-tested against zero-day scenarios.

---

## 📚 Table of Contents
1. [Why Aura Exists](#-why-aura-exists)
2. [Platform Highlights](#-platform-highlights)
3. [Architecture in Motion](#-architecture-in-motion)
4. [Technology Stack](#-technology-stack)
5. [Getting Started](#-getting-started)
6. [Running the Test Suite](#-running-the-test-suite)
7. [Deployment Pathways](#-deployment-pathways)
8. [Operational Excellence](#-operational-excellence)
9. [Author](#-author)

---

## 🧭 Why Aura Exists
Instagram is no longer just a social feed—it is the storefront window for high-intent shoppers. Brands must respond instantly, consistently, and securely. Aura delivers that capability by merging:
- **Deep context** via tenant-aware data ingestion and Milvus-backed embeddings.
- **Conversational intelligence** powered by Google Gemini with OpenAI fallback.
- **Operational rigor** from infrastructure pipelines that expect failure and thrive through it.

---

## ✨ Platform Highlights
- 🏢 **True Multi-Tenancy** — Schema-scoped data, repository-enforced `tenant_id`, and partitioned vector stores prevent bleed-through.
- 🧠 **Precision RAG** — Retrieval-augmented conversations grounded in live product intelligence.
- 🛡️ **Resilience Toolkit** — Tenacity retries, pybreaker circuit breakers, DLQs, idempotent tasks, and poison pill detection.
- 🚀 **Elastic Throughput** — Dedicated Celery queues (`realtime` and `bulk`), Redis-backed idempotency, and RabbitMQ publisher confirms.
- 🔐 **Zero-Trust Secrets** — HashiCorp Vault injection, tenant-specific secret paths, and local Vault bootstrap script.
- 📊 **Observability First** — Metrics for queue depth, circuit breaker state, transactional rollbacks, and LLM health.

---

## 🏗️ Architecture in Motion
```
                    ┌───────────────┐
Instagram Webhooks ─►   API Gateway │ (FastAPI + SlowAPI + Backpressure)
                    └──────┬────────┘
                           │  publishes (persistent)
                           ▼
                   ┌──────────────────┐
                   │   RabbitMQ       │  ◄─ Dead Letter Exchange
                   └──────┬─────┬────┘
                          │     │
             realtime_queue│     │bulk_queue
                          │     │
                ┌─────────▼─┐ ┌─▼──────────┐
                │Intelligence│ │ Ingestion  │
                │  Worker    │ │  Worker    │
                │ (Celery)   │ │ (Celery)   │
                └────┬──────┘ └────┬───────┘
                     │             │
   ┌─────────────────▼─────────────▼────────────────┐
   │            Shared Service Mesh                 │
   │  PostgreSQL │ Milvus (Partitions) │ Redis      │
   │  (SQLAlchemy│ Tenant-specific RAG │ Locks,     │
   │  + Pydantic)│ Embeddings         │ Caching    │
   └───────────────────────────────────────────────┘
                     │
                     ▼
           HashiCorp Vault  (tenant secrets, rotations)
```

**Key flows:**
1. **Webhook ingestion** — signatures validated, payloads normalized, and events published with correlation IDs.
2. **Message intelligence** — AI workers run resilient LLM orchestration with backoff, circuit-breaking, and fallback models.
3. **Data consistency** — Two-phase commits ensure PostgreSQL writes precede Milvus vector ingestion with automatic retries.

---

## 🧰 Technology Stack
| Layer | Technologies | Purpose |
|-------|--------------|---------|
| Application | FastAPI, Pydantic, Celery | APIs, data validation, distributed tasks |
| AI & RAG | Google Gemini, OpenAI, Milvus | Conversational intelligence and embeddings |
| Data | PostgreSQL, SQLAlchemy, Alembic | Tenant-scoped relational storage |
| Messaging | RabbitMQ, kombu | Durable event-driven backbone |
| Caching & Locks | Redis (async) | Idempotency, retry tracking, Celery backend |
| Secrets | HashiCorp Vault, Vault Agent Injector | Secure, tenant-aware secrets delivery |
| Infrastructure | Docker, docker-compose, Kubernetes, Helm | Local-to-production parity |
| Reliability | Tenacity, pybreaker, slowapi | Resilience patterns, rate limiting, circuit breaking |
| Observability | Prometheus-ready metrics | Queue depth, circuit breaker state, pool utilization |

---

## 🚀 Getting Started
### Prerequisites
- Docker & Docker Compose
- Python 3.12+
- GNU Make
- (Optional) HashiCorp Vault CLI for local secret seeding

### Quickstart
```bash
# 1. Clone the repository
git clone https://github.com/TensorScholar/instagram-ai-assistant.git
cd instagram-ai-assistant

# 2. Launch the local platform
make dev-up

# 3. Seed development secrets (optional but recommended)
make setup-vault

# 4. Verify service health
make health-check
```

### Essential Make Targets
```bash
make dev-up          # Start full local stack
make dev-down        # Stop and clean containers
make test            # Execute full pytest suite
make lint            # Run static analysis (flake8, mypy, black --check)
make build           # Build production Docker images
make clean           # Remove containers, volumes, and caches
```

> 💡 **Tip:** Environment variables are loaded from `.env` (see `SECRETS.md` for placeholders). Never commit real secrets.

---

## 🧪 Running the Test Suite
```bash
# Core unit and integration tests
make test

# Granular control
python -m pytest src/ -v                # Unit tests
python -m pytest tests/integration/ -v  # Integration scenarios
python -m pytest tests/stress/ -v       # Load & pool exhaustion simulations
python -m pytest tests/validation/ -v   # Armor-plating multi-pipeline validation
```

---

## 🚢 Deployment Pathways
For full production guidance—Helm values, Vault injector annotations, autoscaling policies—consult [DEPLOYMENT.md](DEPLOYMENT.md).

**Snapshot:**
```bash
# Build hardened images
make build

# Deploy via Helm
helm dependency update kubernetes/helm-chart
helm upgrade --install aura kubernetes/helm-chart --namespace aura-platform

# Observe rollout
kubectl get pods -n aura-platform
```

---

## 📈 Operational Excellence
- **Secrets Management** — `scripts/setup-vault.sh` bootstraps local KV v2, policies, and tenant paths.
- **Observability Hooks** — Exposes metrics such as `celery_tasks_pending`, `llm_circuit_breaker_opens_total`, and `database_transaction_rollbacks_total`.
- **Backpressure Controls** — API Gateway denies traffic when queues approach saturation (HTTP 503 with retry guidance).
- **Idempotency Guarantees** — Redis-backed lock decorator prevents duplicate message processing across workers.

---

## ✍️ Author
This project is passionately crafted and maintained by **Mohammad Atashi**.

---

<div align="center">
  <sub>Built with precision, resilience, and empathy for every conversation.</sub>
</div>
