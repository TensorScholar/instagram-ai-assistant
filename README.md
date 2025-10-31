# Aura

<div align="center">

![Aura Logo](https://img.shields.io/badge/Aura-AI%20Assistant-blue?style=for-the-badge&logo=sparkles)

**AI-powered Instagram commerce assistant**

[![Python](https://img.shields.io/badge/Python-3.12+-3776ab?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ed?style=flat-square&logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Stars](https://img.shields.io/github/stars/TensorScholar/instagram-ai-assistant?style=flat-square&logo=github)](https://github.com/TensorScholar/instagram-ai-assistant)

</div>

---

## 🎯 Overview

Aura turns Instagram DMs into guided shopping conversations. The system is designed with scalability, security, and reliability in mind.

**Key capabilities:**
- 🤖 **Smart Conversations** - AI-assisted replies with product context
- 🏢 **Multi-Tenant** - Per-tenant data isolation
- 🛡️ **Resilience** - Circuit breakers, retries, health checks
- 📈 **Scalability** - Horizontal scale-out via workers and queues

---

## 🏗️ Architecture

<div align="center">

```mermaid
flowchart LR
    A[📱 Instagram] --> B[🚪 API Gateway]
    B --> C[📨 RabbitMQ]
    C --> D[🧠 AI Worker]
    C --> E[📊 Ingestion Worker]
    D --> F[💾 PostgreSQL]
    D --> G[🔍 Milvus]
    D --> H[⚡ Redis]
    E --> F
    E --> G
    I[🔐 Vault] --> B
    I --> D
    I --> E
    D --> A
```

</div>

**Flow:** Instagram webhook → API Gateway → Message Queue → AI Processing → Response

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/TensorScholar/instagram-ai-assistant.git
cd instagram-ai-assistant

# 2. Start the platform
make dev-up

# 3. Verify it's running
make health-check
```

---

## 📁 Project Structure

```
src/
├── api_gateway/          # Webhook handler (FastAPI)
├── intelligence_worker/  # AI processing (Celery)
├── ingestion_worker/     # Data sync (Celery)
└── shared_lib/          # Common utilities
```

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Backend** | Python 3.12, FastAPI, Celery |
| **AI** | Google Gemini, OpenAI (fallback) |
| **Data** | PostgreSQL, Milvus, Redis |
| **Infrastructure** | Docker, Kubernetes, RabbitMQ |
| **Security** | HashiCorp Vault |

---

## ⚙️ Configuration

Create a `.env` file:

```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost/aura

# Message Queue
RABBITMQ_URL=amqp://user:pass@localhost:5672

# AI APIs
GEMINI_API_KEY=your_gemini_key
OPENAI_API_KEY=your_openai_key

# Instagram
INSTAGRAM_APP_ID=your_app_id
INSTAGRAM_APP_SECRET=your_secret
```

---

## 🧪 Testing

```bash
# Run all tests
make test

# Specific test types
pytest src/ -v                    # Unit tests
pytest tests/integration/ -v      # Integration tests
pytest tests/validation/ -v       # Validation tests
```

---

## 🚢 Deployment

### Local Development
```bash
make dev-up    # Start all services
make dev-down  # Stop all services
```

### Production (Kubernetes)
```bash
make build                           # Build images
helm install aura ./kubernetes/helm-chart  # Deploy
kubectl get pods -n aura-platform   # Monitor
```

---

## 🔒 Security

Includes support for:
- Per-tenant data isolation
- Vault-based secret management
- Webhook signature verification
- Input validation & sanitization

---

## 📈 Observability

Hooks available for:
- Queue depth and processing metrics
- AI API response times and errors
- Database connection pool monitoring
- Circuit breaker state tracking

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-change`
3. Commit changes: `git commit -m 'Describe your change'`
4. Push to branch: `git push origin feature/your-change`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Mohammad Atashi** - [@TensorScholar](https://github.com/TensorScholar)

<div align="center">

**⭐ Star this repo if you find it useful!**

</div>