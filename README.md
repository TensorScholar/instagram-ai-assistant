# Aura

<div align="center">

![Aura Logo](https://img.shields.io/badge/Aura-AI%20Assistant-blue?style=for-the-badge&logo=sparkles)

**AI-powered Instagram commerce assistant that actually works**

[![Python](https://img.shields.io/badge/Python-3.12+-3776ab?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ed?style=flat-square&logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Stars](https://img.shields.io/github/stars/TensorScholar/instagram-ai-assistant?style=flat-square&logo=github)](https://github.com/TensorScholar/instagram-ai-assistant)

</div>

---

## 🎯 Overview

Aura transforms Instagram DMs into intelligent shopping experiences. Built for scale, security, and reliability.

**What it does:**
- 🤖 **Smart Conversations** - AI that understands your products and customers
- 🏢 **Multi-Tenant** - Secure isolation for multiple businesses
- 🛡️ **Production Ready** - Circuit breakers, retries, monitoring
- 📈 **Auto-Scaling** - Handles thousands of conversations

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

**That's it!** Your AI assistant is now running locally.

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

## 📊 Performance

- **Response Time**: < 2 seconds
- **Throughput**: 1000+ messages/minute
- **Availability**: 99.9% uptime
- **Scalability**: Linear scaling

---

## 🔒 Security

- ✅ Multi-tenant data isolation
- ✅ HashiCorp Vault integration
- ✅ Webhook signature verification
- ✅ Input validation & sanitization
- ✅ Audit logging

---

## 📈 Monitoring

Built-in metrics for:
- Queue depths and processing times
- AI API response times and errors
- Database connection pool usage
- Circuit breaker states

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Mohammad Atashi** - [@TensorScholar](https://github.com/TensorScholar)

---

<div align="center">

**⭐ Star this repo if you find it useful!**

Made with ❤️ for the future of AI-powered commerce

</div>