# Aura

> **The Resilient AI Assistant Platform for Modern E-commerce on Instagram**

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/TensorScholar/instagram-ai-assistant)
[![Code Coverage](https://img.shields.io/badge/coverage-95%25-brightgreen.svg)](https://github.com/TensorScholar/instagram-ai-assistant)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/TensorScholar/instagram-ai-assistant)
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://python.org)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://docker.com)

## 🌟 Introduction

Aura is a cutting-edge, multi-tenant AI platform designed to revolutionize e-commerce interactions on Instagram. Built with enterprise-grade resilience and fault tolerance, Aura seamlessly integrates AI-powered conversational capabilities with Instagram's ecosystem, enabling businesses to provide intelligent, context-aware customer support and product recommendations.

## ✨ Core Features

- 🏢 **Multi-Tenant Architecture** - Secure, isolated environments for each business
- 🧠 **AI-Powered RAG Pipeline** - Intelligent retrieval-augmented generation for accurate responses
- 🛡️ **Fault-Tolerant & Resilient** - Circuit breakers, retry mechanisms, and poison pill protection
- 🚀 **Scalable by Design** - Event-driven microservices with horizontal scaling capabilities
- 🔒 **Enterprise Security** - Vault integration, tenant isolation, and comprehensive audit trails
- 📊 **Real-time Monitoring** - Comprehensive metrics and alerting for production environments
- 🔄 **Event-Driven Architecture** - Asynchronous processing with RabbitMQ and Celery
- 🐳 **Container-Ready** - Docker and Kubernetes deployment with Helm charts

## 🏗️ Architecture Overview

Aura follows a modern event-driven microservices architecture, designed for high availability and scalability:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   API Gateway   │────│  Intelligence    │────│  Ingestion      │
│   (FastAPI)     │    │  Worker          │    │  Worker         │
│                 │    │  (Celery)        │    │  (Celery)       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌─────────────────┐             │
         └──────────────│   RabbitMQ      │─────────────┘
                        │   (Message      │
                        │    Broker)      │
                        └─────────────────┘
                                 │
         ┌─────────────────────────────────────────────┐
         │              Shared Libraries                │
         │  ┌─────────┐ ┌─────────┐ ┌─────────────────┐ │
         │  │PostgreSQL│ │ Milvus  │ │      Redis      │ │
         │  │(Primary) │ │(Vector) │ │   (Cache/Celery)│ │
         │  └─────────┘ └─────────┘ └─────────────────┘ │
         └─────────────────────────────────────────────┘
```

### Core Components

- **API Gateway**: FastAPI-based entry point handling Instagram webhooks and API requests
- **Intelligence Worker**: AI processing engine with Gemini integration and RAG capabilities
- **Ingestion Worker**: Data processing pipeline for product catalogs and tenant management
- **Shared Libraries**: Common utilities for database access, AI operations, and security

## 🛠️ Technology Stack

| Category | Technology | Purpose |
|----------|------------|---------|
| **Backend** | Python 3.12+ | Core application language |
| **Web Framework** | FastAPI | High-performance API framework |
| **Database** | PostgreSQL | Primary data storage |
| **Vector Store** | Milvus | AI embeddings and similarity search |
| **Cache** | Redis | Caching and Celery backend |
| **Message Queue** | RabbitMQ | Asynchronous task processing |
| **Task Queue** | Celery | Distributed task processing |
| **AI/ML** | Google Gemini | Large language model integration |
| **Containerization** | Docker | Application containerization |
| **Orchestration** | Kubernetes | Production deployment |
| **Package Management** | Helm | Kubernetes package manager |
| **Secrets Management** | HashiCorp Vault | Secure credential storage |
| **Monitoring** | Prometheus/Grafana | Metrics and observability |

## 🚀 Getting Started

### Prerequisites

- Docker and Docker Compose
- Python 3.12+ (for local development)
- Make (for using the Makefile)

### Local Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/TensorScholar/instagram-ai-assistant.git
   cd instagram-ai-assistant
   ```

2. **Start the development environment**
   ```bash
   make dev-up
   ```

3. **Initialize Vault secrets (optional)**
   ```bash
   make setup-vault
   ```

4. **Verify the setup**
   ```bash
   make health-check
   ```

### Available Make Commands

```bash
make dev-up          # Start development environment
make dev-down        # Stop development environment
make test            # Run all tests
make lint            # Run code quality checks
make build           # Build Docker images
make clean           # Clean up containers and volumes
make setup-vault     # Initialize Vault with secrets
make health-check    # Check service health
```

## 🧪 Running Tests

Execute the comprehensive test suite:

```bash
make test
```

For specific test categories:

```bash
# Unit tests
python3 -m pytest src/ -v

# Integration tests
python3 -m pytest tests/integration/ -v

# Validation tests
python3 -m pytest tests/validation/ -v

# Stress tests
python3 -m pytest tests/stress/ -v
```

## 🚀 Deployment

For production deployment instructions, comprehensive configuration guides, and Kubernetes setup, please refer to [DEPLOYMENT.md](DEPLOYMENT.md).

### Quick Production Deployment

```bash
# Build production images
make build

# Deploy to Kubernetes
helm install aura ./kubernetes/helm-chart

# Verify deployment
kubectl get pods -n aura-platform
```

## 📚 Documentation

- [DEPLOYMENT.md](DEPLOYMENT.md) - Comprehensive deployment guide
- [RESILIENCE_PATTERNS.md](RESILIENCE_PATTERNS.md) - Fault tolerance patterns
- [SECRETS.md](SECRETS.md) - Secrets management guide

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines and code of conduct for details on how to get involved.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ✍️ Author

This project is passionately crafted and maintained by **Mohammad Atashi**.

---

<div align="center">
  <strong>Built with ❤️ for the future of AI-powered e-commerce</strong>
</div>