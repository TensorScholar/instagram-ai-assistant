# Aura Platform

A sophisticated, multi-tenant, event-driven SaaS platform designed to provide hyper-personalized, AI-powered customer assistance for e-commerce businesses on Instagram.

## 🎯 Project Vision

Aura connects to a client's Instagram Business Account and their e-commerce website (e.g., Shopify) to function as an intelligent agent within Instagram Direct Messages, capable of:

- Understanding and answering customer queries in natural language
- Performing semantic and visual searches for products based on text descriptions or user-submitted images
- Providing direct links to product pages on the client's website
- Maintaining a unique personality and brand voice for each client (tenant)
- Operating with complete and secure data isolation between tenants

## 🏗️ Architecture

Aura's architecture is an event-driven, service-oriented ecosystem designed for maximum scalability, fault tolerance, and maintainability:

- **Asynchronous Communication**: Services communicate via RabbitMQ message bus
- **Separation of Concerns**: Each service has a single, well-defined responsibility
- **Stateless Services**: Core processing services are stateless for horizontal scaling
- **Database Specialization**: PostgreSQL for transactional data, Milvus for vector similarity searches
- **Infrastructure as Code**: Complete deployment defined using Helm
- **Security First**: Secrets managed by HashiCorp Vault

## 🛠️ Technology Stack

- **Orchestration**: Docker, K3s (Local/On-prem Kubernetes), Helm
- **Backend**: Python 3.11+, FastAPI
- **Task Processing**: Celery
- **Message Bus**: RabbitMQ
- **Databases**: PostgreSQL, Milvus
- **Secrets Management**: HashiCorp Vault
- **AI Orchestration**: LangChain
- **LLM Providers**: Google Gemini API (Primary), OpenAI GPT API (Secondary)

## 📁 Project Structure

```
aura-platform/
├── .github/workflows/          # CI/CD pipelines
├── kubernetes/helm-chart/      # Kubernetes deployment manifests
├── src/
│   ├── api_gateway/           # FastAPI gateway service
│   ├── intelligence_worker/   # AI processing worker
│   ├── ingestion_worker/      # Data ingestion worker
│   └── shared_lib/           # Shared libraries and schemas
├── docker-compose.yml         # Local development environment
├── Makefile                   # Development commands
└── README.md                  # This file
```

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.11+
- Make

### Development Setup

1. **Clone and Setup Environment**
   ```bash
   git clone <repository-url>
   cd aura-platform
   cp .env.example .env
   # Edit .env with your configuration values
   ```

2. **Start Development Environment**
   ```bash
   make dev-up
   ```

3. **Verify Services**
   ```bash
   make health-check
   ```

### Available Commands

- `make dev-up` - Start all services for development
- `make dev-down` - Stop all services
- `make logs` - View logs from all services
- `make test` - Run all tests
- `make lint` - Run linting and formatting
- `make build` - Build all Docker images
- `make clean` - Clean up containers and volumes

## 🔧 Development

### Code Standards

- **PEP 8 Compliance**: All Python code strictly follows PEP 8
- **Type Hinting**: Comprehensive type hints for all functions and variables
- **Docstrings**: Google-style docstrings for all modules, classes, and functions
- **Linting**: Code must pass Flake8 and be formatted with Black
- **Testing**: Comprehensive test coverage with pytest

### Configuration Management

- No hardcoded values allowed
- All configurations loaded from environment variables
- Pydantic BaseSettings for configuration validation
- Secrets managed by HashiCorp Vault

### Security Protocols

- Tenant ID enforcement for all database operations
- Vault secret injection for Kubernetes deployments
- Rigorous input validation using Pydantic models
- Structured logging with correlation IDs and tenant IDs

## 🧪 Testing

The project includes comprehensive testing at multiple levels:

- **Unit Tests**: Individual component testing
- **Integration Tests**: Service-to-service communication testing
- **End-to-End Tests**: Complete workflow validation
- **Pipeline Tests**: Multi-pipeline validation protocols

Run tests with:
```bash
make test
```

## 📦 Deployment

### Local Kubernetes (K3s)

```bash
make k8s-deploy
```

### Production Deployment

```bash
helm install aura ./kubernetes/helm-chart
```

## 📊 Monitoring

- Structured JSON logging across all services
- Prometheus metrics collection
- Health check endpoints
- Distributed tracing with correlation IDs

## ⚡ Scaling Configuration

The Aura platform is designed for horizontal scaling with configurable resource limits. Key scaling parameters can be adjusted via environment variables:

### Database Connection Pool
```bash
# Database connection pool settings
DB_POOL_SIZE=50              # Base connection pool size
DB_MAX_OVERFLOW=100          # Maximum overflow connections
DB_POOL_RECYCLE=1800         # Connection recycle time (30 minutes)
```

### Celery Worker Concurrency
```bash
# Intelligence Worker scaling
INTELLIGENCE_WORKER_CONCURRENCY=20    # Worker concurrency
CELERY_WORKER_PREFETCH_MULTIPLIER=1   # Prefetch multiplier
CELERY_WORKER_MAX_TASKS_PER_CHILD=1000 # Max tasks per worker process
```

### Queue Configuration
The platform uses separate queues for different operation types:
- **realtime_queue**: High-priority messages (limit: 1000 messages)
- **bulk_queue**: Low-priority bulk operations (limit: 5000 messages)

### Kubernetes Resource Limits
Resource quotas are enforced at the namespace level:
- **CPU Limit**: 8 cores total, 2 cores per pod
- **Memory Limit**: 16Gi total, 4Gi per pod
- **Pod Limit**: 20 pods maximum

### Rate Limiting
- **API Gateway**: 60 requests per minute per IP
- **Back-Pressure**: Automatic throttling when queues exceed 5000 messages
- **Health Checks**: Bypass all rate limiting and back-pressure mechanisms

## 🤝 Contributing

1. Follow the coding standards outlined in this README
2. Ensure all tests pass
3. Update documentation as needed
4. Submit pull requests for review

## 📄 License

[License information to be added]

## 🆘 Support

For support and questions, please contact the development team.

---

**Project Aura** - Empowering e-commerce businesses with AI-powered Instagram customer assistance.
