# Aura

> **AI-powered Instagram commerce assistant that actually works**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Transform Instagram DMs into intelligent shopping experiences. Aura handles product recommendations, customer support, and order management through conversational AI that understands context and scales automatically.

## Architecture Flow

![Aura Architecture Flow](https://github.com/TensorScholar/instagram-ai-assistant/assets/aura-architecture-flow.gif)

*Watch the animated flow: Instagram webhooks → AI processing → intelligent responses*

## Why Aura?

Instagram is where people discover products, but most businesses still handle DMs manually. Aura changes that by:

- **Understanding context** - Knows your products, inventory, and customer history
- **Scaling automatically** - Handles thousands of conversations without breaking
- **Staying secure** - Multi-tenant architecture keeps customer data isolated
- **Learning continuously** - Gets better with every interaction

## Quick Start

```bash
# Clone and start
git clone https://github.com/TensorScholar/instagram-ai-assistant.git
cd instagram-ai-assistant
make dev-up

# That's it! Your AI assistant is running locally
```

## What's Inside

```
src/
├── api_gateway/          # FastAPI webhook handler
├── intelligence_worker/  # AI processing with Gemini + fallbacks
├── ingestion_worker/     # Product catalog sync
└── shared_lib/          # Common utilities, models, AI clients
```

## Tech Stack

- **Backend**: Python 3.12, FastAPI, Celery
- **AI**: Google Gemini (primary), OpenAI (fallback)
- **Data**: PostgreSQL, Milvus (vector search), Redis
- **Infrastructure**: Docker, Kubernetes, RabbitMQ
- **Security**: HashiCorp Vault, tenant isolation

## Features

### 🤖 Smart Conversations
- RAG-powered responses using your product catalog
- Context-aware recommendations based on customer history
- Multi-language support with proper fallbacks

### 🏢 Multi-Tenant Ready
- Complete data isolation between businesses
- Tenant-specific AI models and configurations
- Secure secret management per tenant

### 🛡️ Production Hardened
- Circuit breakers prevent cascade failures
- Automatic retries with exponential backoff
- Dead letter queues for problematic messages
- Comprehensive monitoring and alerting

### 📈 Scales Automatically
- Horizontal scaling with Kubernetes
- Separate queues for real-time vs bulk processing
- Connection pooling and resource optimization

## Development

```bash
# Start everything
make dev-up

# Run tests
make test

# Code quality
make lint

# Clean up
make clean
```

## Testing

We test everything:

```bash
# Unit tests
pytest src/ -v

# Integration tests  
pytest tests/integration/ -v

# Stress tests (simulates 100x load)
pytest tests/stress/ -v

# Full validation suite
pytest tests/validation/ -v
```

## Deployment

Production deployment is Kubernetes-ready:

```bash
# Build images
make build

# Deploy with Helm
helm install aura ./kubernetes/helm-chart

# Monitor
kubectl get pods -n aura-platform
```

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed setup instructions.

## Configuration

Environment variables control everything:

```bash
# Required
DATABASE_URL=postgresql://...
REDIS_URL=redis://...
RABBITMQ_URL=amqp://...

# AI APIs
GEMINI_API_KEY=your_key_here
OPENAI_API_KEY=fallback_key_here

# Instagram
INSTAGRAM_APP_ID=your_app_id
INSTAGRAM_APP_SECRET=your_secret
```

## Contributing

We welcome contributions! Here's how:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Architecture

```
Instagram Webhooks → API Gateway → RabbitMQ → AI Workers → Response
                                        ↓
                                   Product Catalog
                                        ↓
                                   Vector Search (Milvus)
```

The system processes Instagram messages through a resilient pipeline that can handle failures gracefully and scale to thousands of concurrent conversations.

## Performance

- **Response time**: < 2 seconds for typical queries
- **Throughput**: 1000+ messages/minute per worker
- **Availability**: 99.9% uptime with proper configuration
- **Scalability**: Linear scaling with additional workers

## Security

- All secrets managed through HashiCorp Vault
- Tenant data completely isolated
- Webhook signature verification
- Input validation and sanitization
- Audit logging for all operations

## Monitoring

Built-in metrics for:
- Queue depths and processing times
- AI API response times and error rates
- Database connection pool utilization
- Circuit breaker states

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Built with ❤️ by [Mohammad Atashi](https://github.com/TensorScholar)

---

**Star this repo if you find it useful!** ⭐