# 🔐 Aura Platform - Secrets Management

This document outlines all secret placeholders used throughout the Aura Platform and their intended sources.

## Secret Placeholders

### Database Secrets
- `POSTGRES_PASSWORD_PLACEHOLDER` - PostgreSQL database password
- `POSTGRES_USER` - PostgreSQL username (default: `aura_user`)

### Message Queue Secrets
- `RABBITMQ_PASSWORD_PLACEHOLDER` - RabbitMQ password
- `RABBITMQ_USER` - RabbitMQ username (default: `aura_user`)

### Cache Secrets
- `REDIS_PASSWORD_PLACEHOLDER` - Redis password for Celery backend

### Application Secrets
- `SECRET_KEY_PLACEHOLDER` - Django/FastAPI secret key for JWT signing
- `INSTAGRAM_APP_ID_PLACEHOLDER` - Instagram App ID
- `INSTAGRAM_APP_SECRET_PLACEHOLDER` - Instagram App Secret
- `INSTAGRAM_WEBHOOK_VERIFY_TOKEN_PLACEHOLDER` - Instagram webhook verification token
- `GEMINI_API_KEY_PLACEHOLDER` - Google Gemini API key
- `OPENAI_API_KEY_PLACEHOLDER` - OpenAI API key (optional fallback)

### Vault Integration
All secrets should be stored in HashiCorp Vault under the path:
```
secret/data/aura/{tenant_id}
```

### Local Development
For local development, use the `scripts/setup-vault.sh` script to populate Vault with example values.

### Production Deployment
In production, secrets should be:
1. Stored in HashiCorp Vault
2. Injected via Vault Agent Injector
3. Never stored in plain text in configuration files
4. Rotated regularly according to security policies

## Security Notes
- All placeholders must be replaced with actual secrets before deployment
- Never commit actual secrets to version control
- Use environment-specific secret management
- Implement proper secret rotation policies
- Monitor secret access and usage
