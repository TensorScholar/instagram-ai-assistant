#!/bin/bash
# Aura Platform - Vault Setup Script
# This script sets up HashiCorp Vault with necessary secrets for local development

set -e

echo "🔐 Setting up HashiCorp Vault for Aura Platform..."

# Check if Vault is running
if ! curl -s http://localhost:8200/v1/sys/health > /dev/null 2>&1; then
    echo "❌ Vault is not running. Please start Vault first:"
    echo "   docker-compose up -d vault"
    exit 1
fi

# Set Vault address
export VAULT_ADDR="http://localhost:8200"
export VAULT_TOKEN="aura-dev-token-2024"

echo "✅ Vault is running and accessible"

# Enable KV secrets engine
echo "📁 Enabling KV secrets engine..."
vault secrets enable -path=secret kv-v2 || echo "KV engine already enabled"

# Create secrets for default tenant
TENANT_ID="default-tenant-001"
SECRET_PATH="secret/data/aura/${TENANT_ID}"

echo "🔑 Creating secrets for tenant: ${TENANT_ID}"

# Create the secrets
vault kv put "${SECRET_PATH}" \
    postgres_password="dev_postgres_password_2024" \
    rabbitmq_password="dev_rabbitmq_password_2024" \
    redis_password="dev_redis_password_2024" \
    secret_key="dev_secret_key_$(openssl rand -hex 32)" \
    instagram_app_id="dev_instagram_app_id" \
    instagram_app_secret="dev_instagram_app_secret" \
    instagram_webhook_verify_token="dev_webhook_token_$(openssl rand -hex 16)" \
    gemini_api_key="dev_gemini_api_key" \
    openai_api_key="dev_openai_api_key"

echo "✅ Secrets created successfully"

# Create Vault policy for Aura Platform
echo "📋 Creating Vault policy..."
vault policy write aura-platform - <<EOF
path "secret/data/aura/*" {
  capabilities = ["read"]
}

path "secret/metadata/aura/*" {
  capabilities = ["list", "read"]
}
