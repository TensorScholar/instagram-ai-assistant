-- Aura Platform Database Initialization Script
-- This script sets up the initial database schema for Phase 0

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create schemas for multi-tenancy
CREATE SCHEMA IF NOT EXISTS tenants;
CREATE SCHEMA IF NOT EXISTS shared;

-- Create basic tables for Phase 0
CREATE TABLE IF NOT EXISTS shared.tenants (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    domain VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT true
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_tenants_domain ON shared.tenants(domain);
CREATE INDEX IF NOT EXISTS idx_tenants_active ON shared.tenants(is_active);

-- Insert a default tenant for development
INSERT INTO shared.tenants (name, domain) 
VALUES ('Development Tenant', 'dev.aura-platform.local')
ON CONFLICT (domain) DO NOTHING;

-- Grant permissions
GRANT USAGE ON SCHEMA shared TO aura_user;
GRANT USAGE ON SCHEMA tenants TO aura_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA shared TO aura_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA shared TO aura_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA tenants TO aura_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA tenants TO aura_user;
