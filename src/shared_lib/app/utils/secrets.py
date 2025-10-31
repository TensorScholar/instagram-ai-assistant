"""
Aura Platform - Tenant Secrets Manager
Secure management of tenant-specific secrets using HashiCorp Vault.
"""

import logging
from typing import Any, Dict, Optional
from uuid import UUID

try:
    import hvac  # type: ignore
    HVAC_AVAILABLE = True
except ImportError:
    hvac = None  # type: ignore
    HVAC_AVAILABLE = False

logger = logging.getLogger(__name__)


class TenantSecretsManager:
    """
    Manages tenant-specific secrets using HashiCorp Vault.
    
    This class provides secure access to tenant-specific secrets stored in Vault,
    ensuring proper isolation and access control.
    """
    
    def __init__(
        self,
        vault_url: str,
        vault_token: Optional[str] = None,
        vault_role: Optional[str] = None,
        vault_mount_point: str = "secret",
    ):
        """
        Initialize the secrets manager.
        
        Args:
            vault_url: Vault server URL
            vault_token: Vault authentication token
            vault_role: Vault role for authentication
            vault_mount_point: Vault mount point for secrets
            
        Raises:
            ImportError: If hvac library is not available
            ValueError: If required parameters are missing
        """
        if hvac is None:
            raise ImportError("hvac library is required for Vault integration")
        
        if not vault_url:
            raise ValueError("vault_url is required")
        
        self.vault_url = vault_url
        self.vault_token = vault_token
        self.vault_role = vault_role
        self.vault_mount_point = vault_mount_point
        
        # Initialize Vault client
        self.client = hvac.Client(url=vault_url)
        
        # Authenticate if token is provided
        if vault_token:
            self.client.token = vault_token
            logger.info("Authenticated with Vault using token")
        elif vault_role:
            # Use Kubernetes authentication
            self._authenticate_with_kubernetes(vault_role)
        
        logger.info(f"TenantSecretsManager initialized with Vault at {vault_url}")
    
    def _authenticate_with_kubernetes(self, role: str) -> None:
        """
        Authenticate with Vault using Kubernetes service account.
        
        Args:
            role: Vault role for authentication
        """
        try:
            # Read Kubernetes service account token
            with open('/var/run/secrets/kubernetes.io/serviceaccount/token', 'r') as f:
                jwt_token = f.read().strip()
            
            # Authenticate with Vault
            auth_response = self.client.auth.kubernetes.login(
                role=role,
                jwt=jwt_token,
            )
            
            if auth_response and 'auth' in auth_response:
                self.client.token = auth_response['auth']['client_token']
                logger.info(f"Authenticated with Vault using Kubernetes role: {role}")
            else:
                raise ValueError("Failed to authenticate with Vault using Kubernetes")
                
        except Exception as e:
            logger.error(f"Kubernetes authentication failed: {e}")
            raise
    
    def _get_tenant_secrets_path(self, tenant_id: UUID) -> str:
        """
        Get the Vault path for tenant secrets.
        
        Args:
            tenant_id: Tenant ID
            
        Returns:
            Vault path for tenant secrets
        """
        return f"{self.vault_mount_point}/data/aura/tenants/{tenant_id}"
    
    async def get_secret(self, tenant_id: UUID, secret_key: str) -> Optional[str]:
        """
        Get a specific secret for a tenant.
        
        Args:
            tenant_id: Tenant ID
            secret_key: Secret key to retrieve
            
        Returns:
            Secret value or None if not found
            
        Raises:
            Exception: If Vault operation fails
        """
        try:
            secrets_path = self._get_tenant_secrets_path(tenant_id)
            
            # Read secret from Vault
            response = self.client.secrets.kv.v2.read_secret_version(
                path=secrets_path,
                mount_point=self.vault_mount_point,
            )
            
            if response and 'data' in response and 'data' in response['data']:
                secrets = response['data']['data']
                secret_value = secrets.get(secret_key)
                
                if secret_value:
                    logger.debug(f"Retrieved secret {secret_key} for tenant {tenant_id}")
                    return secret_value
                else:
                    logger.warning(f"Secret {secret_key} not found for tenant {tenant_id}")
                    return None
            else:
                logger.warning(f"No secrets found for tenant {tenant_id}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to retrieve secret {secret_key} for tenant {tenant_id}: {e}")
            raise
    
    async def get_all_secrets(self, tenant_id: UUID) -> Dict[str, str]:
        """
        Get all secrets for a tenant.
        
        Args:
            tenant_id: Tenant ID
            
        Returns:
            Dictionary of all secrets for the tenant
            
        Raises:
            Exception: If Vault operation fails
        """
        try:
            secrets_path = self._get_tenant_secrets_path(tenant_id)
            
            # Read all secrets from Vault
            response = self.client.secrets.kv.v2.read_secret_version(
                path=secrets_path,
                mount_point=self.vault_mount_point,
            )
            
            if response and 'data' in response and 'data' in response['data']:
                secrets = response['data']['data']
                logger.debug(f"Retrieved {len(secrets)} secrets for tenant {tenant_id}")
                return secrets
            else:
                logger.warning(f"No secrets found for tenant {tenant_id}")
                return {}
                
        except Exception as e:
            logger.error(f"Failed to retrieve secrets for tenant {tenant_id}: {e}")
            raise
    
    async def set_secret(self, tenant_id: UUID, secret_key: str, secret_value: str) -> bool:
        """
        Set a specific secret for a tenant.
        
        Args:
            tenant_id: Tenant ID
            secret_key: Secret key to set
            secret_value: Secret value to store
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            Exception: If Vault operation fails
        """
        try:
            secrets_path = self._get_tenant_secrets_path(tenant_id)
            
            # Get existing secrets
            existing_secrets = await self.get_all_secrets(tenant_id)
            
            # Update with new secret
            existing_secrets[secret_key] = secret_value
            
            # Write secrets to Vault
            self.client.secrets.kv.v2.create_or_update_secret(
                path=secrets_path,
                secret=existing_secrets,
                mount_point=self.vault_mount_point,
            )
            
            logger.info(f"Set secret {secret_key} for tenant {tenant_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set secret {secret_key} for tenant {tenant_id}: {e}")
            raise
    
    async def delete_secret(self, tenant_id: UUID, secret_key: str) -> bool:
        """
        Delete a specific secret for a tenant.
        
        Args:
            tenant_id: Tenant ID
            secret_key: Secret key to delete
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            Exception: If Vault operation fails
        """
        try:
            secrets_path = self._get_tenant_secrets_path(tenant_id)
            
            # Get existing secrets
            existing_secrets = await self.get_all_secrets(tenant_id)
            
            # Remove the secret
            if secret_key in existing_secrets:
                del existing_secrets[secret_key]
                
                # Write updated secrets to Vault
                self.client.secrets.kv.v2.create_or_update_secret(
                    path=secrets_path,
                    secret=existing_secrets,
                    mount_point=self.vault_mount_point,
                )
                
                logger.info(f"Deleted secret {secret_key} for tenant {tenant_id}")
                return True
            else:
                logger.warning(f"Secret {secret_key} not found for tenant {tenant_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete secret {secret_key} for tenant {tenant_id}: {e}")
            raise
    
    async def delete_all_secrets(self, tenant_id: UUID) -> bool:
        """
        Delete all secrets for a tenant.
        
        Args:
            tenant_id: Tenant ID
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            Exception: If Vault operation fails
        """
        try:
            secrets_path = self._get_tenant_secrets_path(tenant_id)
            
            # Delete the entire secret path
            self.client.secrets.kv.v2.delete_metadata_and_all_versions(
                path=secrets_path,
                mount_point=self.vault_mount_point,
            )
            
            logger.info(f"Deleted all secrets for tenant {tenant_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete all secrets for tenant {tenant_id}: {e}")
            raise
    
    async def list_secret_keys(self, tenant_id: UUID) -> list[str]:
        """
        List all secret keys for a tenant.
        
        Args:
            tenant_id: Tenant ID
            
        Returns:
            List of secret keys
            
        Raises:
            Exception: If Vault operation fails
        """
        try:
            secrets = await self.get_all_secrets(tenant_id)
            return list(secrets.keys())
            
        except Exception as e:
            logger.error(f"Failed to list secret keys for tenant {tenant_id}: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check the health of the secrets manager.
        
        Returns:
            Health status information
        """
        try:
            # Check Vault connectivity
            health_response = self.client.sys.read_health_status()
            
            return {
                "status": "healthy",
                "vault_url": self.vault_url,
                "vault_health": health_response,
                "authenticated": self.client.is_authenticated(),
                "mount_point": self.vault_mount_point,
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "vault_url": self.vault_url,
                "mount_point": self.vault_mount_point,
            }


# Factory function for creating secrets manager
def create_secrets_manager(
    vault_url: str,
    vault_token: Optional[str] = None,
    vault_role: Optional[str] = None,
) -> TenantSecretsManager:
    """
    Create a TenantSecretsManager instance.
    
    Args:
        vault_url: Vault server URL
        vault_token: Vault authentication token
        vault_role: Vault role for authentication
        
    Returns:
        TenantSecretsManager instance
        
    Raises:
        ImportError: If hvac library is not available
        ValueError: If required parameters are missing
    """
    return TenantSecretsManager(
        vault_url=vault_url,
        vault_token=vault_token,
        vault_role=vault_role,
    )
