"""Configuration for tool registry example."""

from enum import Enum


class UserRole(Enum):
    """User roles for authorization."""
    ADMIN = "admin"
    USER = "user"
    GUEST = "guest"


class RegistryConfig:
    """Configuration for tool registry."""
    
    # Rate limiting
    DEFAULT_QUOTA = 100  # Requests per window
    QUOTA_WINDOW_MINUTES = 60
    
    # Monitoring
    ENABLE_METRICS = True
    METRICS_RETENTION_HOURS = 24
    
    # Authentication
    ENABLE_AUTH = True
    DEFAULT_ROLE = UserRole.USER
    
    # Execution
    DEFAULT_TIMEOUT = 30  # seconds
    MAX_RETRIES = 3
    
    # OpenAI
    MODEL = "gpt-4"
    MAX_TOKENS = 1000
    TEMPERATURE = 0.7
