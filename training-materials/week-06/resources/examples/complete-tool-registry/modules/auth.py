"""Authentication and authorization."""

from enum import Enum
from typing import Dict, List, Set


class UserRole(Enum):
    """User roles."""
    ADMIN = "admin"
    USER = "user"
    GUEST = "guest"


class AuthorizationManager:
    """
    Manages tool permissions and authorization.
    
    Features:
    - Role-based access control
    - Per-tool permissions
    - User context validation
    """
    
    def __init__(self):
        self._permissions: Dict[str, Set[UserRole]] = {}
        self._default_permissions = {UserRole.USER, UserRole.ADMIN}
    
    def set_permissions(self, tool_name: str, roles: List[UserRole]) -> None:
        """Set allowed roles for a tool."""
        self._permissions[tool_name] = set(roles)
    
    def get_permissions(self, tool_name: str) -> Set[UserRole]:
        """Get allowed roles for a tool."""
        return self._permissions.get(tool_name, self._default_permissions)
    
    def check_permission(self, user: dict, tool_name: str) -> bool:
        """
        Check if user has permission to use tool.
        
        Args:
            user: User context with 'role' field
            tool_name: Tool name
            
        Returns:
            True if authorized
        """
        user_role = user.get("role")
        
        # Convert string to enum if needed
        if isinstance(user_role, str):
            try:
                user_role = UserRole(user_role)
            except ValueError:
                return False
        
        allowed_roles = self.get_permissions(tool_name)
        return user_role in allowed_roles
    
    def get_accessible_tools(self, user: dict, all_tools: List[str]) -> List[str]:
        """Get list of tools user can access."""
        return [
            tool for tool in all_tools
            if self.check_permission(user, tool)
        ]
