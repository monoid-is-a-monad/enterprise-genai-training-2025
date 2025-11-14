"""Core tool registry implementation."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type

from pydantic import BaseModel, ValidationError


class ToolCategory(Enum):
    """Tool categories."""
    UTILITY = "utility"
    DATA = "data"
    COMMUNICATION = "communication"
    ANALYSIS = "analysis"
    OTHER = "other"


@dataclass
class ToolMetadata:
    """Metadata for a registered tool."""
    name: str
    description: str
    func: Callable
    category: ToolCategory
    parameters_schema: Type[BaseModel]
    version: str = "1.0.0"
    author: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


class ToolRegistry:
    """
    Central registry for managing tools.
    
    Features:
    - Register and discover tools
    - Convert to OpenAI schemas
    - Execute tools with validation
    - Version management
    """
    
    def __init__(self):
        self._tools: Dict[str, ToolMetadata] = {}
    
    def register(
        self,
        name: str,
        func: Callable,
        description: str,
        schema: Type[BaseModel],
        category: str = "other",
        version: str = "1.0.0",
        author: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> None:
        """
        Register a tool.
        
        Args:
            name: Unique tool name
            func: Callable function
            description: Tool description
            schema: Pydantic model for parameters
            category: Tool category
            version: Semantic version
            author: Tool author
            tags: Optional tags
        """
        if name in self._tools:
            raise ValueError(f"Tool '{name}' already registered")
        
        # Validate category
        try:
            cat = ToolCategory(category)
        except ValueError:
            cat = ToolCategory.OTHER
        
        # Create metadata
        metadata = ToolMetadata(
            name=name,
            description=description,
            func=func,
            category=cat,
            parameters_schema=schema,
            version=version,
            author=author,
            tags=tags or []
        )
        
        self._tools[name] = metadata
    
    def get(self, name: str) -> ToolMetadata:
        """Get tool by name."""
        if name not in self._tools:
            raise ValueError(f"Tool '{name}' not found")
        return self._tools[name]
    
    def has_tool(self, name: str) -> bool:
        """Check if tool exists."""
        return name in self._tools
    
    def list_tools(
        self,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[ToolMetadata]:
        """
        List all tools, optionally filtered.
        
        Args:
            category: Filter by category
            tags: Filter by tags
        """
        tools = list(self._tools.values())
        
        if category:
            cat = ToolCategory(category)
            tools = [t for t in tools if t.category == cat]
        
        if tags:
            tools = [t for t in tools if any(tag in t.tags for tag in tags)]
        
        return tools
    
    def execute(
        self,
        name: str,
        params: Dict[str, Any],
        user: Optional[dict] = None
    ) -> Any:
        """
        Execute a tool with validation.
        
        Args:
            name: Tool name
            params: Parameters dictionary
            user: Optional user context
            
        Returns:
            Tool execution result
        """
        tool = self.get(name)
        
        # Validate parameters
        try:
            validated_params = tool.parameters_schema(**params)
        except ValidationError as e:
            return {
                "success": False,
                "error": f"Validation error: {e}"
            }
        
        # Execute tool
        try:
            result = tool.func(**validated_params.model_dump())
            return {
                "success": True,
                "result": result,
                "tool": name
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "tool": name
            }
    
    def to_openai_schema(self, name: str) -> dict:
        """
        Convert tool to OpenAI function schema.
        
        Args:
            name: Tool name
            
        Returns:
            OpenAI function schema
        """
        tool = self.get(name)
        
        # Get Pydantic schema
        pydantic_schema = tool.parameters_schema.model_json_schema()
        
        # Convert to OpenAI format
        openai_schema = {
            "name": name,
            "description": tool.description,
            "parameters": {
                "type": "object",
                "properties": pydantic_schema.get("properties", {}),
                "required": pydantic_schema.get("required", [])
            }
        }
        
        return openai_schema
    
    def get_all_openai_schemas(self) -> List[dict]:
        """Get OpenAI schemas for all tools."""
        return [
            self.to_openai_schema(name)
            for name in self._tools.keys()
        ]
    
    def unregister(self, name: str) -> None:
        """Unregister a tool."""
        if name in self._tools:
            del self._tools[name]
