"""
Complete Tool Registry Example - Main Entry Point

Demonstrates a production-ready tool registry with:
- Tool registration and discovery
- OpenAI schema conversion
- Authentication & authorization
- Rate limiting
- Versioning
- Monitoring
- OpenAI integration

Usage:
    python main.py                    # Run demo
    python main.py --interactive      # Interactive mode
    python main.py --use-openai       # Use real OpenAI API
"""

import asyncio
import os
import sys
from typing import Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from config import RegistryConfig, UserRole
from modules.registry import ToolRegistry
from modules.auth import AuthorizationManager
from modules.rate_limit import QuotaManager
from modules.monitoring import UsageMonitor
from modules.tools import setup_example_tools
from modules.integration import RegistryAgent


def print_section(title: str):
    """Print section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def demo_basic_registry():
    """Demonstrate basic registry functionality."""
    print_section("1. Basic Tool Registry")
    
    registry = ToolRegistry()
    
    # Setup example tools
    setup_example_tools(registry)
    
    # List all tools
    tools = registry.list_tools()
    print(f"Registered {len(tools)} tools:")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description}")
    
    # Get specific tool
    calc_tool = registry.get("calculator")
    print(f"\nCalculator tool: {calc_tool.name}")
    print(f"  Category: {calc_tool.category}")
    print(f"  Parameters: {list(calc_tool.parameters_schema.model_fields.keys())}")
    
    # Execute tool
    result = registry.execute("calculator", {
        "operation": "add",
        "a": 10,
        "b": 20
    })
    print(f"\nExecution result: {result}")
    
    return registry


def demo_openai_schemas(registry: ToolRegistry):
    """Demonstrate OpenAI schema conversion."""
    print_section("2. OpenAI Schema Conversion")
    
    # Convert single tool
    calc_schema = registry.to_openai_schema("calculator")
    print("Calculator OpenAI schema:")
    print(f"  Name: {calc_schema['name']}")
    print(f"  Description: {calc_schema['description']}")
    print(f"  Parameters: {list(calc_schema['parameters']['properties'].keys())}")
    
    # Get all schemas
    all_schemas = registry.get_all_openai_schemas()
    print(f"\nTotal OpenAI schemas: {len(all_schemas)}")
    for schema in all_schemas:
        print(f"  - {schema['name']}")


def demo_authorization(registry: ToolRegistry):
    """Demonstrate authentication and authorization."""
    print_section("3. Authentication & Authorization")
    
    auth_manager = AuthorizationManager()
    
    # Set permissions
    auth_manager.set_permissions("calculator", [UserRole.USER, UserRole.ADMIN])
    auth_manager.set_permissions("get_weather", [UserRole.USER, UserRole.ADMIN, UserRole.GUEST])
    auth_manager.set_permissions("send_email", [UserRole.ADMIN])  # Admin only
    
    # Test different users
    users = [
        {"user_id": "admin1", "role": UserRole.ADMIN},
        {"user_id": "user1", "role": UserRole.USER},
        {"user_id": "guest1", "role": UserRole.GUEST},
    ]
    
    tool_name = "calculator"
    print(f"Checking access to '{tool_name}':")
    for user in users:
        can_access = auth_manager.check_permission(user, tool_name)
        print(f"  {user['role'].value}: {'✓ Allowed' if can_access else '✗ Denied'}")
    
    # Test admin-only tool
    tool_name = "send_email"
    print(f"\nChecking access to admin-only '{tool_name}':")
    for user in users:
        can_access = auth_manager.check_permission(user, tool_name)
        print(f"  {user['role'].value}: {'✓ Allowed' if can_access else '✗ Denied'}")


def demo_rate_limiting():
    """Demonstrate rate limiting."""
    print_section("4. Rate Limiting")
    
    quota_manager = QuotaManager(
        default_quota=10,
        window_minutes=1
    )
    
    user_id = "user123"
    
    # Simulate requests
    print(f"User '{user_id}' quota: 10 requests per minute")
    print("\nSimulating requests:")
    
    for i in range(12):
        if quota_manager.check_quota(user_id, cost=1):
            quota_manager.consume_quota(user_id, cost=1)
            remaining = quota_manager.get_remaining_quota(user_id)
            print(f"  Request {i+1}: ✓ Success (remaining: {remaining})")
        else:
            print(f"  Request {i+1}: ✗ Quota exceeded")
    
    # Show quota info
    info = quota_manager.get_quota_info(user_id)
    print(f"\nQuota info:")
    print(f"  Used: {info['used']}")
    print(f"  Limit: {info['limit']}")
    print(f"  Remaining: {info['remaining']}")


def demo_monitoring(registry: ToolRegistry):
    """Demonstrate usage monitoring."""
    print_section("5. Usage Monitoring")
    
    monitor = UsageMonitor()
    
    # Simulate tool usage
    print("Simulating tool usage...")
    
    import time
    import random
    
    tools = ["calculator", "get_weather", "search_web"]
    users = ["user1", "user2", "user3"]
    
    for _ in range(20):
        tool = random.choice(tools)
        user = random.choice(users)
        success = random.random() > 0.1  # 90% success rate
        latency = random.uniform(10, 200)
        
        metric = monitor.record_execution(
            tool_name=tool,
            user_id=user,
            success=success,
            latency_ms=latency
        )
    
    # Show metrics for each tool
    print("\nTool Metrics:")
    for tool_name in tools:
        metrics = monitor.get_tool_metrics(tool_name)
        print(f"\n{tool_name}:")
        print(f"  Total calls: {metrics.total_calls}")
        print(f"  Success rate: {metrics.success_rate:.2%}")
        print(f"  Avg latency: {metrics.avg_latency_ms:.0f}ms")
        print(f"  P95 latency: {metrics.p95_latency_ms:.0f}ms")
        print(f"  Error rate: {metrics.error_rate:.2%}")


async def demo_openai_integration(registry: ToolRegistry, use_real_api: bool = False):
    """Demonstrate OpenAI integration."""
    print_section("6. OpenAI Integration")
    
    if not use_real_api:
        print("⚠️  Using mock OpenAI responses (set --use-openai for real API)")
        print("   Set OPENAI_API_KEY environment variable to use real API\n")
    
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY", "mock-key")
    
    if use_real_api and api_key == "mock-key":
        print("❌ OPENAI_API_KEY not set. Using mock mode.")
        use_real_api = False
    
    # Create agent
    agent = RegistryAgent(
        registry=registry,
        api_key=api_key,
        use_real_api=use_real_api
    )
    
    # Example queries
    queries = [
        "What's 25 + 17?",
        "What's the weather in San Francisco?",
        "Search for Python tutorials",
    ]
    
    user = {
        "user_id": "demo_user",
        "role": UserRole.USER
    }
    
    for query in queries:
        print(f"\nQuery: {query}")
        try:
            response = await agent.chat(query, user=user)
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error: {e}")


def interactive_mode(registry: ToolRegistry):
    """Run in interactive mode."""
    print_section("Interactive Mode")
    print("Type 'exit' to quit, 'help' for commands\n")
    
    user = {
        "user_id": "interactive_user",
        "role": UserRole.USER
    }
    
    while True:
        try:
            command = input("> ").strip()
            
            if command == "exit":
                break
            elif command == "help":
                print("Commands:")
                print("  list - List all tools")
                print("  info <tool> - Show tool info")
                print("  exec <tool> <json_params> - Execute tool")
                print("  schemas - Show OpenAI schemas")
                print("  exit - Exit")
            elif command == "list":
                tools = registry.list_tools()
                for tool in tools:
                    print(f"  {tool.name}: {tool.description}")
            elif command.startswith("info "):
                tool_name = command[5:].strip()
                try:
                    tool = registry.get(tool_name)
                    print(f"Name: {tool.name}")
                    print(f"Description: {tool.description}")
                    print(f"Category: {tool.category}")
                    print(f"Parameters: {list(tool.parameters_schema.model_fields.keys())}")
                except ValueError as e:
                    print(f"Error: {e}")
            elif command.startswith("exec "):
                parts = command[5:].strip().split(" ", 1)
                if len(parts) == 2:
                    tool_name, params_str = parts
                    try:
                        import json
                        params = json.loads(params_str)
                        result = registry.execute(tool_name, params, user=user)
                        print(f"Result: {result}")
                    except Exception as e:
                        print(f"Error: {e}")
                else:
                    print("Usage: exec <tool> <json_params>")
            elif command == "schemas":
                schemas = registry.get_all_openai_schemas()
                for schema in schemas:
                    print(f"  {schema['name']}")
            else:
                print("Unknown command. Type 'help' for commands.")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nGoodbye!")


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Complete Tool Registry Example")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--use-openai", action="store_true", help="Use real OpenAI API")
    args = parser.parse_args()
    
    print("""
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║          Complete Tool Registry Example                  ║
║          Week 6 - Advanced Function Calling              ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
    """)
    
    if args.interactive:
        registry = ToolRegistry()
        setup_example_tools(registry)
        interactive_mode(registry)
    else:
        # Run all demos
        registry = demo_basic_registry()
        demo_openai_schemas(registry)
        demo_authorization(registry)
        demo_rate_limiting()
        demo_monitoring(registry)
        await demo_openai_integration(registry, use_real_api=args.use_openai)
        
        print_section("Demo Complete")
        print("✓ All features demonstrated successfully!")
        print("\nTry interactive mode: python main.py --interactive")
        print("Or with OpenAI API: python main.py --use-openai")


if __name__ == "__main__":
    asyncio.run(main())
