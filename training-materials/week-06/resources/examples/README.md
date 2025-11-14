# Week 6 Examples - Runnable Code

This directory contains complete, production-ready examples demonstrating advanced function calling patterns.

## Examples

### 1. [complete-tool-registry/](./complete-tool-registry/)
**Full-featured tool registry implementation**

A production-ready tool registry system with:
- Tool registration and management
- OpenAI schema conversion
- Pydantic validation
- Authentication & authorization
- Rate limiting
- Versioning
- Usage monitoring
- Complete integration with OpenAI

**Run it**: `python complete-tool-registry/main.py`

### 2. [production-workflow/](./production-workflow/)
**Complete workflow orchestration system**

Real-world workflow examples including:
- Sequential chains with parameter resolution
- Parallel execution with fork-join
- Conditional workflows
- DAG workflows with visualization
- Saga pattern with compensation
- Workflow monitoring and observability

**Run it**: `python production-workflow/main.py`

### 3. [monitoring-setup/](./monitoring-setup/)
**Production monitoring and observability**

Complete monitoring solution with:
- Metrics collection and aggregation
- Structured logging
- Performance tracking
- Error rate monitoring
- Alerting system
- Dashboard data generation

**Run it**: `python monitoring-setup/main.py`

### 4. [error-handling-demo/](./error-handling-demo/)
**Comprehensive error handling**

Demonstrates production error handling:
- Retry strategies with exponential backoff
- Circuit breakers
- Fallback chains
- Timeout handling
- Saga pattern compensation
- Error recovery patterns

**Run it**: `python error-handling-demo/main.py`

## Setup

Each example has its own `requirements.txt`. Install dependencies:

```bash
# Install all dependencies
cd examples
pip install -r requirements.txt

# Or install for specific example
cd complete-tool-registry
pip install -r requirements.txt
```

## Environment Variables

Some examples require OpenAI API access:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

For examples that don't need actual API calls, they use mock implementations.

## Running Examples

### Quick Start

Run all examples to see them in action:

```bash
# Tool Registry
python complete-tool-registry/main.py

# Workflows
python production-workflow/main.py

# Monitoring
python monitoring-setup/main.py

# Error Handling
python error-handling-demo/main.py
```

### Interactive Mode

Some examples support interactive mode:

```bash
# Interactive tool registry
python complete-tool-registry/main.py --interactive

# Interactive workflow builder
python production-workflow/main.py --interactive
```

### With Real OpenAI API

Enable actual OpenAI API calls:

```bash
export OPENAI_API_KEY="sk-..."
python complete-tool-registry/main.py --use-openai
```

## Example Structure

Each example follows this structure:

```
example-name/
├── README.md           # Detailed documentation
├── requirements.txt    # Dependencies
├── main.py            # Entry point
├── config.py          # Configuration
└── modules/           # Implementation modules
    ├── core.py        # Core functionality
    ├── monitoring.py  # Monitoring code
    └── utils.py       # Utilities
```

## Learning Path

1. **Start with Tool Registry** (`complete-tool-registry/`)
   - Understand tool registration and management
   - Learn OpenAI schema conversion
   - See authentication and rate limiting in action

2. **Move to Workflows** (`production-workflow/`)
   - Build on tool registry knowledge
   - Learn workflow orchestration patterns
   - Understand dependency management

3. **Add Monitoring** (`monitoring-setup/`)
   - Instrument your workflows
   - Track metrics and performance
   - Set up alerting

4. **Master Error Handling** (`error-handling-demo/`)
   - Implement resilient systems
   - Use retry and fallback strategies
   - Handle failures gracefully

## Customization

All examples are designed to be easily customized:

```python
# Customize tool registry
from complete_tool_registry.config import Config

config = Config(
    max_tools=200,
    enable_auth=True,
    rate_limit_per_user=1000
)

# Customize workflow engine
from production_workflow.config import WorkflowConfig

workflow_config = WorkflowConfig(
    max_parallel_tasks=10,
    timeout_seconds=30,
    enable_monitoring=True
)
```

## Testing

Each example includes tests:

```bash
# Run tests for specific example
cd complete-tool-registry
pytest

# Run all tests
pytest examples/
```

## Performance

Performance characteristics:

| Example | Typical Runtime | Memory Usage | API Calls |
|---------|----------------|--------------|-----------|
| Tool Registry | < 1s | 50MB | 0-5 |
| Workflows | 2-5s | 100MB | 5-15 |
| Monitoring | < 1s | 30MB | 0 |
| Error Handling | 5-10s | 80MB | 10-20 |

## Troubleshooting

### Import Errors

Make sure you're in the right directory:

```bash
# From examples/ directory
python complete-tool-registry/main.py

# Or use module syntax
python -m complete-tool-registry.main
```

### OpenAI API Errors

If using real OpenAI API:

1. Check API key is set: `echo $OPENAI_API_KEY`
2. Verify API key is valid
3. Check rate limits

Most examples work without API key using mocks.

### Dependencies

If you get import errors:

```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Check Python version (requires 3.9+)
python --version
```

## Production Deployment

These examples are production-ready but consider:

1. **Environment Variables**: Use proper secret management
2. **Monitoring**: Integrate with your observability stack
3. **Error Handling**: Adjust retry/timeout for your use case
4. **Rate Limiting**: Configure based on your quotas
5. **Authentication**: Implement your auth system
6. **Logging**: Configure for your log aggregation

## Related Resources

- `../function-calling-best-practices.md` — Best practices
- `../tool-schema-design-guide.md` — Schema design
- `../error-handling-patterns.md` — Error handling
- `../workflow-patterns-cheatsheet.md` — Workflow patterns

## Contributing

To add a new example:

1. Create directory: `examples/new-example/`
2. Add README.md with description
3. Include requirements.txt
4. Write main.py with clear examples
5. Add tests
6. Update this README

## License

These examples are part of the Enterprise GenAI Training materials.
