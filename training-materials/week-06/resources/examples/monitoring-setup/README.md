# Monitoring Setup Example

Production-grade monitoring and observability for tool systems.

## Features

✅ **Metrics Collection**
- Request counts
- Success/error rates
- Latency (avg, P95, P99)
- Throughput

✅ **Structured Logging**
- JSON formatted logs
- Correlation IDs
- Context preservation
- Log levels

✅ **Performance Tracking**
- Per-tool metrics
- Per-user metrics
- Aggregate statistics
- Historical trends

✅ **Error Monitoring**
- Error rates
- Error categories
- Stack traces
- Alerting

✅ **Alerting System**
- Threshold-based alerts
- Rate-based alerts
- Anomaly detection
- Multi-channel notifications

✅ **Dashboard Integration**
- Prometheus metrics
- Grafana dashboards
- Real-time updates

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run monitoring demo
python main.py

# With Prometheus
python main.py --prometheus

# Generate dashboard data
python main.py --dashboard
```

## Metrics Collected

- `tool_calls_total`: Total tool invocations
- `tool_calls_success`: Successful calls
- `tool_calls_failed`: Failed calls
- `tool_latency_seconds`: Execution latency
- `tool_rate_limited`: Rate limit hits
- `tool_cache_hits`: Cache hit rate

## Related Resources

- `../../function-calling-best-practices.md` (Monitoring section)
- `../../error-handling-patterns.md` (Error monitoring)
