"""Usage monitoring and metrics collection."""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List
import statistics


@dataclass
class ToolUsageMetric:
    """Metrics for a tool."""
    tool_name: str
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_latency_ms: float = 0.0
    latencies: List[float] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_calls == 0:
            return 0.0
        return self.successful_calls / self.total_calls
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate."""
        return 1.0 - self.success_rate
    
    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency."""
        if self.total_calls == 0:
            return 0.0
        return self.total_latency_ms / self.total_calls
    
    @property
    def p95_latency_ms(self) -> float:
        """Calculate P95 latency."""
        if not self.latencies:
            return 0.0
        sorted_latencies = sorted(self.latencies)
        index = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[min(index, len(sorted_latencies) - 1)]


class UsageMonitor:
    """
    Monitor tool usage and collect metrics.
    
    Features:
    - Per-tool metrics
    - Success/error tracking
    - Latency measurements
    - Aggregate statistics
    """
    
    def __init__(self):
        self._metrics: Dict[str, ToolUsageMetric] = defaultdict(
            lambda: ToolUsageMetric(tool_name="")
        )
        self._user_activity: Dict[str, List[dict]] = defaultdict(list)
    
    def record_execution(
        self,
        tool_name: str,
        user_id: str,
        success: bool,
        latency_ms: float,
        error: str = None
    ) -> ToolUsageMetric:
        """
        Record a tool execution.
        
        Args:
            tool_name: Tool name
            user_id: User identifier
            success: Whether execution succeeded
            latency_ms: Execution latency
            error: Optional error message
            
        Returns:
            Updated metrics
        """
        # Update tool metrics
        if tool_name not in self._metrics:
            self._metrics[tool_name] = ToolUsageMetric(tool_name=tool_name)
        
        metric = self._metrics[tool_name]
        metric.total_calls += 1
        metric.total_latency_ms += latency_ms
        metric.latencies.append(latency_ms)
        
        if success:
            metric.successful_calls += 1
        else:
            metric.failed_calls += 1
        
        # Record user activity
        self._user_activity[user_id].append({
            "tool": tool_name,
            "timestamp": datetime.now(),
            "success": success,
            "latency_ms": latency_ms,
            "error": error
        })
        
        return metric
    
    def get_tool_metrics(self, tool_name: str) -> ToolUsageMetric:
        """Get metrics for a tool."""
        return self._metrics.get(tool_name, ToolUsageMetric(tool_name=tool_name))
    
    def get_all_metrics(self) -> Dict[str, ToolUsageMetric]:
        """Get metrics for all tools."""
        return dict(self._metrics)
    
    def get_user_activity(self, user_id: str) -> List[dict]:
        """Get activity log for user."""
        return self._user_activity.get(user_id, [])
    
    def get_summary(self) -> dict:
        """Get overall summary."""
        total_calls = sum(m.total_calls for m in self._metrics.values())
        total_success = sum(m.successful_calls for m in self._metrics.values())
        
        return {
            "total_tools": len(self._metrics),
            "total_calls": total_calls,
            "overall_success_rate": total_success / total_calls if total_calls > 0 else 0.0,
            "total_users": len(self._user_activity),
            "tools": {
                name: {
                    "calls": metric.total_calls,
                    "success_rate": metric.success_rate,
                    "avg_latency_ms": metric.avg_latency_ms
                }
                for name, metric in self._metrics.items()
            }
        }
