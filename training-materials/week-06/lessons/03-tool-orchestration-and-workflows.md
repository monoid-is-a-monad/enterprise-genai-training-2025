# Week 6 - Lesson 3: Tool Orchestration & Workflows

**Duration:** 2 hours  
**Level:** Advanced  
**Prerequisites:** Lessons 1-2, Understanding of state machines

---

## 📋 Table of Contents

1. [Introduction](#introduction)
2. [Tool Chaining Patterns](#tool-chaining-patterns)
3. [Conditional Execution](#conditional-execution)
4. [State Management](#state-management)
5. [Workflow Orchestration](#workflow-orchestration)
6. [Error Recovery](#error-recovery)
7. [Workflow Monitoring](#workflow-monitoring)
8. [Advanced Patterns](#advanced-patterns)

---

## Introduction

### Lesson Objectives

By the end of this lesson, you will be able to:
- Chain multiple tools together in sequences
- Implement conditional tool execution based on results
- Manage workflow state across tool calls
- Build complex orchestration patterns
- Recover from failures in multi-step workflows
- Monitor and visualize workflow execution
- Apply advanced patterns like DAGs and sagas

### Why Tool Orchestration?

Real-world tasks often require multiple tools working together:

```
User Query: "Find tech news about AI, summarize the top 3, and email me"

Workflow:
1. search_web("AI technology news")
2. summarize_text(article_1, article_2, article_3)
3. send_email(recipient, summary)
```

**Challenges:**
- Tools have dependencies (output → input)
- Failures in one step affect downstream steps
- State must be maintained across calls
- Parallel execution where possible

---

## Tool Chaining Patterns

### Sequential Chaining

```python
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json

@dataclass
class ToolCall:
    """Represents a single tool call in a chain."""
    tool_name: str
    parameters: Dict[str, Any]
    output_key: Optional[str] = None  # Store result under this key

@dataclass
class ChainResult:
    """Result of a tool chain execution."""
    success: bool
    outputs: Dict[str, Any]
    error: Optional[str] = None

class ToolChain:
    """
    Execute tools in sequence, passing outputs to next tool.
    """
    
    def __init__(self, registry):
        self.registry = registry
        self.context: Dict[str, Any] = {}
    
    def execute_chain(self, chain: List[ToolCall]) -> ChainResult:
        """
        Execute a chain of tool calls sequentially.
        
        Args:
            chain: List of ToolCall objects
        
        Returns:
            ChainResult with all outputs
        """
        self.context = {}
        
        for i, tool_call in enumerate(chain):
            print(f"Step {i+1}/{len(chain)}: {tool_call.tool_name}")
            
            try:
                # Resolve parameters from context
                resolved_params = self._resolve_parameters(tool_call.parameters)
                
                # Execute tool
                result = self.registry.execute(
                    tool_call.tool_name,
                    **resolved_params
                )
                
                # Store result in context
                if tool_call.output_key:
                    self.context[tool_call.output_key] = result
                else:
                    self.context[f"step_{i}"] = result
                
            except Exception as e:
                return ChainResult(
                    success=False,
                    outputs=self.context.copy(),
                    error=f"Failed at step {i+1} ({tool_call.tool_name}): {e}"
                )
        
        return ChainResult(
            success=True,
            outputs=self.context.copy()
        )
    
    def _resolve_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve parameter values from context.
        
        Parameters can reference context using {key} syntax:
        {"location": "{user_location}", "unit": "celsius"}
        """
        resolved = {}
        
        for key, value in params.items():
            if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
                # Extract context key
                context_key = value[1:-1]
                if context_key in self.context:
                    resolved[key] = self.context[context_key]
                else:
                    raise ValueError(f"Context key '{context_key}' not found")
            else:
                resolved[key] = value
        
        return resolved

# Example: Weather → Outfit recommendation chain
def get_weather(location: str) -> dict:
    return {"location": location, "temp": 18, "condition": "rainy"}

def recommend_outfit(temperature: int, condition: str) -> dict:
    if condition == "rainy":
        return {"outfit": "Raincoat, umbrella, boots"}
    elif temperature < 15:
        return {"outfit": "Jacket, long pants"}
    else:
        return {"outfit": "T-shirt, shorts"}

# Register tools
from pydantic import BaseModel

class WeatherParams(BaseModel):
    location: str

class OutfitParams(BaseModel):
    temperature: int
    condition: str

registry.register(
    name="get_weather",
    description="Get weather",
    category=ToolCategory.DATA,
    version="1.0.0",
    parameters_schema=WeatherParams,
    function=get_weather
)

registry.register(
    name="recommend_outfit",
    description="Recommend outfit based on weather",
    category=ToolCategory.COMPUTATION,
    version="1.0.0",
    parameters_schema=OutfitParams,
    function=recommend_outfit
)

# Create chain
chain = ToolChain(registry)

workflow = [
    ToolCall(
        tool_name="get_weather",
        parameters={"location": "London"},
        output_key="weather"
    ),
    ToolCall(
        tool_name="recommend_outfit",
        parameters={
            "temperature": "{weather[temp]}",
            "condition": "{weather[condition]}"
        },
        output_key="outfit"
    )
]

result = chain.execute_chain(workflow)
print(f"Success: {result.success}")
print(f"Outfit: {result.outputs['outfit']}")
```

### Parallel Chaining with Dependencies

```python
from typing import Set
from collections import defaultdict

@dataclass
class WorkflowStep:
    """A step in a workflow with dependencies."""
    id: str
    tool_name: str
    parameters: Dict[str, Any]
    depends_on: Set[str] = None  # IDs of steps this depends on
    
    def __post_init__(self):
        if self.depends_on is None:
            self.depends_on = set()

class WorkflowExecutor:
    """
    Execute workflow steps with dependency management.
    Allows parallel execution where possible.
    """
    
    def __init__(self, registry):
        self.registry = registry
        self.context: Dict[str, Any] = {}
    
    def execute_workflow(self, steps: List[WorkflowStep]) -> ChainResult:
        """
        Execute workflow with dependency-aware parallelization.
        """
        # Build dependency graph
        step_map = {step.id: step for step in steps}
        completed = set()
        self.context = {}
        
        while len(completed) < len(steps):
            # Find steps ready to execute
            ready_steps = [
                step for step in steps
                if step.id not in completed
                and step.depends_on.issubset(completed)
            ]
            
            if not ready_steps:
                # No progress possible - circular dependency or error
                incomplete = set(step_map.keys()) - completed
                return ChainResult(
                    success=False,
                    outputs=self.context.copy(),
                    error=f"Cannot complete steps: {incomplete}"
                )
            
            # Execute ready steps in parallel
            for step in ready_steps:
                try:
                    resolved_params = self._resolve_parameters(step.parameters)
                    result = self.registry.execute(
                        step.tool_name,
                        **resolved_params
                    )
                    self.context[step.id] = result
                    completed.add(step.id)
                except Exception as e:
                    return ChainResult(
                        success=False,
                        outputs=self.context.copy(),
                        error=f"Step '{step.id}' failed: {e}"
                    )
        
        return ChainResult(
            success=True,
            outputs=self.context.copy()
        )
    
    def _resolve_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve parameters from context."""
        resolved = {}
        for key, value in params.items():
            if isinstance(value, str) and value.startswith("$"):
                # Reference to step output: $step_id.field
                parts = value[1:].split(".", 1)
                step_id = parts[0]
                
                if step_id not in self.context:
                    raise ValueError(f"Step '{step_id}' not executed yet")
                
                step_output = self.context[step_id]
                
                if len(parts) == 2:
                    field = parts[1]
                    resolved[key] = step_output.get(field)
                else:
                    resolved[key] = step_output
            else:
                resolved[key] = value
        
        return resolved

# Example: Parallel data gathering
workflow = [
    WorkflowStep(
        id="weather",
        tool_name="get_weather",
        parameters={"location": "London"}
    ),
    WorkflowStep(
        id="stock",
        tool_name="get_stock_price",
        parameters={"symbol": "AAPL"}
    ),
    WorkflowStep(
        id="news",
        tool_name="search_web",
        parameters={"query": "technology news"}
    ),
    WorkflowStep(
        id="summary",
        tool_name="summarize_data",
        parameters={
            "weather": "$weather",
            "stock": "$stock.price",
            "news": "$news"
        },
        depends_on={"weather", "stock", "news"}
    )
]

executor = WorkflowExecutor(registry)
result = executor.execute_workflow(workflow)
```

---

## Conditional Execution

### Rule-Based Conditions

```python
from typing import Callable
from enum import Enum

class ConditionOperator(str, Enum):
    EQUALS = "=="
    NOT_EQUALS = "!="
    GREATER_THAN = ">"
    LESS_THAN = "<"
    CONTAINS = "contains"
    IS_NONE = "is_none"

@dataclass
class Condition:
    """A condition to evaluate."""
    field: str  # Field to check (e.g., "weather.temp")
    operator: ConditionOperator
    value: Any = None
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate condition against context."""
        # Extract field value from context
        field_value = self._get_field_value(context)
        
        if self.operator == ConditionOperator.EQUALS:
            return field_value == self.value
        elif self.operator == ConditionOperator.NOT_EQUALS:
            return field_value != self.value
        elif self.operator == ConditionOperator.GREATER_THAN:
            return field_value > self.value
        elif self.operator == ConditionOperator.LESS_THAN:
            return field_value < self.value
        elif self.operator == ConditionOperator.CONTAINS:
            return self.value in field_value
        elif self.operator == ConditionOperator.IS_NONE:
            return field_value is None
        
        return False
    
    def _get_field_value(self, context: Dict[str, Any]) -> Any:
        """Extract nested field value."""
        parts = self.field.split(".")
        value = context
        
        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            else:
                return None
        
        return value

@dataclass
class ConditionalStep:
    """A workflow step with conditional execution."""
    id: str
    tool_name: str
    parameters: Dict[str, Any]
    condition: Optional[Condition] = None
    depends_on: Set[str] = None
    
    def should_execute(self, context: Dict[str, Any]) -> bool:
        """Check if step should execute based on condition."""
        if self.condition is None:
            return True
        return self.condition.evaluate(context)

class ConditionalWorkflowExecutor:
    """Execute workflow with conditional steps."""
    
    def __init__(self, registry):
        self.registry = registry
        self.context: Dict[str, Any] = {}
    
    def execute_workflow(self, steps: List[ConditionalStep]) -> ChainResult:
        """Execute workflow, skipping steps that don't meet conditions."""
        step_map = {step.id: step for step in steps}
        completed = set()
        skipped = set()
        self.context = {}
        
        while len(completed) + len(skipped) < len(steps):
            ready_steps = [
                step for step in steps
                if step.id not in completed
                and step.id not in skipped
                and (step.depends_on or set()).issubset(completed | skipped)
            ]
            
            if not ready_steps:
                incomplete = set(step_map.keys()) - completed - skipped
                return ChainResult(
                    success=False,
                    outputs=self.context.copy(),
                    error=f"Cannot complete steps: {incomplete}"
                )
            
            for step in ready_steps:
                # Check condition
                if not step.should_execute(self.context):
                    print(f"Skipping step '{step.id}' (condition not met)")
                    skipped.add(step.id)
                    self.context[step.id] = {"skipped": True}
                    continue
                
                try:
                    resolved_params = self._resolve_parameters(step.parameters)
                    result = self.registry.execute(
                        step.tool_name,
                        **resolved_params
                    )
                    self.context[step.id] = result
                    completed.add(step.id)
                except Exception as e:
                    return ChainResult(
                        success=False,
                        outputs=self.context.copy(),
                        error=f"Step '{step.id}' failed: {e}"
                    )
        
        return ChainResult(
            success=True,
            outputs=self.context.copy()
        )
    
    def _resolve_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve parameters from context."""
        resolved = {}
        for key, value in params.items():
            if isinstance(value, str) and value.startswith("$"):
                parts = value[1:].split(".", 1)
                step_id = parts[0]
                step_output = self.context.get(step_id, {})
                
                if len(parts) == 2:
                    resolved[key] = step_output.get(parts[1])
                else:
                    resolved[key] = step_output
            else:
                resolved[key] = value
        
        return resolved

# Example: Send alert only if temperature is extreme
workflow = [
    ConditionalStep(
        id="weather",
        tool_name="get_weather",
        parameters={"location": "London"}
    ),
    ConditionalStep(
        id="cold_alert",
        tool_name="send_alert",
        parameters={
            "message": "Temperature is below freezing!",
            "severity": "high"
        },
        condition=Condition(
            field="weather.temp",
            operator=ConditionOperator.LESS_THAN,
            value=0
        ),
        depends_on={"weather"}
    ),
    ConditionalStep(
        id="hot_alert",
        tool_name="send_alert",
        parameters={
            "message": "Temperature is very high!",
            "severity": "high"
        },
        condition=Condition(
            field="weather.temp",
            operator=ConditionOperator.GREATER_THAN,
            value=35
        ),
        depends_on={"weather"}
    )
]

executor = ConditionalWorkflowExecutor(registry)
result = executor.execute_workflow(workflow)
```

---

## State Management

### Workflow State Machine

```python
from enum import Enum
from typing import Any, Dict, List
from datetime import datetime

class WorkflowStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class WorkflowState:
    """Complete state of a workflow execution."""
    workflow_id: str
    status: WorkflowStatus
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    context: Dict[str, Any] = None
    completed_steps: Set[str] = None
    failed_step: Optional[str] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}
        if self.completed_steps is None:
            self.completed_steps = set()

class StatefulWorkflowExecutor:
    """
    Workflow executor with persistent state.
    Supports pause/resume and failure recovery.
    """
    
    def __init__(self, registry):
        self.registry = registry
        self.workflows: Dict[str, WorkflowState] = {}
    
    def start_workflow(
        self,
        workflow_id: str,
        steps: List[WorkflowStep]
    ) -> WorkflowState:
        """
        Start a new workflow or resume existing one.
        """
        # Check if workflow exists
        if workflow_id in self.workflows:
            state = self.workflows[workflow_id]
            if state.status == WorkflowStatus.RUNNING:
                raise ValueError(f"Workflow '{workflow_id}' already running")
            elif state.status == WorkflowStatus.COMPLETED:
                raise ValueError(f"Workflow '{workflow_id}' already completed")
        else:
            # Create new workflow state
            state = WorkflowState(
                workflow_id=workflow_id,
                status=WorkflowStatus.PENDING
            )
            self.workflows[workflow_id] = state
        
        # Start/resume execution
        state.status = WorkflowStatus.RUNNING
        state.started_at = datetime.utcnow()
        
        try:
            self._execute_workflow(workflow_id, steps)
            state.status = WorkflowStatus.COMPLETED
            state.completed_at = datetime.utcnow()
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error = str(e)
            raise
        
        return state
    
    def _execute_workflow(self, workflow_id: str, steps: List[WorkflowStep]):
        """Execute workflow with state tracking."""
        state = self.workflows[workflow_id]
        step_map = {step.id: step for step in steps}
        
        while len(state.completed_steps) < len(steps):
            # Find ready steps (excluding already completed)
            ready_steps = [
                step for step in steps
                if step.id not in state.completed_steps
                and step.depends_on.issubset(state.completed_steps)
            ]
            
            if not ready_steps:
                incomplete = set(step_map.keys()) - state.completed_steps
                raise Exception(f"Cannot complete steps: {incomplete}")
            
            for step in ready_steps:
                try:
                    # Resolve parameters
                    resolved_params = self._resolve_parameters(
                        step.parameters,
                        state.context
                    )
                    
                    # Execute tool
                    result = self.registry.execute(
                        step.tool_name,
                        **resolved_params
                    )
                    
                    # Update state
                    state.context[step.id] = result
                    state.completed_steps.add(step.id)
                    
                    # Persist state (in production, save to database)
                    self._save_state(workflow_id)
                    
                except Exception as e:
                    state.failed_step = step.id
                    raise Exception(f"Step '{step.id}' failed: {e}")
    
    def _resolve_parameters(
        self,
        params: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Resolve parameters from context."""
        resolved = {}
        for key, value in params.items():
            if isinstance(value, str) and value.startswith("$"):
                parts = value[1:].split(".", 1)
                step_id = parts[0]
                step_output = context.get(step_id, {})
                
                if len(parts) == 2:
                    resolved[key] = step_output.get(parts[1])
                else:
                    resolved[key] = step_output
            else:
                resolved[key] = value
        
        return resolved
    
    def _save_state(self, workflow_id: str):
        """
        Persist workflow state.
        In production, save to database or Redis.
        """
        # Placeholder for state persistence
        pass
    
    def get_state(self, workflow_id: str) -> Optional[WorkflowState]:
        """Get current state of a workflow."""
        return self.workflows.get(workflow_id)
    
    def cancel_workflow(self, workflow_id: str):
        """Cancel a running workflow."""
        state = self.workflows.get(workflow_id)
        if not state:
            raise ValueError(f"Workflow '{workflow_id}' not found")
        
        if state.status == WorkflowStatus.RUNNING:
            state.status = WorkflowStatus.CANCELLED
            state.completed_at = datetime.utcnow()

# Example: Long-running workflow with state persistence
executor = StatefulWorkflowExecutor(registry)

workflow_steps = [
    WorkflowStep(id="step1", tool_name="process_data", parameters={}),
    WorkflowStep(id="step2", tool_name="analyze", parameters={"data": "$step1"}, depends_on={"step1"}),
    WorkflowStep(id="step3", tool_name="generate_report", parameters={"analysis": "$step2"}, depends_on={"step2"}),
]

# Start workflow
state = executor.start_workflow("workflow-123", workflow_steps)

# Check state later
state = executor.get_state("workflow-123")
print(f"Status: {state.status}")
print(f"Completed steps: {state.completed_steps}")
```

---

## Workflow Orchestration

### DAG (Directed Acyclic Graph) Execution

```python
from typing import Set, Dict, List
import networkx as nx

class DAGWorkflow:
    """
    Workflow represented as a Directed Acyclic Graph.
    """
    
    def __init__(self, registry):
        self.registry = registry
        self.graph = nx.DiGraph()
        self.steps: Dict[str, WorkflowStep] = {}
    
    def add_step(self, step: WorkflowStep):
        """Add a step to the workflow."""
        self.steps[step.id] = step
        self.graph.add_node(step.id)
        
        # Add edges for dependencies
        for dep in step.depends_on:
            self.graph.add_edge(dep, step.id)
    
    def validate(self) -> bool:
        """
        Validate workflow is a valid DAG.
        
        Returns:
            True if valid, raises exception otherwise
        """
        if not nx.is_directed_acyclic_graph(self.graph):
            cycles = list(nx.simple_cycles(self.graph))
            raise ValueError(f"Workflow contains cycles: {cycles}")
        
        return True
    
    def get_execution_order(self) -> List[List[str]]:
        """
        Get execution order as list of parallel batches.
        
        Returns:
            List of lists, where each inner list contains step IDs
            that can be executed in parallel.
        """
        # Topological sort with levels
        levels = []
        in_degree = dict(self.graph.in_degree())
        
        while in_degree:
            # Find nodes with no dependencies
            ready = [node for node, degree in in_degree.items() if degree == 0]
            
            if not ready:
                raise ValueError("Circular dependency detected")
            
            levels.append(ready)
            
            # Remove ready nodes and update in-degrees
            for node in ready:
                del in_degree[node]
                for successor in self.graph.successors(node):
                    if successor in in_degree:
                        in_degree[successor] -= 1
        
        return levels
    
    def execute(self) -> ChainResult:
        """Execute workflow in optimal parallel order."""
        self.validate()
        
        execution_order = self.get_execution_order()
        context = {}
        
        for batch_num, batch in enumerate(execution_order):
            print(f"Executing batch {batch_num + 1}: {batch}")
            
            # Execute all steps in batch (potentially in parallel)
            for step_id in batch:
                step = self.steps[step_id]
                
                try:
                    # Resolve parameters
                    resolved_params = {}
                    for key, value in step.parameters.items():
                        if isinstance(value, str) and value.startswith("$"):
                            ref_id = value[1:].split(".")[0]
                            resolved_params[key] = context.get(ref_id)
                        else:
                            resolved_params[key] = value
                    
                    # Execute
                    result = self.registry.execute(
                        step.tool_name,
                        **resolved_params
                    )
                    context[step_id] = result
                    
                except Exception as e:
                    return ChainResult(
                        success=False,
                        outputs=context.copy(),
                        error=f"Step '{step_id}' failed: {e}"
                    )
        
        return ChainResult(
            success=True,
            outputs=context.copy()
        )
    
    def visualize(self) -> str:
        """
        Generate Mermaid diagram of workflow.
        """
        mermaid = ["graph TD"]
        
        for node in self.graph.nodes():
            step = self.steps[node]
            mermaid.append(f"    {node}[{step.tool_name}]")
        
        for edge in self.graph.edges():
            mermaid.append(f"    {edge[0]} --> {edge[1]}")
        
        return "\n".join(mermaid)

# Example: Complex data pipeline
workflow = DAGWorkflow(registry)

workflow.add_step(WorkflowStep(
    id="fetch_data",
    tool_name="fetch_api_data",
    parameters={"endpoint": "/data"},
    depends_on=set()
))

workflow.add_step(WorkflowStep(
    id="validate",
    tool_name="validate_data",
    parameters={"data": "$fetch_data"},
    depends_on={"fetch_data"}
))

workflow.add_step(WorkflowStep(
    id="transform",
    tool_name="transform_data",
    parameters={"data": "$validate"},
    depends_on={"validate"}
))

workflow.add_step(WorkflowStep(
    id="analyze_a",
    tool_name="analyze_method_a",
    parameters={"data": "$transform"},
    depends_on={"transform"}
))

workflow.add_step(WorkflowStep(
    id="analyze_b",
    tool_name="analyze_method_b",
    parameters={"data": "$transform"},
    depends_on={"transform"}
))

workflow.add_step(WorkflowStep(
    id="merge_results",
    tool_name="merge_analyses",
    parameters={"a": "$analyze_a", "b": "$analyze_b"},
    depends_on={"analyze_a", "analyze_b"}
))

# Visualize
print(workflow.visualize())

# Execute
result = workflow.execute()
```

---

## Error Recovery

### Retry with Compensation

```python
from typing import Optional, Callable

@dataclass
class CompensatingAction:
    """Action to undo a step's effects."""
    tool_name: str
    parameters: Dict[str, Any]

@dataclass
class RecoverableStep:
    """Workflow step with compensation action."""
    id: str
    tool_name: str
    parameters: Dict[str, Any]
    compensation: Optional[CompensatingAction] = None
    max_retries: int = 3
    depends_on: Set[str] = None

class SagaWorkflowExecutor:
    """
    Execute workflow with saga pattern (compensation on failure).
    """
    
    def __init__(self, registry):
        self.registry = registry
        self.context: Dict[str, Any] = {}
        self.executed_steps: List[str] = []
    
    def execute_workflow(self, steps: List[RecoverableStep]) -> ChainResult:
        """
        Execute workflow with automatic compensation on failure.
        """
        self.context = {}
        self.executed_steps = []
        
        try:
            return self._execute_steps(steps)
        except Exception as e:
            # Rollback executed steps
            print(f"Workflow failed: {e}")
            print("Rolling back...")
            self._rollback()
            
            return ChainResult(
                success=False,
                outputs=self.context.copy(),
                error=str(e)
            )
    
    def _execute_steps(self, steps: List[RecoverableStep]) -> ChainResult:
        """Execute steps with retry logic."""
        step_map = {step.id: step for step in steps}
        completed = set()
        
        while len(completed) < len(steps):
            ready_steps = [
                step for step in steps
                if step.id not in completed
                and (step.depends_on or set()).issubset(completed)
            ]
            
            if not ready_steps:
                raise Exception("Cannot make progress - possible circular dependency")
            
            for step in ready_steps:
                # Execute with retry
                success = False
                last_error = None
                
                for attempt in range(step.max_retries):
                    try:
                        resolved_params = self._resolve_parameters(step.parameters)
                        result = self.registry.execute(
                            step.tool_name,
                            **resolved_params
                        )
                        
                        self.context[step.id] = result
                        self.executed_steps.append(step.id)
                        completed.add(step.id)
                        success = True
                        break
                        
                    except Exception as e:
                        last_error = e
                        if attempt < step.max_retries - 1:
                            print(f"Retry {attempt + 1}/{step.max_retries} for {step.id}")
                
                if not success:
                    raise Exception(f"Step '{step.id}' failed after {step.max_retries} retries: {last_error}")
        
        return ChainResult(
            success=True,
            outputs=self.context.copy()
        )
    
    def _rollback(self):
        """Execute compensation actions in reverse order."""
        for step_id in reversed(self.executed_steps):
            step = next(s for s in steps if s.id == step_id)
            
            if step.compensation:
                try:
                    print(f"Compensating {step_id}...")
                    resolved_params = self._resolve_parameters(
                        step.compensation.parameters
                    )
                    self.registry.execute(
                        step.compensation.tool_name,
                        **resolved_params
                    )
                except Exception as e:
                    print(f"Compensation failed for {step_id}: {e}")
    
    def _resolve_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve parameters from context."""
        resolved = {}
        for key, value in params.items():
            if isinstance(value, str) and value.startswith("$"):
                parts = value[1:].split(".", 1)
                step_id = parts[0]
                step_output = self.context.get(step_id, {})
                
                if len(parts) == 2:
                    resolved[key] = step_output.get(parts[1])
                else:
                    resolved[key] = step_output
            else:
                resolved[key] = value
        
        return resolved

# Example: E-commerce order workflow with compensation
steps = [
    RecoverableStep(
        id="reserve_inventory",
        tool_name="reserve_items",
        parameters={"items": ["item1", "item2"]},
        compensation=CompensatingAction(
            tool_name="release_reservation",
            parameters={"reservation_id": "$reserve_inventory.id"}
        ),
        max_retries=3
    ),
    RecoverableStep(
        id="charge_payment",
        tool_name="process_payment",
        parameters={"amount": 100, "method": "credit_card"},
        compensation=CompensatingAction(
            tool_name="refund_payment",
            parameters={"transaction_id": "$charge_payment.id"}
        ),
        depends_on={"reserve_inventory"},
        max_retries=3
    ),
    RecoverableStep(
        id="ship_order",
        tool_name="create_shipment",
        parameters={"order_id": "$charge_payment.order_id"},
        compensation=CompensatingAction(
            tool_name="cancel_shipment",
            parameters={"shipment_id": "$ship_order.id"}
        ),
        depends_on={"charge_payment"},
        max_retries=3
    )
]

executor = SagaWorkflowExecutor(registry)
result = executor.execute_workflow(steps)
```

---

## Workflow Monitoring

### Real-Time Monitoring

```python
from datetime import datetime
from typing import Dict, List
import json

@dataclass
class StepExecution:
    """Record of a single step execution."""
    step_id: str
    tool_name: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    success: bool = False
    error: Optional[str] = None
    duration_ms: Optional[float] = None

class WorkflowMonitor:
    """
    Monitor workflow execution with real-time metrics.
    """
    
    def __init__(self):
        self.executions: Dict[str, List[StepExecution]] = {}
        self.active_workflows: Set[str] = set()
    
    def start_workflow(self, workflow_id: str):
        """Start monitoring a workflow."""
        self.executions[workflow_id] = []
        self.active_workflows.add(workflow_id)
    
    def record_step_start(
        self,
        workflow_id: str,
        step_id: str,
        tool_name: str
    ) -> StepExecution:
        """Record the start of a step."""
        execution = StepExecution(
            step_id=step_id,
            tool_name=tool_name,
            started_at=datetime.utcnow()
        )
        
        if workflow_id not in self.executions:
            self.executions[workflow_id] = []
        
        self.executions[workflow_id].append(execution)
        return execution
    
    def record_step_complete(
        self,
        workflow_id: str,
        step_id: str,
        success: bool,
        error: Optional[str] = None
    ):
        """Record the completion of a step."""
        executions = self.executions.get(workflow_id, [])
        
        # Find most recent execution of this step
        for execution in reversed(executions):
            if execution.step_id == step_id and execution.completed_at is None:
                execution.completed_at = datetime.utcnow()
                execution.success = success
                execution.error = error
                execution.duration_ms = (
                    execution.completed_at - execution.started_at
                ).total_seconds() * 1000
                break
    
    def end_workflow(self, workflow_id: str):
        """Mark workflow as completed."""
        self.active_workflows.discard(workflow_id)
    
    def get_workflow_summary(self, workflow_id: str) -> Dict:
        """Get summary statistics for a workflow."""
        executions = self.executions.get(workflow_id, [])
        
        if not executions:
            return {"error": "Workflow not found"}
        
        total_steps = len(executions)
        successful_steps = sum(1 for e in executions if e.success)
        failed_steps = sum(1 for e in executions if not e.success and e.completed_at)
        
        completed_executions = [e for e in executions if e.duration_ms is not None]
        
        return {
            "workflow_id": workflow_id,
            "total_steps": total_steps,
            "successful": successful_steps,
            "failed": failed_steps,
            "in_progress": total_steps - successful_steps - failed_steps,
            "total_duration_ms": sum(e.duration_ms for e in completed_executions),
            "avg_step_duration_ms": (
                sum(e.duration_ms for e in completed_executions) / len(completed_executions)
                if completed_executions else 0
            ),
            "is_active": workflow_id in self.active_workflows
        }
    
    def get_step_timeline(self, workflow_id: str) -> List[Dict]:
        """Get timeline of step executions."""
        executions = self.executions.get(workflow_id, [])
        
        timeline = []
        for execution in executions:
            timeline.append({
                "step_id": execution.step_id,
                "tool_name": execution.tool_name,
                "started_at": execution.started_at.isoformat(),
                "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
                "duration_ms": execution.duration_ms,
                "success": execution.success,
                "error": execution.error
            })
        
        return timeline

# Integrate monitoring with workflow executor
class MonitoredWorkflowExecutor:
    """Workflow executor with integrated monitoring."""
    
    def __init__(self, registry):
        self.registry = registry
        self.monitor = WorkflowMonitor()
    
    def execute_workflow(
        self,
        workflow_id: str,
        steps: List[WorkflowStep]
    ) -> ChainResult:
        """Execute workflow with monitoring."""
        self.monitor.start_workflow(workflow_id)
        context = {}
        completed = set()
        
        try:
            step_map = {step.id: step for step in steps}
            
            while len(completed) < len(steps):
                ready_steps = [
                    step for step in steps
                    if step.id not in completed
                    and (step.depends_on or set()).issubset(completed)
                ]
                
                if not ready_steps:
                    raise Exception("Cannot make progress")
                
                for step in ready_steps:
                    # Record start
                    self.monitor.record_step_start(
                        workflow_id,
                        step.id,
                        step.tool_name
                    )
                    
                    try:
                        # Execute
                        resolved_params = self._resolve_parameters(step.parameters, context)
                        result = self.registry.execute(
                            step.tool_name,
                            **resolved_params
                        )
                        
                        context[step.id] = result
                        completed.add(step.id)
                        
                        # Record success
                        self.monitor.record_step_complete(
                            workflow_id,
                            step.id,
                            success=True
                        )
                        
                    except Exception as e:
                        # Record failure
                        self.monitor.record_step_complete(
                            workflow_id,
                            step.id,
                            success=False,
                            error=str(e)
                        )
                        raise
            
            return ChainResult(
                success=True,
                outputs=context.copy()
            )
            
        finally:
            self.monitor.end_workflow(workflow_id)
    
    def _resolve_parameters(self, params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve parameters from context."""
        resolved = {}
        for key, value in params.items():
            if isinstance(value, str) and value.startswith("$"):
                parts = value[1:].split(".", 1)
                step_id = parts[0]
                step_output = context.get(step_id, {})
                
                if len(parts) == 2:
                    resolved[key] = step_output.get(parts[1])
                else:
                    resolved[key] = step_output
            else:
                resolved[key] = value
        
        return resolved
    
    def get_workflow_summary(self, workflow_id: str) -> Dict:
        """Get workflow summary."""
        return self.monitor.get_workflow_summary(workflow_id)

# Usage
executor = MonitoredWorkflowExecutor(registry)
result = executor.execute_workflow("workflow-123", steps)

# Get metrics
summary = executor.get_workflow_summary("workflow-123")
print(json.dumps(summary, indent=2))
```

---

## Advanced Patterns

### Dynamic Workflow Generation

```python
class DynamicWorkflowBuilder:
    """
    Build workflows dynamically based on LLM decisions.
    """
    
    def __init__(self, registry, openai_client):
        self.registry = registry
        self.client = openai_client
    
    def generate_workflow(self, user_goal: str) -> List[WorkflowStep]:
        """
        Use LLM to generate workflow steps for a goal.
        """
        # Get available tools
        available_tools = self.registry.get_all_openai_schemas()
        
        # Ask LLM to plan workflow
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a workflow planner. Given a user goal and available tools, "
                        "create a sequence of tool calls to achieve the goal. "
                        "Respond with JSON array of steps with: id, tool_name, parameters, depends_on."
                    )
                },
                {
                    "role": "user",
                    "content": f"Goal: {user_goal}\n\nAvailable tools: {json.dumps(available_tools, indent=2)}"
                }
            ],
            response_format={"type": "json_object"}
        )
        
        # Parse workflow steps
        workflow_data = json.loads(response.choices[0].message.content)
        steps = []
        
        for step_data in workflow_data.get("steps", []):
            step = WorkflowStep(
                id=step_data["id"],
                tool_name=step_data["tool_name"],
                parameters=step_data.get("parameters", {}),
                depends_on=set(step_data.get("depends_on", []))
            )
            steps.append(step)
        
        return steps

# Example
builder = DynamicWorkflowBuilder(registry, openai_client)

workflow = builder.generate_workflow(
    "Get weather for London, and if it's raining, send me an email alert"
)

# Execute generated workflow
executor = MonitoredWorkflowExecutor(registry)
result = executor.execute_workflow("dynamic-workflow", workflow)
```

---

## Summary

### Key Takeaways

1. **Tool chaining** connects multiple tools into sequences and pipelines
2. **Conditional execution** enables branching logic based on results
3. **State management** allows pause/resume and failure recovery
4. **DAG workflows** optimize execution with parallelization
5. **Error recovery** uses compensation actions (saga pattern)
6. **Monitoring** provides visibility into workflow execution
7. **Dynamic workflows** can be generated by LLMs based on goals

### Workflow Design Checklist

- [ ] Identify tool dependencies (which outputs feed which inputs)
- [ ] Define conditional branches for different scenarios
- [ ] Implement state persistence for long-running workflows
- [ ] Add compensation actions for critical steps
- [ ] Set up monitoring and logging
- [ ] Handle failures gracefully with retries
- [ ] Optimize with parallel execution where possible
- [ ] Test edge cases (failures, timeouts, missing data)
- [ ] Document workflow behavior and error handling
- [ ] Set up alerts for workflow failures

### Next Steps

- **Lesson 4:** Real-World Tool Integrations
- **Lab 2:** Building a Tool Registry System
- **Lab 3:** Multi-Tool Workflow Orchestration

---

## Additional Resources

- Temporal Workflows: https://temporal.io/
- Apache Airflow: https://airflow.apache.org/
- AWS Step Functions: https://aws.amazon.com/step-functions/
- Saga Pattern: https://microservices.io/patterns/data/saga.html
- DAG Processing: https://en.wikipedia.org/wiki/Directed_acyclic_graph

**Last Updated:** November 14, 2025
