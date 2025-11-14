# Workflow Patterns Cheatsheet

Quick reference for common tool orchestration patterns.

## Table of Contents
- [Sequential Patterns](#sequential-patterns)
- [Parallel Patterns](#parallel-patterns)
- [Conditional Patterns](#conditional-patterns)
- [Loop Patterns](#loop-patterns)
- [Error Handling Patterns](#error-handling-patterns)
- [State Management](#state-management)
- [Advanced Patterns](#advanced-patterns)

---

## Sequential Patterns

### Simple Chain

Execute tools in sequence, passing data forward.

```
A → B → C
```

```python
def sequential_chain(steps: List[str]) -> dict:
    """Execute steps sequentially."""
    context = {}
    
    for step in steps:
        result = execute_tool(step, context)
        context[step] = result
    
    return context
```

**Use cases:**
- User lookup → Fetch data → Format response
- Validate → Process → Store
- Extract → Transform → Load (ETL)

### Parameter Resolution Chain

Pass specific outputs as inputs to next tool.

```
A.output → B.input → C.input
```

```python
workflow = [
    {"tool": "get_user", "params": {"user_id": "123"}, "output": "user"},
    {"tool": "get_orders", "params": {"user_id": "{user.id}"}, "output": "orders"},
    {"tool": "calculate_total", "params": {"orders": "{orders}"}, "output": "total"}
]
```

**Use cases:**
- Drill-down queries
- Progressive data enrichment
- Multi-step calculations

---

## Parallel Patterns

### Fork-Join

Execute independent tasks in parallel, then combine results.

```
     ┌→ B →┐
A →  ┼→ C →┼→ E
     └→ D →┘
```

```python
async def fork_join(tasks: List[Callable]) -> dict:
    """Execute tasks in parallel."""
    results = await asyncio.gather(*[task() for task in tasks])
    return {"results": results}
```

**Use cases:**
- Fetch user profile + orders + reviews simultaneously
- Multi-source data aggregation
- Parallel API calls

### Map-Reduce

Apply same operation to multiple items in parallel.

```
Items: [1, 2, 3, 4]
Map: [f(1), f(2), f(3), f(4)] (parallel)
Reduce: combine results
```

```python
async def map_reduce(items: List, func: Callable, reducer: Callable):
    """Map-reduce pattern."""
    # Map phase (parallel)
    results = await asyncio.gather(*[func(item) for item in items])
    
    # Reduce phase
    return reducer(results)

# Usage
user_ids = ["1", "2", "3"]
profiles = await map_reduce(
    user_ids,
    get_user_profile,
    lambda results: {"profiles": results, "count": len(results)}
)
```

**Use cases:**
- Batch processing
- Multi-user operations
- Aggregating data from multiple sources

### Scatter-Gather

Send request to multiple services, use first/best response.

```
     ┌→ ServiceA →┐
Req ┼→ ServiceB →┼→ First/Best
     └→ ServiceC →┘
```

```python
async def scatter_gather(services: List[Callable], use_first: bool = True):
    """Call multiple services, use first or best."""
    if use_first:
        # Return first successful result
        tasks = [service() for service in services]
        for coro in asyncio.as_completed(tasks):
            result = await coro
            if result.get("success"):
                return result
    else:
        # Get all results, pick best
        results = await asyncio.gather(*[service() for service in services])
        return max(results, key=lambda r: r.get("score", 0))
```

**Use cases:**
- Multi-provider APIs (use fastest)
- Redundant services
- Competitive queries

---

## Conditional Patterns

### If-Then-Else

Branch based on condition.

```
     ┌→ B (if condition)
A →  ┤
     └→ C (else)
```

```python
def conditional_workflow(condition: Callable, if_true: Callable, if_false: Callable):
    """Execute based on condition."""
    result_a = step_a()
    
    if condition(result_a):
        return if_true(result_a)
    else:
        return if_false(result_a)
```

**Use cases:**
- Inventory check → Order/Notify
- Authentication → Protected/Public action
- Validation → Process/Reject

### Switch-Case

Multiple conditional branches.

```
     ┌→ B (case 1)
A →  ┼→ C (case 2)
     ┼→ D (case 3)
     └→ E (default)
```

```python
def switch_workflow(value: str, cases: Dict[str, Callable], default: Callable):
    """Switch-case pattern."""
    handler = cases.get(value, default)
    return handler()
```

**Use cases:**
- Status-based routing
- Type-specific processing
- Priority handling

### Guard Pattern

Skip steps based on preconditions.

```
A → [Guard] → B → [Guard] → C
    ↓ skip          ↓ skip
    End            End
```

```python
def guarded_workflow(steps: List[tuple[Callable, Callable]]):
    """Execute steps with guards."""
    context = {}
    
    for step, guard in steps:
        if guard(context):
            result = step(context)
            context.update(result)
        else:
            break
    
    return context
```

**Use cases:**
- Permission checks before actions
- Prerequisite validation
- Progressive disclosure

---

## Loop Patterns

### While Loop

Repeat until condition met.

```
A → [condition?] ─Yes→ B → (back to A)
         │
         No
         ↓
         End
```

```python
def while_loop(condition: Callable, action: Callable, max_iterations: int = 10):
    """While loop pattern."""
    context = {}
    iteration = 0
    
    while condition(context) and iteration < max_iterations:
        result = action(context)
        context.update(result)
        iteration += 1
    
    return context
```

**Use cases:**
- Polling until ready
- Retry with condition
- Progressive refinement

### For-Each Loop

Process each item in collection.

```
Items: [A, B, C]
Process A → Process B → Process C
```

```python
def for_each(items: List, processor: Callable) -> List:
    """Process each item."""
    results = []
    for item in items:
        result = processor(item)
        results.append(result)
    return results
```

**Use cases:**
- Batch updates
- Multi-item validation
- Sequential processing

### Pagination Loop

Process data in pages.

```
Page 1 → Page 2 → ... → Page N
  ↓        ↓              ↓
Process  Process       Process
```

```python
async def paginated_workflow(fetch_page: Callable, process: Callable):
    """Process paginated data."""
    page = 1
    all_results = []
    
    while True:
        data = await fetch_page(page)
        
        if not data or len(data) == 0:
            break
        
        results = await process(data)
        all_results.extend(results)
        page += 1
    
    return all_results
```

**Use cases:**
- Large dataset processing
- API pagination
- Incremental loading

---

## Error Handling Patterns

### Try-Catch-Finally

Handle errors with cleanup.

```
Try
  ├→ Step A
  ├→ Step B
  └→ Step C
Catch Error
  └→ Handle
Finally
  └→ Cleanup
```

```python
def try_catch_finally(
    try_steps: List[Callable],
    catch_handler: Callable,
    finally_handler: Callable
):
    """Try-catch-finally pattern."""
    try:
        for step in try_steps:
            step()
    except Exception as e:
        catch_handler(e)
    finally:
        finally_handler()
```

### Retry with Backoff

Retry failed operations.

```
Try → Fail → Wait → Retry → Fail → Wait → Retry → Success
```

```python
def retry_pattern(
    func: Callable,
    max_attempts: int = 3,
    backoff: float = 2.0
):
    """Retry with exponential backoff."""
    for attempt in range(max_attempts):
        try:
            return func()
        except Exception as e:
            if attempt == max_attempts - 1:
                raise
            time.sleep(backoff ** attempt)
```

### Fallback Chain

Try alternatives on failure.

```
Primary → Fail → Secondary → Fail → Cache → Fail → Error
```

```python
def fallback_chain(strategies: List[Callable]):
    """Try strategies until one succeeds."""
    for strategy in strategies:
        try:
            result = strategy()
            if result.get("success"):
                return result
        except Exception:
            continue
    
    return {"success": False, "error": "All strategies failed"}
```

### Saga Pattern

Compensate on failure.

```
A → B → C → Fail
    ↓   ↓
   Undo Undo (compensate)
```

```python
def saga_pattern(steps: List[tuple[Callable, Callable]]):
    """Execute with compensation."""
    completed = []
    
    try:
        for action, compensation in steps:
            result = action()
            completed.append((result, compensation))
        return {"success": True}
    except Exception as e:
        # Rollback
        for result, compensation in reversed(completed):
            compensation(result)
        return {"success": False, "error": str(e)}
```

---

## State Management

### State Machine

Transition between states based on events.

```
[Idle] --start--> [Running] --complete--> [Done]
                     │
                     └--error--> [Failed]
```

```python
class WorkflowState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"

class StateMachine:
    def __init__(self):
        self.state = WorkflowState.IDLE
        self.transitions = {
            WorkflowState.IDLE: {"start": WorkflowState.RUNNING},
            WorkflowState.RUNNING: {
                "complete": WorkflowState.DONE,
                "error": WorkflowState.FAILED
            }
        }
    
    def transition(self, event: str):
        """Transition to new state."""
        allowed = self.transitions.get(self.state, {})
        new_state = allowed.get(event)
        
        if new_state:
            self.state = new_state
        else:
            raise ValueError(f"Invalid transition: {self.state} --{event}-->")
```

**Use cases:**
- Order processing states
- Approval workflows
- Task lifecycle management

### Checkpoint/Resume

Save progress and resume from checkpoint.

```
A → B → [Checkpoint] → C → Fail
                       ↓
                    Resume from checkpoint
```

```python
class CheckpointWorkflow:
    def __init__(self):
        self.checkpoints = {}
    
    def execute_with_checkpoints(self, steps: List[Callable], workflow_id: str):
        """Execute with checkpointing."""
        start_index = self.checkpoints.get(workflow_id, 0)
        
        for i, step in enumerate(steps[start_index:], start=start_index):
            try:
                result = step()
                self.checkpoints[workflow_id] = i + 1
            except Exception as e:
                return {"error": str(e), "checkpoint": i}
        
        del self.checkpoints[workflow_id]
        return {"success": True}
```

**Use cases:**
- Long-running workflows
- Expensive operations
- Failure recovery

### Event Sourcing

Store events, rebuild state from events.

```
Events: [Created, Updated, Approved]
State = replay(events)
```

```python
class EventSourcedWorkflow:
    def __init__(self):
        self.events = []
    
    def record_event(self, event: dict):
        """Record event."""
        self.events.append({
            **event,
            "timestamp": datetime.now()
        })
    
    def replay_state(self) -> dict:
        """Rebuild state from events."""
        state = {}
        for event in self.events:
            self._apply_event(state, event)
        return state
    
    def _apply_event(self, state: dict, event: dict):
        """Apply event to state."""
        event_type = event["type"]
        if event_type == "created":
            state["id"] = event["id"]
        elif event_type == "updated":
            state.update(event["data"])
```

---

## Advanced Patterns

### DAG (Directed Acyclic Graph)

Dependencies form a graph, execute in optimal order.

```
     ┌→ B →┐
A →  ┤     ├→ E
     ├→ C →┤
     └→ D →┘
```

```python
import networkx as nx

def execute_dag(graph: nx.DiGraph):
    """Execute DAG in topological order."""
    # Get execution order
    for level in nx.topological_generations(graph):
        # Execute level in parallel
        results = await asyncio.gather(*[
            execute_node(graph, node) for node in level
        ])
```

**Use cases:**
- Complex workflows with dependencies
- Optimal parallelization
- Build systems

### Dynamic Workflow

Generate workflow based on runtime data.

```
Input → Analyze → Generate Plan → Execute Plan
```

```python
def dynamic_workflow(input_data: dict):
    """Generate workflow dynamically."""
    # Analyze input
    analysis = analyze_input(input_data)
    
    # Generate plan
    if analysis["type"] == "simple":
        plan = [step_a, step_b]
    else:
        plan = [step_x, step_y, step_z]
    
    # Execute plan
    return execute_plan(plan, input_data)
```

**Use cases:**
- User-driven workflows
- Context-dependent processing
- Adaptive systems

### Pipeline

Stream processing with transformations.

```
Input → Transform1 → Transform2 → Transform3 → Output
```

```python
from typing import Iterator

def pipeline(data: Iterator, *transforms: Callable) -> Iterator:
    """Pipeline pattern."""
    result = data
    for transform in transforms:
        result = transform(result)
    return result

# Usage
data = [1, 2, 3, 4, 5]
result = pipeline(
    data,
    lambda x: (i * 2 for i in x),  # Double
    lambda x: (i + 1 for i in x),  # Add 1
    lambda x: (i for i in x if i > 5)  # Filter
)
```

**Use cases:**
- Data transformation
- Stream processing
- ETL pipelines

### Pub-Sub (Event-Driven)

Publish events, subscribers react.

```
Event → [Subscribers]
         ├→ Handler A
         ├→ Handler B
         └→ Handler C
```

```python
class EventBus:
    def __init__(self):
        self.subscribers = defaultdict(list)
    
    def subscribe(self, event_type: str, handler: Callable):
        """Subscribe to event."""
        self.subscribers[event_type].append(handler)
    
    def publish(self, event_type: str, data: dict):
        """Publish event to subscribers."""
        for handler in self.subscribers[event_type]:
            handler(data)

# Usage
bus = EventBus()
bus.subscribe("user_created", send_welcome_email)
bus.subscribe("user_created", log_user_creation)
bus.publish("user_created", {"user_id": "123"})
```

**Use cases:**
- Decoupled systems
- Real-time notifications
- Event-driven architectures

---

## Pattern Selection Guide

| Scenario | Recommended Pattern |
|----------|-------------------|
| Sequential steps with dependencies | Sequential Chain |
| Independent parallel tasks | Fork-Join |
| Process collection items | Map-Reduce |
| Conditional logic | If-Then-Else / Switch |
| Retry on failure | Retry with Backoff |
| Multiple fallback options | Fallback Chain |
| Rollback on failure | Saga Pattern |
| Long-running with failures | Checkpoint/Resume |
| Complex dependencies | DAG |
| Need audit trail | Event Sourcing |
| Real-time reactions | Pub-Sub |

---

## Anti-Patterns

### ❌ Callback Hell

Deeply nested callbacks.

```python
# Bad
def process():
    step_a(lambda result_a:
        step_b(result_a, lambda result_b:
            step_c(result_b, lambda result_c:
                step_d(result_c, lambda result_d:
                    final_handler(result_d)))))
```

**Better**: Use async/await or chain pattern

### ❌ Tight Coupling

Steps directly depend on each other's internals.

```python
# Bad
def step_b():
    # Directly accesses step_a's internals
    return step_a().internal_data.process()
```

**Better**: Pass data through interfaces

### ❌ No Error Handling

Workflows fail without recovery.

```python
# Bad
def workflow():
    step_a()  # No error handling
    step_b()
    step_c()
```

**Better**: Add try-catch and fallbacks

### ❌ Synchronous When Parallel Possible

Sequential execution of independent tasks.

```python
# Bad
result_a = fetch_data_a()  # 2s
result_b = fetch_data_b()  # 2s
# Total: 4s
```

**Better**: Use parallel execution (2s total)

---

## Quick Reference

### Basic Patterns
- **Sequential**: A → B → C
- **Parallel**: A + B + C → Combine
- **Conditional**: A → if X then B else C

### Error Handling
- **Retry**: Try → Retry → Retry → Give up
- **Fallback**: Primary → Secondary → Cache
- **Saga**: Do → Undo on failure

### Advanced
- **DAG**: Optimal parallel + sequential
- **Dynamic**: Generate plan at runtime
- **Event-Driven**: Pub-Sub pattern

---

## Related Resources

- [function-calling-best-practices.md](./function-calling-best-practices.md) — Overall best practices
- [error-handling-patterns.md](./error-handling-patterns.md) — Error handling
- `../lessons/03-tool-orchestration-and-workflows.md` — Detailed orchestration
- `../labs/lab-03-multi-tool-workflow-orchestration.ipynb` — Hands-on practice
