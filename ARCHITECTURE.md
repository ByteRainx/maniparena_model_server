# Architecture Notes

> This document is for developers who want to understand design decisions.  
> For daily usage, start with `README.md` and `QUICKSTART.md`.

## System Overview

### 1) Module boundaries

```text
┌─────────────────────────────────────────┐
│ x2robot_client                          │
│ - Collect sensor data (images, state)  │
│ - Send observations via WebSocket       │
│ - Receive and execute actions           │
└──────────────┬──────────────────────────┘
               │ WebSocket (msgpack)
               ↓
┌─────────────────────────────────────────┐
│ websocket_server.py                     │
│ - Manage WebSocket connections          │
│ - Send metadata                         │
│ - Receive observation, call infer()     │
│ - Return action output                  │
└──────────────┬──────────────────────────┘
               │ call
               ↓
┌─────────────────────────────────────────┐
│ policy_base.py                          │
│ - Defines load/convert/infer interface  │
│ - Implements standard pipeline          │
│   convert_input → run_inference →       │
│   convert_output                        │
└──────────────┬──────────────────────────┘
               │ inherit
               ↓
┌─────────────────────────────────────────┐
│ my_policy.py                            │
│ - load_model()                          │
│ - convert_input() (optional override)   │
│ - convert_output()                      │
│ - run_inference()                       │
└─────────────────────────────────────────┘
```

### 2) Data flow

```text
client sends observation
    ↓
convert_input()
    ↓
model input format
    ↓
run_inference()
    ↓
raw model output
    ↓
convert_output()
    ↓
server sends action response
```

## Design Principles

### Modularity
- `websocket_server.py`: transport only, no model logic
- `policy_base.py`: shared policy contract and common flow
- `my_policy.py`: participant-owned model logic only

### Simplicity
- Keep startup and control flow straightforward
- Keep each file focused on one responsibility

### Extensibility
- Extend policy behavior by overriding conversion/inference methods
- Add optional output fields (head/lift/chassis) for CX001 as needed
- Support both `joints` and `end_pose` modes

### Framework-neutral
- No hard dependency on a specific model framework
- Compatible with any backend that can produce expected action tensors

## Key Decisions

### Why a base policy class?

```python
class ModelPolicy(ABC):
    def infer(self, obs):
        model_input = self.convert_input(obs)
        model_output = self.run_inference(model_input)
        return self.convert_output(model_output)
```

Benefits:
- Participants do not reimplement pipeline boilerplate
- Only model-specific parts need editing
- Easier testing and debugging

### Why separate conversion logic?

```python
def convert_input(self, obs):
    pass

def convert_output(self, model_output):
    pass
```

Benefits:
- Your model can keep its own internal data schema
- Conversion code is isolated and reusable
- Easier protocol adaptation

### Why WebSocket + msgpack?

- Real-time bidirectional communication
- Efficient binary transport
- Works well with image and array payloads

## Extension Points

1. Add new control modes
2. Add new input/output fields
3. Add policy state reset and lifecycle hooks
4. Add richer error handling and recovery

## Summary

This design provides:
1. Clear responsibility boundaries
2. Minimal participant implementation burden
3. Framework-agnostic integration
4. Easy extension for task-specific needs

## Related Docs

- `README.md`: complete usage and protocol guide
- `QUICKSTART.md`: minimal setup guide
