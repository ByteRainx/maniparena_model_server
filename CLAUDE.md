# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

WebSocket model server template for the **ManipArena CVPR Benchmark**. Participants implement their policy in `my_policy.py`; the server handles all WebSocket/msgpack communication with the `x2robot_client` evaluation robot.

## Commands

```bash
# Install
pip install -r requirements.txt

# Start server
python launch_server.py --checkpoint /path/to/ckpt --control-mode end_pose --port 8000

# Start OpenPI server (standalone, recommended)
python openpi_branch/serve_openpi.py \
    --config pi0_x2robot_pick_banana_merged_ee \
    --checkpoint-dir /path/to/checkpoints/step_10000 \
    --port 8000

# Self-check tools (run against a running server)
python tools/mock_ping.py --uri ws://127.0.0.1:8000
python tools/mock_schema_check.py --uri ws://127.0.0.1:8000 --mode both
python tools/mock_openloop_eval.py --uri ws://127.0.0.1:8000 --data-dir /path/to/dataset --save-dir /path/to/output
```

## Architecture

The pipeline is a strict chain:

```
x2robot_client → [WebSocket/msgpack] → websocket_server.py → policy.infer(obs)
                                                                  ↓
                                                        convert_input(obs)
                                                                  ↓
                                                        run_inference(model_input)
                                                                  ↓
                                                        convert_output(model_output)
                                                                  ↓
                                                       [WebSocket/msgpack] → client
```

**Files and editability:**

| File | Edit? | Role |
|------|-------|------|
| `my_policy.py` | **YES** | Participant's model: `load_model`, `run_inference`, `convert_input`, `convert_output` |
| `utils.py` | optional | I/O conversion helpers (`convert_observation_to_model_input`, `convert_model_output_to_x2robot_format`, `convert_model_output_to_legacy_format`) |
| `policy_base.py` | **NO** | Abstract `ModelPolicy` base class defining the `infer()` pipeline |
| `websocket_server.py` | **NO** | WebSocket server core (msgpack transport, connection handling) |
| `launch_server.py` | **NO** | CLI entry point using argparse |
| `openpi_branch/` | varies | OpenPI-specific serving server and policy (`serve_openpi.py` is standalone) |

## Two Task Types

- **Desktop (14D)**: EX001 dual-arm. Action = 14D (2×7D arms). Output uses **lowercase** keys (`follow1_pos`, `follow2_pos`). Use `convert_model_output_to_x2robot_format()`.
- **CX001 Mobile Manipulation (20D)**: 14D arms + 2D head + 1D lift + 3D chassis. Output uses **UPPERCASE** keys (`FOLLOW1_POS`, `FOLLOW2_POS`, `HEAD_POS`, `LIFT_OUT`, `CAR_POSE_OUT`). Use `convert_model_output_to_legacy_format()`.

## Critical Pitfalls

1. **`.tolist()` is mandatory** on all output arrays. The client does `[current_pos] + actions` — if `actions` is numpy, `+` silently broadcasts instead of concatenating, corrupting the trajectory.
2. **Key casing must match the target client.** Wrong casing causes `.get()` to return `None`/`[]` and the robot silently does nothing.
3. **Values must be `List[List[float]]`** (not flat lists, not numpy arrays).
4. **Observation formats differ**: Desktop sends nested dict with base64 JPEG images; CX001 sends flat dict with numpy RGB arrays and UPPERCASE keys. `convert_observation_to_model_input()` handles both.

## Protocol

1. On connect: server sends metadata via msgpack (`control_mode`, `action_horizon`, `state_dim`, `protocol_version`).
2. Loop: client sends observation (msgpack) → server returns action (msgpack).
