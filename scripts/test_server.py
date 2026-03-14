#!/usr/bin/env python3
"""Quick self-check for a running ManipArena model server.

Tests:
1) WebSocket handshake + metadata validation
2) Send a dummy Desktop observation → validate response schema

Usage:
    python scripts/test_server.py                         # default ws://127.0.0.1:8000
    python scripts/test_server.py --uri ws://host:port
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
from typing import Any

import cv2
import msgpack
import msgpack_numpy as m
import numpy as np
import websockets

m.patch()


# ── Helpers ───────────────────────────────────────────────────────


def _encode_jpeg(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        raise ValueError("cv2.imencode failed")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _validate_metadata(meta: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key, typ in {"control_mode": str, "action_horizon": int, "state_dim": int}.items():
        if key not in meta:
            errors.append(f"missing key: {key}")
        elif not isinstance(meta[key], typ):
            errors.append(f"{key}: expected {typ.__name__}, got {type(meta[key]).__name__}")
    if meta.get("control_mode") not in ("joints", "end_pose", None):
        errors.append(f"control_mode={meta.get('control_mode')!r} not in (joints, end_pose)")
    if isinstance(meta.get("action_horizon"), int) and meta["action_horizon"] <= 0:
        errors.append("action_horizon must be > 0")
    return errors


def _validate_trajectory(obj: Any, key: str, dim: int) -> list[str]:
    errors: list[str] = []
    if not isinstance(obj, list):
        return [f"{key} must be List[List[float]], got {type(obj).__name__}"]
    if len(obj) == 0:
        return [f"{key} is empty"]
    for i, row in enumerate(obj):
        if not isinstance(row, list):
            errors.append(f"{key}[{i}] must be list, got {type(row).__name__}")
        elif len(row) != dim:
            errors.append(f"{key}[{i}] expected dim={dim}, got {len(row)}")
        if i > 2 and errors:
            errors.append(f"  ... (showing first few)")
            break
    return errors


def _build_desktop_payload() -> dict[str, Any]:
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    jpeg = _encode_jpeg(img)
    return {
        "state": {
            "follow1_pos": [0.1, 0.2, 0.3, 0.0, 0.1, 0.2, 0.5],
            "follow2_pos": [0.1, -0.2, 0.3, 0.0, -0.1, 0.2, 0.5],
        },
        "views": {
            "camera_left": jpeg,
            "camera_front": jpeg,
            "camera_right": jpeg,
        },
        "instruction": "self-check",
    }


# ── Main ──────────────────────────────────────────────────────────


async def run(uri: str, timeout: float) -> int:
    failed = 0

    # Step 1: connect + metadata
    print(f"[TEST] Connecting to {uri}")
    try:
        ws = await websockets.connect(uri, open_timeout=timeout)
    except Exception as exc:
        print(f"[FAIL] Cannot connect: {exc}")
        return 1

    first = await asyncio.wait_for(ws.recv(), timeout=timeout)
    if isinstance(first, str):
        print(f"[FAIL] Expected msgpack metadata, got text: {first!r}")
        return 1
    meta = msgpack.unpackb(first, raw=False)
    if not isinstance(meta, dict):
        print(f"[FAIL] Metadata is not a dict")
        return 1

    errs = _validate_metadata(meta)
    if errs:
        print("[FAIL] Metadata:")
        for e in errs:
            print(f"  - {e}")
        failed += 1
    else:
        print(f"[PASS] Metadata: {json.dumps(meta, default=str)}")

    # Step 2: inference round-trip
    print("[TEST] Sending dummy Desktop observation...")
    payload = _build_desktop_payload()
    await ws.send(msgpack.packb(payload, use_bin_type=True))
    msg = await asyncio.wait_for(ws.recv(), timeout=timeout)
    if isinstance(msg, str):
        print(f"[FAIL] Server returned error: {msg}")
        failed += 1
    else:
        resp = msgpack.unpackb(msg, raw=False)
        errs = []
        for key in ("follow1_pos", "follow2_pos"):
            if key not in resp:
                errs.append(f"missing key: {key}")
            else:
                errs.extend(_validate_trajectory(resp[key], key, dim=7))
        if errs:
            print("[FAIL] Response schema:")
            for e in errs:
                print(f"  - {e}")
            failed += 1
        else:
            horizon = len(resp["follow1_pos"])
            print(f"[PASS] Response: follow1_pos({horizon}x7), follow2_pos({horizon}x7)")

    await ws.close()
    if failed == 0:
        print("[PASS] All checks passed.")
    else:
        print(f"[FAIL] {failed} check(s) failed.")
    return failed


def main():
    p = argparse.ArgumentParser(description="Self-check a running ManipArena server.")
    p.add_argument("--uri", default="ws://127.0.0.1:8000", help="Server WebSocket URI")
    p.add_argument("--timeout", type=float, default=10.0, help="Recv timeout (seconds)")
    args = p.parse_args()
    raise SystemExit(asyncio.run(run(args.uri, args.timeout)))


if __name__ == "__main__":
    main()
