#!/usr/bin/env python3
"""Protocol/schema validator for ManipArena model server.

This script sends minimal valid requests and validates response schema:
- Desktop mode: lowercase keys (`follow1_pos`, `follow2_pos`)
- CX001 mode: uppercase keys (`FOLLOW1_POS`, `FOLLOW2_POS`, and optional MM extras)
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


def _encode_jpeg_base64(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        raise ValueError("cv2.imencode(.jpg) failed")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float, np.floating, np.integer))


def _validate_trajectory(
    obj: Any,
    key: str,
    dim: int,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(obj, list):
        return [f"{key} must be List[List[float]], got {type(obj).__name__}"]
    if len(obj) == 0:
        return [f"{key} is empty"]
    for i, row in enumerate(obj):
        if not isinstance(row, list):
            errors.append(f"{key}[{i}] must be list, got {type(row).__name__}")
            continue
        if len(row) != dim:
            errors.append(f"{key}[{i}] dim mismatch: expected {dim}, got {len(row)}")
            continue
        bad = [j for j, v in enumerate(row) if not _is_number(v)]
        if bad:
            errors.append(f"{key}[{i}] contains non-numeric values at indices {bad[:5]}")
    return errors


def _build_desktop_payload() -> dict[str, Any]:
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, :, 1] = 180
    jpeg = _encode_jpeg_base64(img)
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
        "instruction": "self-check desktop",
    }


def _build_cx001_payload() -> dict[str, Any]:
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, :, 2] = 200
    return {
        "CAMERA_LEFT": img,
        "CAMERA_FRONT": img,
        "CAMERA_RIGHT": img,
        "ACTION_FOLLOW1_POS": np.array([0.1, 0.2, 0.3, 0.0, 0.1, 0.2, 0.6], dtype=np.float32),
        "ACTION_FOLLOW2_POS": np.array([0.1, -0.2, 0.3, 0.0, -0.1, 0.2, 0.6], dtype=np.float32),
        "ACTION_FOLLOW1_JOINTS_CUR": np.zeros((7,), dtype=np.float32),
        "ACTION_FOLLOW2_JOINTS_CUR": np.zeros((7,), dtype=np.float32),
        "CAR_POSE": np.zeros((3,), dtype=np.float32),
        "LIFT": np.zeros((1,), dtype=np.float32),
        "HEAD_POS": np.zeros((2,), dtype=np.float32),
        "INSTRUCTION": "self-check cx001",
    }


def _validate_desktop_response(resp: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in ("follow1_pos", "follow2_pos"):
        if key not in resp:
            errors.append(f"missing key: {key}")
        else:
            errors.extend(_validate_trajectory(resp[key], key, dim=7))

    for wrong in ("FOLLOW1_POS", "FOLLOW2_POS"):
        if wrong in resp:
            errors.append(f"wrong key casing for Desktop: found {wrong}")
    return errors


def _validate_cx001_response(resp: dict[str, Any], require_mm_extras: bool) -> list[str]:
    errors: list[str] = []
    for key in ("FOLLOW1_POS", "FOLLOW2_POS"):
        if key not in resp:
            errors.append(f"missing key: {key}")
        else:
            errors.extend(_validate_trajectory(resp[key], key, dim=7))

    for wrong in ("follow1_pos", "follow2_pos"):
        if wrong in resp:
            errors.append(f"wrong key casing for CX001: found {wrong}")

    if require_mm_extras:
        extra_dims = {
            "HEAD_POS": 2,
            "LIFT_OUT": 1,
            "CAR_POSE_OUT": 3,
        }
        for key, dim in extra_dims.items():
            if key not in resp:
                errors.append(f"missing MM extra key: {key}")
            else:
                errors.extend(_validate_trajectory(resp[key], key, dim=dim))
    return errors


async def _send_and_recv(ws, payload: dict[str, Any], timeout_sec: float) -> dict[str, Any]:
    await ws.send(msgpack.packb(payload, use_bin_type=True))
    msg = await asyncio.wait_for(ws.recv(), timeout=timeout_sec)
    if isinstance(msg, str):
        raise RuntimeError(f"server returned text error: {msg}")
    resp = msgpack.unpackb(msg, raw=False)
    if not isinstance(resp, dict):
        raise RuntimeError(f"response is not dict: {type(resp).__name__}")
    return resp


async def run(uri: str, mode: str, timeout_sec: float, require_mm_extras: bool) -> int:
    print(f"[SCHEMA] Connecting to {uri} mode={mode}")
    async with websockets.connect(uri, open_timeout=timeout_sec) as ws:
        # First frame must be metadata
        first = await asyncio.wait_for(ws.recv(), timeout=timeout_sec)
        if isinstance(first, str):
            print(f"[FAIL] expected metadata msgpack, got text: {first}")
            return 1
        metadata = msgpack.unpackb(first, raw=False)
        print("[INFO] metadata:", json.dumps(metadata, ensure_ascii=False, default=str))

        checks: list[tuple[str, dict[str, Any], Any]] = []
        if mode in {"desktop", "both"}:
            checks.append(("desktop", _build_desktop_payload(), _validate_desktop_response))
        if mode in {"cx001", "both"}:
            checks.append(
                (
                    "cx001",
                    _build_cx001_payload(),
                    lambda r: _validate_cx001_response(r, require_mm_extras=require_mm_extras),
                )
            )

        failed = 0
        for name, payload, validator in checks:
            print(f"[CHECK] {name}")
            try:
                resp = await _send_and_recv(ws, payload, timeout_sec=timeout_sec)
                errors = validator(resp)
            except Exception as exc:  # noqa: BLE001
                print(f"[FAIL] {name}: {exc}")
                failed += 1
                continue

            if errors:
                failed += 1
                print(f"[FAIL] {name}:")
                for e in errors:
                    print(f"  - {e}")
            else:
                print(f"[PASS] {name}")

        if failed == 0:
            print("[PASS] All schema checks passed.")
            return 0
        print(f"[FAIL] {failed} check(s) failed.")
        return 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate request/response schema against server.")
    p.add_argument("--uri", type=str, default="ws://127.0.0.1:8000", help="Server WebSocket URI.")
    p.add_argument(
        "--mode",
        choices=["desktop", "cx001", "both"],
        default="both",
        help="Which schema set to validate.",
    )
    p.add_argument("--timeout-sec", type=float, default=8.0, help="Connect/recv timeout in seconds.")
    p.add_argument(
        "--require-mm-extras",
        action="store_true",
        help="For CX001 check, require HEAD_POS/LIFT_OUT/CAR_POSE_OUT.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    code = asyncio.run(
        run(
            uri=args.uri,
            mode=args.mode,
            timeout_sec=args.timeout_sec,
            require_mm_extras=args.require_mm_extras,
        )
    )
    raise SystemExit(code)


if __name__ == "__main__":
    main()

